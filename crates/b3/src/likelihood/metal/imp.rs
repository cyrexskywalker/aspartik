use anyhow::{Context, Result, anyhow};
use metal::{
	Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device,
	MTLResourceOptions, MTLSize,
};
use parking_lot::MutexGuard;

use std::env;
use std::mem;
use std::ptr;

use super::Calculator;
use crate::{Transitions, parameters::Tree};

const THREADS_PER_THREADGROUP_ENV: &str =
	"ASPARTIK_METAL_THREADS_PER_THREADGROUP";

#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct KernelParams {
	num_sites: u32,
	num_updated_nodes: u32,
	leaves_end: u32,
	scale_ln: u32,
	scale: f32,
	inv_scale: f32,
	_pad: [f32; 2],
	frequencies: [f32; 4],
}

pub struct MetalLikelihood {
	queue: CommandQueue,
	pipeline: ComputePipelineState,

	leaves: Buffer,
	projections: Buffer,
	projections_backup: Buffer,
	scales: Buffer,
	scales_backup: Buffer,
	scale_sums: Buffer,
	scale_sums_backup: Vec<u32>,
	likelihoods: Buffer,

	nodes: Buffer,
	children: Buffer,
	transitions: Buffer,
	params: Buffer,

	num_patterns: usize,
	pattern_weights: Vec<u32>,
	threads_per_threadgroup: u64,
	scale_ln: u32,
	scale: f32,
	inv_scale: f32,
}

impl MetalLikelihood {
	pub fn new(
		pattern_weights: Vec<u32>,
		leaves: Vec<u8>,
		scale_ln: u32,
	) -> Result<Self> {
		let num_patterns = pattern_weights.len();

		let device = Device::system_default()
			.ok_or_else(|| anyhow!("Metal device not found"))?;
		let queue = device.new_command_queue();

		let source = include_str!("kernels.metal");
		let options = CompileOptions::new();
		let library = device
			.new_library_with_source(source, &options)
			.map_err(anyhow::Error::msg)
			.context("failed to compile Metal library")?;
		let function = library
			.get_function("propose_kernel", None)
			.map_err(anyhow::Error::msg)
			.context("failed to get propose_kernel")?;
		let pipeline = device
			.new_compute_pipeline_state_with_function(&function)
			.map_err(anyhow::Error::msg)
			.context("failed to create compute pipeline")?;
		let threads_per_threadgroup = threads_per_threadgroup(&pipeline);

		let num_leaves = leaves.len() / num_patterns;
		let num_internals = num_leaves - 1;
		let num_nodes = num_leaves + num_internals;
		let options = MTLResourceOptions::StorageModeShared;

		let leaves = new_buffer(&device, &leaves, options);
		let projections = new_zeroed_buffer::<f32>(
			&device,
			num_nodes * num_patterns * 4,
			options,
		);
		let projections_backup = new_zeroed_buffer::<f32>(
			&device,
			num_nodes * num_patterns * 4,
			options,
		);
		let scales = new_zeroed_buffer::<u8>(
			&device,
			num_nodes * num_patterns,
			options,
		);
		let scales_backup = new_zeroed_buffer::<u8>(
			&device,
			num_nodes * num_patterns,
			options,
		);
		let scale_sums =
			new_zeroed_buffer::<u32>(&device, num_patterns, options);
		let likelihoods =
			new_zeroed_buffer::<f32>(&device, num_patterns, options);

		let nodes = new_zeroed_buffer::<u32>(&device, num_nodes, options);
		let children =
			new_zeroed_buffer::<u32>(&device, num_internals * 2, options);
		let transitions = new_zeroed_buffer::<[f32; 4]>(
			&device,
			num_internals * 2 * 4,
			options,
		);
		let params =
			new_zeroed_buffer::<KernelParams>(&device, 1, options);

		let scale = (-(scale_ln as f64)).exp() as f32;
		let inv_scale = 1.0f32 / scale;

		Ok(Self {
			queue,
			pipeline,
			leaves,
			projections,
			projections_backup,
			scales,
			scales_backup,
			scale_sums,
			scale_sums_backup: vec![0; num_patterns],
			likelihoods,
			nodes,
			children,
			transitions,
			params,
			num_patterns,
			pattern_weights,
			threads_per_threadgroup,
			scale_ln,
			scale,
			inv_scale,
		})
	}

	fn encode_propose(&self) {
		let command_buffer = self.queue.new_command_buffer();
		let encoder = command_buffer.new_compute_command_encoder();

		encoder.set_compute_pipeline_state(&self.pipeline);
		encoder.set_buffer(0, Some(&self.leaves), 0);
		encoder.set_buffer(1, Some(&self.projections), 0);
		encoder.set_buffer(2, Some(&self.scales), 0);
		encoder.set_buffer(3, Some(&self.scale_sums), 0);
		encoder.set_buffer(4, Some(&self.nodes), 0);
		encoder.set_buffer(5, Some(&self.children), 0);
		encoder.set_buffer(6, Some(&self.transitions), 0);
		encoder.set_buffer(7, Some(&self.likelihoods), 0);
		encoder.set_buffer(8, Some(&self.params), 0);

		let num_threads = self.num_patterns as u64 * 4;
		let threads_per_grid = MTLSize {
			width: num_threads,
			height: 1,
			depth: 1,
		};
		let threads_per_threadgroup = MTLSize {
			width: self.threads_per_threadgroup.min(num_threads),
			height: 1,
			depth: 1,
		};

		encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
		encoder.end_encoding();
		command_buffer.commit();
		command_buffer.wait_until_completed();
	}

	fn blit_copy(&self, src: &Buffer, dst: &Buffer) {
		let command_buffer = self.queue.new_command_buffer();
		let blit = command_buffer.new_blit_command_encoder();
		blit.copy_from_buffer(src, 0, dst, 0, src.length());
		blit.end_encoding();
		command_buffer.commit();
		command_buffer.wait_until_completed();
	}
}

impl Calculator<4, f64> for MetalLikelihood {
	fn likelihood(
		&mut self,
		mut tree: MutexGuard<Tree>,
		transitions: &Transitions<4, f64>,
	) -> Result<f64> {
		let (nodes, children, leaves_end) = tree.propagation_lists();
		let frequencies = transitions.frequencies();
		let matrices = transitions.matrices(&nodes[..nodes.len() - 1]);
		drop(tree);

		let nodes_u32: Vec<u32> = nodes.iter().map(|&n| n as u32).collect();
		let children_u32: Vec<u32> = children
			.iter()
			.flat_map(|&[l, r]| [l as u32, r as u32])
			.collect();
		let mut transitions_rows: Vec<[f32; 4]> =
			Vec::with_capacity(matrices.len() * 4);
		for tm in matrices {
			transitions_rows.push(cast_row(tm[0]));
			transitions_rows.push(cast_row(tm[1]));
			transitions_rows.push(cast_row(tm[2]));
			transitions_rows.push(cast_row(tm[3]));
		}

		unsafe {
			// SAFETY: `likelihood` has `&mut self`, so no concurrent host-side
			// access can go through this calculator. These buffers are populated on
			// the host before the next kernel launch.
			write_buffer(&self.nodes, &nodes_u32);
			write_buffer(&self.children, &children_u32);
			write_buffer(&self.transitions, &transitions_rows);
			write_buffer(
				&self.params,
				&[KernelParams {
					num_sites: self.num_patterns as u32,
					num_updated_nodes: (nodes.len() - 1) as u32,
					leaves_end: leaves_end as u32,
					scale_ln: self.scale_ln,
					scale: self.scale,
					inv_scale: self.inv_scale,
					_pad: [0.0; 2],
					frequencies: [
						frequencies[0] as f32,
						frequencies[1] as f32,
						frequencies[2] as f32,
						frequencies[3] as f32,
					],
				}],
			);
		}

		self.encode_propose();

		let mut likelihoods = vec![0f32; self.num_patterns];
		unsafe {
			// SAFETY: `encode_propose` waits for completion before returning, so the
			// device is no longer mutating these buffers when the host reads them.
			read_buffer(&self.likelihoods, &mut likelihoods);
		}

		let mut scale_sums = vec![0u32; self.num_patterns];
		unsafe {
			// SAFETY: same as for `likelihoods` above.
			read_buffer(&self.scale_sums, &mut scale_sums);
		}

		let mut sum = 0.0;
		for ((likelihood, scale), weight) in likelihoods
			.into_iter()
			.zip(scale_sums)
			.zip(&self.pattern_weights)
		{
			let weighted =
				(f64::from(likelihood) - f64::from(scale)) * f64::from(*weight);
			sum += weighted;
		}

		Ok(sum)
	}

	fn accept(&mut self) -> Result<()> {
		let mut scale_sums_backup = vec![0u32; self.num_patterns];
		unsafe {
			// SAFETY: `accept` has `&mut self`, and all prior command buffers are
			// synchronized before any host-side access to these buffers.
			read_buffer(&self.scale_sums, &mut scale_sums_backup);
		}
		self.scale_sums_backup = scale_sums_backup;
		self.blit_copy(&self.projections, &self.projections_backup);
		self.blit_copy(&self.scales, &self.scales_backup);
		Ok(())
	}

	fn reject(&mut self) -> Result<()> {
		self.blit_copy(&self.projections_backup, &self.projections);
		self.blit_copy(&self.scales_backup, &self.scales);
		let scale_sums_backup = self.scale_sums_backup.clone();
		unsafe {
			// SAFETY: `reject` has `&mut self`, and all prior command buffers are
			// synchronized before the host restores the accepted scale sums.
			write_buffer(&self.scale_sums, &scale_sums_backup);
		}
		Ok(())
	}

	fn num_patterns(&self) -> usize {
		self.num_patterns
	}
}

fn cast_row(row: [f64; 4]) -> [f32; 4] {
	[row[0] as f32, row[1] as f32, row[2] as f32, row[3] as f32]
}

fn threads_per_threadgroup(pipeline: &ComputePipelineState) -> u64 {
	let max = (pipeline.max_total_threads_per_threadgroup() as u64).max(4);
	let max = max - max % 4;
	let default = 256.min(max);

	env::var(THREADS_PER_THREADGROUP_ENV)
		.ok()
		.and_then(|value| value.parse::<u64>().ok())
		.filter(|&value| value > 0)
		.map(|value| value.min(max))
		.map(|value| (value - value % 4).max(4))
		.unwrap_or(default)
}

unsafe fn read_buffer<T: Copy>(buf: &Buffer, out: &mut [T]) {
	assert_eq!(buf.length() as usize, out.len() * mem::size_of::<T>());

	// SAFETY: the caller must guarantee that no host or device write races with
	// this read. The assertion guarantees that `out` has exactly enough space for
	// `out.len()` values of `T`, and `T: Copy` makes the bytewise copy sound.
	let src = buf.contents() as *const T;
	unsafe { ptr::copy_nonoverlapping(src, out.as_mut_ptr(), out.len()) };
}

unsafe fn write_buffer<T: Copy>(buf: &Buffer, data: &[T]) {
	assert!(buf.length() as usize >= data.len() * mem::size_of::<T>());

	// SAFETY: the caller must guarantee exclusive access with respect to host and
	// device readers/writers. The assertion guarantees that the destination is
	// large enough for `data.len()` values of `T`, and `T: Copy` makes the raw
	// copy sound.
	let dst = buf.contents() as *mut T;
	unsafe { ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len()) };
}

fn new_buffer<T>(
	device: &Device,
	data: &[T],
	options: MTLResourceOptions,
) -> Buffer {
	// This helper is safe because the allocation size is derived directly from
	// `data.len()` and `size_of::<T>()`, and the copied range is exactly `data`.
	let size = (data.len() * mem::size_of::<T>()) as u64;
	let buf = device.new_buffer(size, options);

	// SAFETY: `buf` was allocated with exactly `size` writable bytes, where
	// `size` was computed from `data.len()` and `size_of::<T>()`, so copying
	// `data.len()` values of `T` into it is in-bounds.
	unsafe {
		let dst = buf.contents() as *mut T;
		ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
	}

	buf
}

fn new_zeroed_buffer<T>(
	device: &Device,
	len: usize,
	options: MTLResourceOptions,
) -> Buffer {
	// This helper is safe because it allocates exactly `len * size_of::<T>()`
	// bytes and only zero-fills that newly allocated range.
	let size = (len * mem::size_of::<T>()) as u64;
	let buf = device.new_buffer(size, options);

	// SAFETY: `buf` owns `size` writable bytes, so zero-filling that exact range
	// is valid regardless of `T`.
	unsafe {
		ptr::write_bytes(buf.contents(), 0, size as usize);
	}

	buf
}
