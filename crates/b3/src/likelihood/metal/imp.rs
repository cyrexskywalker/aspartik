use anyhow::{Context, Result, anyhow};
use metal::{
	Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device,
	MTLResourceOptions, MTLSize,
};

use std::mem;
use std::ptr;

use super::Calculator;

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
	num_sites: usize,
	scale_ln: u32,
	scale: f32,
	inv_scale: f32,
	num_updated_nodes: usize,
}

impl MetalLikelihood {
	pub fn new(num_sites: usize, leaves: Vec<u8>, scale_ln: u32) -> Result<Self> {
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

		let num_leaves = leaves.len() / num_sites;
		let num_internals = num_leaves - 1;
		let num_nodes = num_leaves + num_internals;

		let options = MTLResourceOptions::StorageModeShared;

		let leaves = new_buffer(&device, &leaves, options);
		let projections =
			new_zeroed_buffer::<[f32; 4]>(&device, num_nodes * num_sites, options);
		let projections_backup =
			new_zeroed_buffer::<[f32; 4]>(&device, num_nodes * num_sites, options);
		let scales =
			new_zeroed_buffer::<u8>(&device, num_nodes * num_sites, options);
		let scales_backup =
			new_zeroed_buffer::<u8>(&device, num_nodes * num_sites, options);
		let scale_sums = new_zeroed_buffer::<u32>(&device, num_sites, options);
		let likelihoods = new_zeroed_buffer::<f32>(&device, num_sites, options);

		let nodes = new_zeroed_buffer::<u32>(&device, num_nodes, options);
		let children =
			new_zeroed_buffer::<u32>(&device, num_internals * 2, options);
		let transitions = new_zeroed_buffer::<[f32; 4]>(
			&device,
			num_internals * 2 * 4,
			options,
		);

		let params = new_zeroed_buffer::<KernelParams>(&device, 1, options);

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
			scale_sums_backup: vec![0; num_sites],
			likelihoods,
			nodes,
			children,
			transitions,
			params,
			num_sites,
			scale_ln,
			scale,
			inv_scale,
			num_updated_nodes: 0,
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

		let threads_per_grid = MTLSize {
			width: self.num_sites as u64,
			height: 1,
			depth: 1,
		};
		let threads_per_threadgroup = MTLSize {
			width: 256u64.min(self.num_sites as u64),
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
	fn propose(
		&mut self,
		nodes: &[usize],
		children: &[(usize, usize)],
		transitions: &[[[f64; 4]; 4]],
		leaves_end: usize,
		frequencies: [f64; 4],
	) -> Result<()> {
		if nodes.is_empty() {
			return Ok(());
		}

		let num_updated_nodes = nodes.len() - 1;
		self.num_updated_nodes = num_updated_nodes;
		if num_updated_nodes == 0 {
			return Ok(());
		}

		let nodes_u32: Vec<u32> = nodes.iter().map(|&n| n as u32).collect();
		let children_u32: Vec<u32> = children
			.iter()
			.flat_map(|&(l, r)| [l as u32, r as u32])
			.collect();

		let mut transitions_rows: Vec<[f32; 4]> =
			Vec::with_capacity(transitions.len() * 4);
		for t in transitions {
			transitions_rows.push(cast_row(t[0]));
			transitions_rows.push(cast_row(t[1]));
			transitions_rows.push(cast_row(t[2]));
			transitions_rows.push(cast_row(t[3]));
		}

		unsafe {
			// SAFETY: `propose` has `&mut self`, so there is no concurrent host-side
			// access through this calculator instance. No command buffer touching these
			// inputs is in flight yet; they are written before `encode_propose`.
			write_buffer(&self.nodes, &nodes_u32);
			write_buffer(&self.children, &children_u32);
			write_buffer(&self.transitions, &transitions_rows);
		}

		let params = [KernelParams {
			num_sites: self.num_sites as u32,
			num_updated_nodes: num_updated_nodes as u32,
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
		}];
		unsafe {
			// SAFETY: same invariant as above; the parameter buffer is populated on
			// the host before the kernel is launched.
			write_buffer(&self.params, &params);
		}

		self.encode_propose();
		Ok(())
	}

	fn likelihood(&mut self, patterns: &mut [f64]) -> Result<()> {
		let mut likelihoods = vec![0f32; self.num_sites];
		unsafe {
			// SAFETY: all compute and blit command buffers in this type are waited on
			// before returning, and `likelihood` has `&mut self`, so this host read
			// cannot race with another host/device access through this calculator.
			read_buffer(&self.likelihoods, &mut likelihoods);
		}

		let mut scale_sums = vec![0u32; self.num_sites];
		unsafe {
			// SAFETY: same invariant as for `likelihoods` above.
			read_buffer(&self.scale_sums, &mut scale_sums);
		}

		for i in 0..self.num_sites {
			patterns[i] = f64::from(likelihoods[i]) - f64::from(scale_sums[i]);
		}

		Ok(())
	}

	fn accept(&mut self) -> Result<()> {
		if self.num_updated_nodes == 0 {
			return Ok(());
		}

		let mut scale_sums_backup = vec![0u32; self.num_sites];
		unsafe {
			// SAFETY: `accept` has `&mut self`, and the latest kernel launch already
			// completed before we snapshot the accepted scale sums.
			read_buffer(&self.scale_sums, &mut scale_sums_backup);
		}
		self.scale_sums_backup = scale_sums_backup;
		self.blit_copy(&self.projections, &self.projections_backup);
		self.blit_copy(&self.scales, &self.scales_backup);
		self.num_updated_nodes = 0;

		Ok(())
	}

	fn reject(&mut self) -> Result<()> {
		if self.num_updated_nodes == 0 {
			return Ok(());
		}

		self.blit_copy(&self.projections_backup, &self.projections);
		self.blit_copy(&self.scales_backup, &self.scales);
		let scale_sums_backup = self.scale_sums_backup.clone();
		unsafe {
			// SAFETY: `reject` has `&mut self`, and this host write happens after the
			// previous command buffers have completed.
			write_buffer(&self.scale_sums, &scale_sums_backup);
		}
		self.num_updated_nodes = 0;

		Ok(())
	}
}

fn cast_row(row: [f64; 4]) -> [f32; 4] {
	[row[0] as f32, row[1] as f32, row[2] as f32, row[3] as f32]
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
	let size = (len * mem::size_of::<T>()) as u64;
	let buf = device.new_buffer(size, options);

	// SAFETY: `buf` owns `size` writable bytes, so zero-filling that exact range
	// is valid regardless of `T`.
	unsafe {
		ptr::write_bytes(buf.contents(), 0, size as usize);
	}

	buf
}
