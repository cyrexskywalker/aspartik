use super::Calculator;

#[cfg(target_os = "macos")]
mod imp {
	use super::Calculator;
	use anyhow::{Result, anyhow};
	use metal::{
		Buffer, CommandQueue, CompileOptions, ComputePipelineState,
		Device, MTLResourceOptions, MTLSize,
	};

	use std::mem;
	use std::ptr;

	/// Metal implementation of the 4-state (DNA) Felsenstein likelihood calculator.
	///
	/// Data layout follows CPU/CUDA calculators:
	/// - per-node/per-site partials are stored contiguously by node
	/// - one GPU thread computes all updates for exactly one site
	/// - scaling state is tracked per (node, site) and accumulated per site
	pub struct MetalLikelihood {
		queue: CommandQueue,
		pipeline: ComputePipelineState,

		/// Bitmask-encoded leaves (A/C/G/T ambiguity mask) for all leaves and sites.
		leaves: Buffer,
		/// Working partial likelihoods for all nodes and sites.
		projections: Buffer,
		/// Accepted snapshot of `projections` for reject rollback.
		projections_backup: Buffer,
		/// Per (node, site) flag: whether this partial was scaled in accepted state.
		scales: Buffer,
		/// Accepted snapshot of `scales` for reject rollback.
		scales_backup: Buffer,
		/// Per-site accumulated scaling offset (in log-space units, i.e. scale_ln steps).
		scale_sums: Buffer,
		/// CPU-side backup of accepted scale sums for reject rollback.
		scale_sums_backup: Vec<u32>,
		/// Per-site root log-likelihood values produced by kernel.
		likelihoods: Buffer,

		/// Update order of nodes for current proposal (`nodes.last()` is root).
		nodes: Buffer,
		/// Pairs of children for internal nodes in current proposal.
		children: Buffer,
		/// Transition rows in linear space.
		///
		/// For each updated edge i we store 4 rows (state 0..3), each row is float4.
		/// Total length: `(num_internals * 2) * 4`.
		transitions: Buffer,

		/// Packed integer params for kernel:
		/// [num_sites, num_updated_nodes, leaves_end, scale_ln].
		params_u32: Buffer,
		/// Packed float params for kernel: [scale, inv_scale].
		params_f32: Buffer,
		/// Root frequencies as one float4.
		freq: Buffer,
		num_sites: usize,
		scale_ln: u32,
		scale: f32,
		inv_scale: f32,
		/// Number of updated non-root nodes in the latest launched proposal.
		/// Used to skip no-op accept/reject and keep rollback logic correct.
		num_updated_nodes: usize,
	}

	impl MetalLikelihood {
		pub fn new(num_sites: usize, leaves: Vec<u8>, scale_ln: u32) -> Result<Self> {
			// 1) Create Metal device/queue and compile the kernel at runtime.
			let device = Device::system_default()
				.ok_or_else(|| anyhow!("Metal device not found"))?;
			let queue = device.new_command_queue();

			let source = include_str!("kernels.metal");
			let options = CompileOptions::new();
			let library = device
				.new_library_with_source(source, &options)
				.map_err(|e| anyhow!("failed to compile Metal library: {e}"))?;
			let function = library
				.get_function("propose_kernel", None)
				.map_err(|e| anyhow!("failed to get propose_kernel: {e}"))?;
			let pipeline = device
				.new_compute_pipeline_state_with_function(&function)
				.map_err(|e| anyhow!("failed to create compute pipeline: {e}"))?;

			// 2) Infer tree dimensions from encoded leaves.
			let num_leaves = leaves.len() / num_sites;
			let num_internals = num_leaves - 1;
			let num_nodes = num_leaves + num_internals;

			// Shared mode is simple and reliable for this stage:
			// CPU can read/write without explicit staging buffers.
			let options = MTLResourceOptions::StorageModeShared;

			// 3) Allocate long-lived buffers once.
			let leaves = new_buffer(&device, &leaves, options);
			let projections = new_zeroed_buffer::<[f32; 4]>(
				&device,
				num_nodes * num_sites,
				options,
			);
			let projections_backup = new_zeroed_buffer::<[f32; 4]>(
				&device,
				num_nodes * num_sites,
				options,
			);
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

			let params_u32 = new_zeroed_buffer::<u32>(&device, 4, options);
			let params_f32 = new_zeroed_buffer::<f32>(&device, 2, options);
			let freq = new_zeroed_buffer::<[f32; 4]>(&device, 1, options);

			// Keep scaling convention identical to CPU/CUDA:
			// scale = exp(-scale_ln), inv_scale = 1/scale.
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
				params_u32,
				params_f32,
				freq,
				num_sites,
				scale_ln,
				scale,
				inv_scale,
				num_updated_nodes: 0,
			})
		}

		fn encode_propose(&self) {
			// Record and run a single compute dispatch that updates:
			// leaves -> internals -> root likelihood (for each site/thread).
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
			encoder.set_buffer(8, Some(&self.params_u32), 0);
			encoder.set_buffer(9, Some(&self.params_f32), 0);
			encoder.set_buffer(10, Some(&self.freq), 0);

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
			// Device-side buffer copy, used by accept/reject snapshots.
			let command_buffer = self.queue.new_command_buffer();
			let blit = command_buffer.new_blit_command_encoder();
			blit.copy_from_buffer(src, 0, dst, 0, src.length());
			blit.end_encoding();
			command_buffer.commit();
			command_buffer.wait_until_completed();
		}

		fn read_buffer<T: Copy>(&self, buf: &Buffer, out: &mut [T]) {
			// Safe due to size assertion and exact plain-old-data copy.
			assert_eq!(buf.length() as usize, out.len() * mem::size_of::<T>());
			unsafe {
				let src = buf.contents() as *const T;
				ptr::copy_nonoverlapping(src, out.as_mut_ptr(), out.len());
			}
		}

		fn write_buffer<T: Copy>(&self, buf: &Buffer, data: &[T]) {
			// Safe due to capacity assertion and exact plain-old-data copy.
			assert!(buf.length() as usize >= data.len() * mem::size_of::<T>());
			unsafe {
				let dst = buf.contents() as *mut T;
				ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
			}
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
			// No changed nodes => cached likelihood remains valid.
			if nodes.is_empty() {
				return Ok(());
			}

			// `nodes` includes root at the end, but transitions are only for edges,
			// so updated edge count is `nodes.len() - 1`.
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

			// Cast transition matrices from f64 host representation into f32 rows
			// consumed by the kernel.
			let mut transitions_rows: Vec<[f32; 4]> =
				Vec::with_capacity(transitions.len() * 4);
			for t in transitions {
				transitions_rows.push(cast_row(t[0]));
				transitions_rows.push(cast_row(t[1]));
				transitions_rows.push(cast_row(t[2]));
				transitions_rows.push(cast_row(t[3]));
			}

			self.write_buffer(&self.nodes, &nodes_u32);
			self.write_buffer(&self.children, &children_u32);
			self.write_buffer(&self.transitions, &transitions_rows);
			// Integer params: shape/loop bounds + scaling unit.
			let p = [
				self.num_sites as u32,
				num_updated_nodes as u32,
				leaves_end as u32,
				self.scale_ln,
			];
			self.write_buffer(&self.params_u32, &p);
			// Float params: scaling thresholds/factors.
			self.write_buffer(&self.params_f32, &[self.scale, self.inv_scale]);
			// Root frequencies used in final root likelihood dot product.
			let f = [[
				frequencies[0] as f32,
				frequencies[1] as f32,
				frequencies[2] as f32,
				frequencies[3] as f32,
			]];
			self.write_buffer(&self.freq, &f);

			self.encode_propose();
			Ok(())
		}

		fn likelihood(&mut self, patterns: &mut [f64]) -> Result<()> {
			// Read per-site root log-likelihood and subtract accumulated scaling.
			let mut likelihoods = vec![0f32; self.num_sites];
			self.read_buffer(&self.likelihoods, &mut likelihoods);
			let mut scale_sums = vec![0u32; self.num_sites];
			self.read_buffer(&self.scale_sums, &mut scale_sums);

			for i in 0..self.num_sites {
				patterns[i] = f64::from(likelihoods[i]) - f64::from(scale_sums[i]);
			}
			Ok(())
		}

		fn accept(&mut self) -> Result<()> {
			// No launched update => nothing to snapshot.
			if self.num_updated_nodes == 0 {
				return Ok(());
			}
			// Persist accepted scaling offsets on CPU and device-side partial/scales.
			let mut scale_sums_backup = vec![0u32; self.num_sites];
			self.read_buffer(&self.scale_sums, &mut scale_sums_backup);
			self.scale_sums_backup = scale_sums_backup;
			self.blit_copy(&self.projections, &self.projections_backup);
			self.blit_copy(&self.scales, &self.scales_backup);
			self.num_updated_nodes = 0;
			Ok(())
		}

		fn reject(&mut self) -> Result<()> {
			// No launched update => nothing to roll back.
			if self.num_updated_nodes == 0 {
				return Ok(());
			}
			// Restore accepted device snapshots and accepted per-site scale sums.
			self.blit_copy(&self.projections_backup, &self.projections);
			self.blit_copy(&self.scales_backup, &self.scales);
			let scale_sums_backup = self.scale_sums_backup.clone();
			self.write_buffer(&self.scale_sums, &scale_sums_backup);
			self.num_updated_nodes = 0;
			Ok(())
		}
	}

	fn cast_row(row: [f64; 4]) -> [f32; 4] {
		// Explicit cast to keep host math in f64 while kernel uses f32.
		[row[0] as f32, row[1] as f32, row[2] as f32, row[3] as f32]
	}

	fn new_buffer<T>(
		device: &Device,
		data: &[T],
		options: MTLResourceOptions,
	) -> Buffer {
		// Allocate and initialize a buffer from host data.
		let size = (data.len() * mem::size_of::<T>()) as u64;
		let buf = device.new_buffer(size, options);
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
		// Allocate zero-filled typed buffer.
		let size = (len * mem::size_of::<T>()) as u64;
		let buf = device.new_buffer(size, options);
		unsafe {
			ptr::write_bytes(buf.contents(), 0, size as usize);
		}
		buf
	}
}

#[cfg(target_os = "macos")]
pub use imp::MetalLikelihood;

// Non-macOS fallback: keep the same public type/API available for cross-platform
// builds, but return a clear runtime error because Metal backend exists only on macOS.
#[cfg(not(target_os = "macos"))]
pub struct MetalLikelihood;

#[cfg(not(target_os = "macos"))]
impl MetalLikelihood {
	pub fn new(_num_sites: usize, _leaves: Vec<u8>, _scale_ln: u32) -> Result<Self> {
		bail!("Metal backend is only available on macOS");
	}
}

#[cfg(not(target_os = "macos"))]
impl Calculator<4, f64> for MetalLikelihood {
	fn propose(
		&mut self,
		_nodes: &[usize],
		_children: &[(usize, usize)],
		_transitions: &[[[f64; 4]; 4]],
		_leaves_end: usize,
		_frequencies: [f64; 4],
	) -> Result<()> {
		bail!("Metal backend is only available on macOS");
	}

	fn likelihood(&mut self, _patterns: &mut [f64]) -> Result<()> {
		bail!("Metal backend is only available on macOS");
	}

	fn accept(&mut self) -> Result<()> {
		bail!("Metal backend is only available on macOS");
	}

	fn reject(&mut self) -> Result<()> {
		bail!("Metal backend is only available on macOS");
	}
}
