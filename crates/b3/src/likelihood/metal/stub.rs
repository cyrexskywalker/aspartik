use anyhow::{Result, bail};

use super::Calculator;

// Keep the type available on non-macOS targets so the crate still builds,
// but fail explicitly at runtime because Metal backend exists only on macOS.
pub struct MetalLikelihood;

impl MetalLikelihood {
	pub fn new(_num_sites: usize, _leaves: Vec<u8>, _scale_ln: u32) -> Result<Self> {
		bail!("Metal backend is only available on macOS");
	}
}

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
