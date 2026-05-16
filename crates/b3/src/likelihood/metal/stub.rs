use anyhow::{Result, bail};
use parking_lot::MutexGuard;

use super::Calculator;
use crate::{Transitions, parameters::Tree};

// Keep the type available on non-macOS targets so the crate still builds, but
// fail explicitly at runtime because Metal backend exists only on macOS.
pub struct MetalLikelihood;

impl MetalLikelihood {
	pub fn new(
		_pattern_weights: Vec<u32>,
		_leaves: Vec<u8>,
		_scale_ln: u32,
	) -> Result<Self> {
		bail!("Metal backend is only available on macOS");
	}
}

impl Calculator<4, f64> for MetalLikelihood {
	fn likelihood(
		&mut self,
		_tree: MutexGuard<Tree>,
		_transitions: &Transitions<4, f64>,
	) -> Result<f64> {
		bail!("Metal backend is only available on macOS");
	}

	fn accept(&mut self) -> Result<()> {
		bail!("Metal backend is only available on macOS");
	}

	fn reject(&mut self) -> Result<()> {
		bail!("Metal backend is only available on macOS");
	}

	fn num_patterns(&self) -> usize {
		0
	}
}
