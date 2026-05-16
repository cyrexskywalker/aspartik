use super::Calculator;

#[cfg(target_os = "macos")]
mod imp;
#[cfg(not(target_os = "macos"))]
mod stub;

#[cfg(target_os = "macos")]
pub use imp::MetalLikelihood;
#[cfg(not(target_os = "macos"))]
pub use stub::MetalLikelihood;
