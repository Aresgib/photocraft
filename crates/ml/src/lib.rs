//! Local, optional ML inference. The document and UI remain independent of the runtime.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

pub const MODEL_SIDE: usize = 1024;
pub const MAX_IMAGE_PIXELS: usize = 64_000_000;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Invalid(String),
    #[error("AI background removal was cancelled")]
    Cancelled,
    #[error("AI background removal requires a native build with the ai-background feature")]
    Unsupported,
    #[error("{0}")]
    Runtime(String),
}
pub type Result<T> = std::result::Result<T, Error>;

/// Cooperative progress/cancellation, usable without an engine dependency.
pub trait Progress: Sync {
    fn cancelled(&self) -> bool;
    fn report(&self, fraction: f32, message: &str);
    fn check(&self) -> Result<()> {
        if self.cancelled() { Err(Error::Cancelled) } else { Ok(()) }
    }
}

/// Straight sRGB in 0..1, resampled to the model's exact input dimensions.
#[derive(Clone, Debug)]
pub struct ModelInput {
    pub rgba: Vec<[f32; 4]>,
}

impl ModelInput {
    pub fn normalized(&self) -> Result<Vec<f32>> {
        let plane = MODEL_SIDE * MODEL_SIDE;
        if self.rgba.len() != plane || self.rgba.iter().flatten().any(|v| !v.is_finite()) {
            return Err(Error::Invalid("invalid BiRefNet input; expected finite 1024 x 1024 RGBA".into()));
        }
        let mut data = vec![0.0; plane * 3];
        // BiRefNet's published ImageNet normalization; transparency is shown on neutral gray.
        for (c, ((channel, mean), std)) in data.chunks_exact_mut(plane).zip([0.485, 0.456, 0.406]).zip([0.229, 0.224, 0.225]).enumerate() {
            for (out, px) in channel.iter_mut().zip(&self.rgba) {
                let a = px.get(3).copied().unwrap_or(0.0).clamp(0.0, 1.0);
                let rgb = px.get(c).copied().unwrap_or(0.0).clamp(0.0, 1.0) * a + 0.5 * (1.0 - a);
                *out = (rgb - mean) / std;
            }
        }
        Ok(data)
    }
}

/// Soft coverage, never thresholded to a binary selection or quantized to eight bits.
#[derive(Clone, Debug)]
pub struct Matte {
    pub width: usize,
    pub height: usize,
    pub alpha: Vec<f32>,
}

impl Matte {
    pub fn new(width: usize, height: usize, alpha: Vec<f32>) -> Result<Self> {
        let count = width
            .checked_mul(height)
            .filter(|n| *n > 0 && *n <= MAX_IMAGE_PIXELS)
            .ok_or_else(|| Error::Invalid("invalid or oversized matte dimensions".into()))?;
        if alpha.len() != count || alpha.iter().any(|v| !v.is_finite() || !(0.0..=1.0).contains(v)) {
            return Err(Error::Invalid("invalid matte coverage".into()));
        }
        Ok(Self { width, height, alpha })
    }

    /// Bilinear, pixel-centred sampling with edge clamping (coordinates in this matte).
    pub fn sample(&self, x: f32, y: f32) -> f32 {
        if !x.is_finite() || !y.is_finite() || self.width == 0 || self.height == 0 {
            return 0.0;
        }
        let x = x.clamp(0.0, self.width.saturating_sub(1) as f32);
        let y = y.clamp(0.0, self.height.saturating_sub(1) as f32);
        let (x0, y0) = (x.floor() as usize, y.floor() as usize);
        let (tx, ty) = (x - x0 as f32, y - y0 as f32);
        let at = |xx: usize, yy: usize| yy.checked_mul(self.width).and_then(|i| i.checked_add(xx)).and_then(|i| self.alpha.get(i)).copied().unwrap_or(0.0);
        let x1 = x0.saturating_add(1).min(self.width.saturating_sub(1));
        let y1 = y0.saturating_add(1).min(self.height.saturating_sub(1));
        (at(x0, y0) * (1.0 - tx) + at(x1, y0) * tx) * (1.0 - ty) + (at(x0, y1) * (1.0 - tx) + at(x1, y1) * tx) * ty
    }
}

pub trait InferenceBackend {
    fn matte_subject(&self, image: &ModelInput, progress: &dyn Progress) -> Result<Matte>;
}

#[cfg(all(feature = "onnx", not(target_arch = "wasm32"), any(target_arch = "x86_64", target_arch = "aarch64")))]
mod native;
#[cfg(all(feature = "onnx", not(target_arch = "wasm32"), any(target_arch = "x86_64", target_arch = "aarch64")))]
pub use native::{OnnxBackend, model_available};

#[cfg(not(all(feature = "onnx", not(target_arch = "wasm32"), any(target_arch = "x86_64", target_arch = "aarch64"))))]
pub fn model_available() -> bool {
    false
}

pub fn supported() -> bool {
    cfg!(all(feature = "onnx", not(target_arch = "wasm32"), any(target_arch = "x86_64", target_arch = "aarch64")))
}

pub fn infer(image: &ModelInput, download: bool, progress: &dyn Progress) -> Result<Matte> {
    #[cfg(all(feature = "onnx", not(target_arch = "wasm32"), any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        OnnxBackend { download }.matte_subject(image, progress)
    }
    #[cfg(not(all(feature = "onnx", not(target_arch = "wasm32"), any(target_arch = "x86_64", target_arch = "aarch64"))))]
    {
        let _ = (image, download, progress);
        Err(Error::Unsupported)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn validates_shapes_and_nonfinite_coverage() {
        assert!(Matte::new(usize::MAX, 2, vec![]).is_err());
        assert!(Matte::new(0, 1, vec![]).is_err());
        assert!(Matte::new(1, 1, vec![f32::NAN]).is_err());
        assert!(ModelInput { rgba: vec![] }.normalized().is_err());
        let m = Matte::new(2, 2, vec![0.0, 1.0, 0.0, 1.0]).unwrap();
        assert_eq!(m.sample(0.5, 0.5), 0.5);
        assert_eq!(m.sample(f32::NAN, 0.0), 0.0);
        assert_eq!(m.sample(100.0, 0.0), 1.0);
    }
    #[test]
    fn normalizes_rgb_without_thresholding_transparency() {
        let input = ModelInput { rgba: vec![[1.0, 0.0, 0.0, 1.0]; MODEL_SIDE * MODEL_SIDE] };
        let data = input.normalized().unwrap();
        assert!((data[0] - (1.0 - 0.485) / 0.229).abs() < 1e-5);
        assert!((data[MODEL_SIDE * MODEL_SIDE] - -0.456 / 0.224).abs() < 1e-5);
    }
}
