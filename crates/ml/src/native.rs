//! Pinned, hash-verified BiRefNet ONNX asset, CPU inference and a single cached session.
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, PoisonError};

use ort::session::Session;
use ort::value::Tensor;
use sha2::{Digest, Sha256};

use crate::{Error, InferenceBackend, MODEL_SIDE, Matte, ModelInput, Progress, Result};

pub const MODEL_URL: &str = "https://huggingface.co/onnx-community/BiRefNet_lite-ONNX/resolve/de15b22/onnx/model.onnx";
pub const MODEL_SHA256: &str = "5600024376f572a557870a5eb0afb1e5961636bef4e1e22132025467d0f03333";
const MAX_MODEL_BYTES: u64 = 268_435_456;
const LICENSE: &str = include_str!("../../../assets/models/LICENSE-BiRefNet.txt");
static SESSION: Mutex<Option<Session>> = Mutex::new(None);

pub struct OnnxBackend {
    pub download: bool,
}

fn runtime<T>(r: ort::Result<T>) -> Result<T> {
    r.map_err(|e| {
        let message = e.to_string();
        if message.contains("Failed to allocate memory") || message.contains("bad_alloc") {
            Error::Runtime("Not enough memory for AI background removal; close other applications and try again".into())
        } else {
            Error::Runtime(format!("ONNX Runtime: {message}"))
        }
    })
}
fn io(e: std::io::Error) -> Error {
    Error::Runtime(format!("AI model cache: {e}"))
}

fn cache_dir() -> Result<PathBuf> {
    if let Some(dir) = std::env::var_os("PHOTOCRAFT_MODEL_DIR").filter(|s| !s.is_empty()) {
        return Ok(dir.into());
    }
    #[cfg(target_os = "windows")]
    let root = std::env::var_os("LOCALAPPDATA").map(PathBuf::from);
    #[cfg(not(target_os = "windows"))]
    let root = std::env::var_os("XDG_CACHE_HOME").map(PathBuf::from).or_else(|| std::env::var_os("HOME").map(|s| PathBuf::from(s).join(".cache")));
    root.map(|p| p.join("PhotoCraft").join("models")).ok_or_else(|| Error::Runtime("No model cache directory; set PHOTOCRAFT_MODEL_DIR".into()))
}

pub fn model_available() -> bool {
    cache_dir().is_ok_and(|p| p.join("birefnet-lite-de15b22.onnx").is_file())
}

fn verify(path: &Path, progress: &dyn Progress) -> Result<()> {
    let mut file = File::open(path).map_err(io)?;
    if file.metadata().map_err(io)?.len() > MAX_MODEL_BYTES {
        return Err(Error::Invalid("BiRefNet model exceeds the size limit".into()));
    }
    let mut hash = Sha256::new();
    let mut buf = [0u8; 65_536];
    loop {
        progress.check()?;
        let count = file.read(&mut buf).map_err(io)?;
        if count == 0 {
            break;
        }
        hash.update(buf.get(..count).ok_or_else(|| Error::Invalid("invalid model read".into()))?);
    }
    let actual = format!("{:x}", hash.finalize());
    if actual != MODEL_SHA256 {
        return Err(Error::Invalid("BiRefNet model checksum mismatch; remove the cached model and download again".into()));
    }
    Ok(())
}

/// A failed/cancelled download never becomes an installed model. Do not remove somebody else's
/// partial file: create_new owns this path, and concurrent processes get an actionable error.
struct PartialFile {
    path: PathBuf,
    file: Option<File>,
}
impl Drop for PartialFile {
    fn drop(&mut self) {
        // Windows cannot remove an open file. Close our handle before removing our partial.
        self.file.take();
        let _ = fs::remove_file(&self.path);
    }
}

fn model_path(download: bool, progress: &dyn Progress) -> Result<PathBuf> {
    let dir = cache_dir()?;
    let target = dir.join("birefnet-lite-de15b22.onnx");
    if target.is_file() {
        verify(&target, progress)?;
        return Ok(target);
    }
    if !download {
        return Err(Error::Runtime("BiRefNet is not installed; enable Download model once (about 224 MB)".into()));
    }
    fs::create_dir_all(&dir).map_err(io)?;
    let partial_path = dir.join("birefnet-lite-de15b22.onnx.part");
    let file = OpenOptions::new().write(true).create_new(true).open(&partial_path).map_err(io)?;
    let mut partial = PartialFile { path: partial_path, file: Some(file) };
    let agent = ureq::AgentBuilder::new().timeout_connect(std::time::Duration::from_secs(20)).timeout_read(std::time::Duration::from_secs(30)).build();
    let response = agent.get(MODEL_URL).set("User-Agent", "Photocraft-dev").call().map_err(|e| Error::Runtime(format!("Could not download BiRefNet: {e}")))?;
    let total = response.header("Content-Length").and_then(|s| s.parse::<u64>().ok()).unwrap_or(MAX_MODEL_BYTES);
    if total > MAX_MODEL_BYTES {
        return Err(Error::Invalid("model download exceeds the size limit".into()));
    }
    let mut reader = response.into_reader();
    let mut buf = [0u8; 65_536];
    let mut received = 0u64;
    let mut hash = Sha256::new();
    loop {
        progress.check()?;
        let count = reader.read(&mut buf).map_err(io)?;
        if count == 0 {
            break;
        }
        received = received
            .checked_add(count as u64)
            .filter(|n| *n <= MAX_MODEL_BYTES)
            .ok_or_else(|| Error::Invalid("model download exceeds the size limit".into()))?;
        let bytes = buf.get(..count).ok_or_else(|| Error::Invalid("invalid model download".into()))?;
        partial.file.as_mut().ok_or_else(|| Error::Runtime("Download file closed".into()))?.write_all(bytes).map_err(io)?;
        hash.update(bytes);
        progress.report(0.05 + 0.3 * (received as f32 / total.max(1) as f32).min(1.0), "Downloading BiRefNet (once)");
    }
    if format!("{:x}", hash.finalize()) != MODEL_SHA256 {
        return Err(Error::Invalid("BiRefNet download checksum mismatch".into()));
    }
    partial.file.as_ref().ok_or_else(|| Error::Runtime("Download file closed".into()))?.sync_all().map_err(io)?;
    partial.file.take();
    progress.check()?;
    fs::write(dir.join("LICENSE-BiRefNet.txt"), LICENSE).map_err(io)?;
    fs::rename(&partial.path, &target).map_err(io)?;
    Ok(target)
}

fn initialize() -> Result<()> {
    runtime(ort::init().with_name("PhotoCraft ML").with_telemetry(false).commit())?;
    Ok(())
}

impl InferenceBackend for OnnxBackend {
    fn matte_subject(&self, image: &ModelInput, progress: &dyn Progress) -> Result<Matte> {
        let data = image.normalized()?;
        progress.check()?;
        // Keep an escaped third-party runtime panic at the foreign-library boundary, also for
        // direct library callers outside the app guard.
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut cached = SESSION.lock().unwrap_or_else(PoisonError::into_inner);
            progress.check()?;
            if cached.is_none() {
                initialize()?;
                progress.report(0.05, "Checking BiRefNet model");
                let path = model_path(self.download, progress)?;
                progress.report(0.4, "Loading BiRefNet");
                let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(8);
                let builder = runtime(runtime(Session::builder())?.with_intra_threads(threads))?;
                let builder = runtime(builder.with_memory_pattern(false))?;
                let builder =
                    runtime(builder.with_execution_providers([ort::execution_providers::CPUExecutionProvider::default().with_arena_allocator(false).build()]))?;
                *cached = Some(builder.commit_from_file(path).map_err(|e| Error::Runtime(format!("Could not load BiRefNet: {e}")))?);
            }
            let session = cached.as_mut().ok_or_else(|| Error::Runtime("BiRefNet session unavailable".into()))?;
            progress.check()?;
            progress.report(0.55, "Detecting subject with BiRefNet (CPU)");
            let tensor = runtime(Tensor::from_array(([1usize, 3, MODEL_SIDE, MODEL_SIDE], data.into_boxed_slice())))?;
            let options = runtime(ort::session::RunOptions::new())?;
            let finished = AtomicBool::new(false);
            struct Finish<'a>(&'a AtomicBool);
            impl Drop for Finish<'_> {
                fn drop(&mut self) {
                    self.0.store(true, Ordering::Release);
                }
            }
            let outputs = std::thread::scope(|scope| {
                let monitor = std::thread::Builder::new()
                    .name("photocraft-ai-cancel".into())
                    .spawn_scoped(scope, || {
                        while !finished.load(Ordering::Acquire) {
                            if progress.cancelled() {
                                let _ = options.terminate();
                                break;
                            }
                            std::thread::sleep(std::time::Duration::from_millis(20));
                        }
                    })
                    .map_err(io)?;
                let guard = Finish(&finished);
                let result = session.run_with_options(ort::inputs![tensor], &options);
                drop(guard);
                monitor.join().map_err(|_| Error::Runtime("AI cancellation monitor failed".into()))?;
                progress.check()?;
                runtime(result)
            })?;
            progress.check()?;
            let (_, value) = outputs.iter().next().ok_or_else(|| Error::Invalid("BiRefNet returned no matte".into()))?;
            let (shape, values) = runtime(value.try_extract_tensor::<f32>())?;
            if shape.as_ref() != [1, 1, MODEL_SIDE as i64, MODEL_SIDE as i64] {
                return Err(Error::Invalid(format!("Unexpected BiRefNet output shape: {shape:?}")));
            }
            if values.iter().any(|v| !v.is_finite()) {
                return Err(Error::Invalid("BiRefNet returned non-finite coverage".into()));
            }
            // The pinned export returns foreground logits, not binary labels.
            let alpha = values
                .iter()
                .map(|v| {
                    if *v >= 0.0 {
                        1.0 / (1.0 + (-v).exp())
                    } else {
                        let e = v.exp();
                        e / (1.0 + e)
                    }
                })
                .collect();
            Matte::new(MODEL_SIDE, MODEL_SIDE, alpha)
        }))
        .map_err(|_| Error::Runtime("ONNX Runtime failed; check the installed runtime version".into()))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct NoProgress;
    impl Progress for NoProgress {
        fn cancelled(&self) -> bool {
            false
        }
        fn report(&self, _: f32, _: &str) {}
    }
    #[test]
    fn incomplete_owned_download_closes_and_removes_the_file() {
        let path = std::env::temp_dir().join(format!("photocraft-ai-partial-{}", std::process::id()));
        let file = OpenOptions::new().write(true).create_new(true).open(&path).unwrap();
        {
            let _partial = PartialFile { path: path.clone(), file: Some(file) };
        }
        assert!(!path.exists());
    }
    #[test]
    fn corrupt_model_is_rejected_without_starting_inference() {
        let path = std::env::temp_dir().join(format!("photocraft-ai-corrupt-{}", std::process::id()));
        std::fs::write(&path, b"not an ONNX model").unwrap();
        let result = verify(&path, &NoProgress);
        std::fs::remove_file(path).unwrap();
        assert!(matches!(result, Err(Error::Invalid(_))));
    }
}
