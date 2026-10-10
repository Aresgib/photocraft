//! Pinned, hash-verified BiRefNet ONNX asset, configurable inference and one cached session.
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, PoisonError};

use ort::session::Session;
use ort::value::Tensor;
use sha2::{Digest, Sha256};

use crate::{Backend, Error, InferenceBackend, InferenceOptions, InferenceOutput, MODEL_SIDE, Matte, ModelInput, Progress, Result};

pub const MODEL_URL: &str = "https://huggingface.co/onnx-community/BiRefNet_lite-ONNX/resolve/de15b22/onnx/model.onnx";
pub const MODEL_SHA256: &str = "5600024376f572a557870a5eb0afb1e5961636bef4e1e22132025467d0f03333";
const MAX_MODEL_BYTES: u64 = 268_435_456;
const LICENSE: &str = include_str!("../../../assets/models/LICENSE-BiRefNet.txt");
struct CachedSession {
    options: InferenceOptions,
    provider: Backend,
    fallback_reason: Option<String>,
    device_id: u32,
    session: Session,
}
static SESSION: Mutex<Option<CachedSession>> = Mutex::new(None);

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

fn create_session(path: &Path, provider: Backend, device_id: u32) -> Result<Session> {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(8);
    let builder = runtime(runtime(Session::builder())?.with_intra_threads(threads))?;
    // DirectML requires sequential execution and memory patterns disabled.
    let builder = runtime(runtime(builder.with_memory_pattern(false))?.with_parallel_execution(false))?;
    let builder = match provider {
        Backend::Cpu => runtime(
            builder
                .with_execution_providers([ort::execution_providers::CPUExecutionProvider::default().with_arena_allocator(false).build().error_on_failure()]),
        )?,
        Backend::DirectMl => {
            #[cfg(target_os = "windows")]
            {
                let device_id = i32::try_from(device_id).map_err(|_| Error::Invalid("invalid AI GPU adapter index".into()))?;
                // BiRefNet's fused FP32 attention graph can demand a single allocation larger
                // than an 8 GB card can provide. Keep operator allocations reusable and bounded
                // by their lifetimes instead of retaining fused-graph and arena workspaces.
                let builder = runtime(builder.with_config_entry("ep.dml.disable_graph_fusion", "1"))?;
                let builder = runtime(builder.with_config_entry("ep.dml.disable_memory_arena", "1"))?;
                runtime(builder.with_execution_providers([
                    ort::execution_providers::DirectMLExecutionProvider::default().with_device_id(device_id).build().error_on_failure(),
                ]))?
            }
            #[cfg(not(target_os = "windows"))]
            {
                let _ = device_id;
                return Err(Error::Runtime("DirectML AI inference is only available on Windows".into()));
            }
        }
        Backend::Auto => return Err(Error::Invalid("Automatic must resolve to an inference provider".into())),
    };
    builder.commit_from_file(path).map_err(|e| Error::Runtime(format!("Could not load BiRefNet on {}: {e}", provider.label())))
}

/// Keep explicit GPU failures visible, while automatic selection can safely recover on CPU.
fn select_session<T>(options: InferenceOptions, gpu_supported: bool, mut create: impl FnMut(Backend) -> Result<T>) -> Result<(T, Backend, Option<String>)> {
    if options.backend == Backend::Auto && gpu_supported {
        match create(Backend::DirectMl) {
            Ok(session) => return Ok((session, Backend::DirectMl, None)),
            Err(Error::Cancelled) => return Err(Error::Cancelled),
            Err(e) => return Ok((create(Backend::Cpu)?, Backend::Cpu, Some(e.to_string()))),
        }
    }
    let provider = if options.backend == Backend::Auto { Backend::Cpu } else { options.backend };
    Ok((create(provider)?, provider, None))
}

pub fn infer(image: &ModelInput, download: bool, options: InferenceOptions, progress: &dyn Progress) -> Result<InferenceOutput> {
    options.validate()?;
    let data = image.normalized()?;
    progress.check()?;
    // Guard the third-party runtime boundary, including session creation and provider registration.
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut cached = SESSION.lock().unwrap_or_else(PoisonError::into_inner);
        progress.check()?;
        if cached.as_ref().is_none_or(|c| c.options != options) {
            // Release the old GPU allocations before loading the replacement session.
            *cached = None;
            initialize()?;
            progress.report(0.05, "Checking BiRefNet model");
            let path = model_path(download, progress)?;
            progress.report(0.4, "Loading BiRefNet");
            let device_id = if options.backend == Backend::Auto {
                crate::gpu_adapters().iter().max_by_key(|a| a.dedicated_memory_bytes).map_or(0, |a| a.device_id)
            } else {
                options.device_id
            };
            let (session, provider, fallback_reason) = select_session(options, crate::directml_supported(), |provider| {
                progress.check()?;
                create_session(&path, provider, device_id)
            })?;
            *cached = Some(CachedSession { options, provider, fallback_reason, device_id, session });
        }
        let state = cached.as_mut().ok_or_else(|| Error::Runtime("BiRefNet session unavailable".into()))?;
        let result = run_inference(&mut state.session, &data, state.provider, progress);
        let matte = match result {
            Err(Error::Runtime(reason)) if options.backend == Backend::Auto && state.provider == Backend::DirectMl => {
                progress.check()?;
                // A GPU that registers successfully may still run out of memory on this model.
                *cached = None;
                let path = model_path(download, progress)?;
                let session = create_session(&path, Backend::Cpu, options.device_id)?;
                *cached = Some(CachedSession { options, provider: Backend::Cpu, fallback_reason: Some(reason), device_id: options.device_id, session });
                let state = cached.as_mut().ok_or_else(|| Error::Runtime("CPU inference session unavailable".into()))?;
                run_inference(&mut state.session, &data, Backend::Cpu, progress)?
            }
            result => result?,
        };
        let state = cached.as_ref().ok_or_else(|| Error::Runtime("BiRefNet session unavailable".into()))?;
        Ok(InferenceOutput {
            matte,
            provider: state.provider,
            fallback_reason: state.fallback_reason.clone(),
            device_id: (state.provider == Backend::DirectMl).then_some(state.device_id),
        })
    }))
    .map_err(|_| Error::Runtime("ONNX Runtime failed; check the installed runtime version".into()))?
}

fn run_inference(session: &mut Session, data: &[f32], provider: Backend, progress: &dyn Progress) -> Result<Matte> {
    progress.check()?;
    progress.report(0.55, &format!("Detecting subject with BiRefNet ({})", provider.label()));
    let tensor = runtime(Tensor::from_array(([1usize, 3, MODEL_SIDE, MODEL_SIDE], data.to_vec().into_boxed_slice())))?;
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
}

impl InferenceBackend for OnnxBackend {
    fn matte_subject(&self, image: &ModelInput, progress: &dyn Progress) -> Result<Matte> {
        infer(image, self.download, InferenceOptions::default(), progress).map(|out| out.matte)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires cached BiRefNet; exercises the selected real hardware without downloads"]
    fn cached_model_runs_on_requested_provider() {
        let backend = std::env::var("PHOTOCRAFT_AI_TEST_BACKEND").unwrap_or_else(|_| "auto".into());
        let backend = Backend::parse(&backend).expect("backend must be auto, cpu or directml");
        let device_id = std::env::var("PHOTOCRAFT_AI_TEST_DEVICE").ok().map(|s| s.parse().expect("integer adapter index")).unwrap_or(0);
        // Synthetic silhouette: no personal photos or external test assets.
        let mut rgba = vec![[0.85, 0.9, 0.95, 1.0]; MODEL_SIDE * MODEL_SIDE];
        for y in 0..MODEL_SIDE {
            for x in 0..MODEL_SIDE {
                let dx = x as f32 - 512.0;
                let dy = y as f32 - 512.0;
                if dx * dx + dy * dy < 280.0 * 280.0 {
                    rgba[y * MODEL_SIDE + x] = [0.75, 0.15, 0.1, 1.0];
                }
            }
        }
        let image = ModelInput { rgba };
        for attempt in 0..2 {
            let started = std::time::Instant::now();
            let output = infer(&image, false, InferenceOptions { backend, device_id }, &NoProgress).unwrap();
            eprintln!(
                "BiRefNet attempt {attempt}: {:.3}s, provider={}, fallback={:?}",
                started.elapsed().as_secs_f64(),
                output.provider.name(),
                output.fallback_reason
            );
            if backend != Backend::Auto {
                assert_eq!(output.provider, backend);
                assert!(output.fallback_reason.is_none());
            }
            assert_eq!(output.matte.alpha.len(), MODEL_SIDE * MODEL_SIDE);
            assert!(output.matte.alpha.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)));
        }
        // Switching policy in the same process must replace the cached GPU session.
        let started = std::time::Instant::now();
        let output = infer(&image, false, InferenceOptions { backend: Backend::Cpu, device_id: 0 }, &NoProgress).unwrap();
        eprintln!("BiRefNet switch to CPU: {:.3}s", started.elapsed().as_secs_f64());
        assert_eq!(output.provider, Backend::Cpu);
        assert!(output.device_id.is_none());
        assert!(output.fallback_reason.is_none());
    }
    #[test]
    fn automatic_gpu_fallback_is_reported_and_explicit_gpu_failure_is_preserved() {
        let options = InferenceOptions { backend: Backend::Auto, device_id: 0 };
        let mut attempts = Vec::new();
        let (value, provider, reason) = select_session(options, true, |provider| {
            attempts.push(provider);
            if provider == Backend::DirectMl { Err(Error::Runtime("GPU allocation failed".into())) } else { Ok(42) }
        })
        .unwrap();
        assert_eq!(value, 42);
        assert_eq!(attempts, [Backend::DirectMl, Backend::Cpu]);
        assert_eq!(provider, Backend::Cpu);
        assert_eq!(reason.as_deref(), Some("GPU allocation failed"));
        attempts.clear();
        let result = select_session(InferenceOptions { backend: Backend::DirectMl, ..options }, true, |provider| {
            attempts.push(provider);
            Err::<(), _>(Error::Runtime("GPU unavailable".into()))
        });
        assert!(result.is_err());
        assert_eq!(attempts, [Backend::DirectMl]);
    }

    #[test]
    fn automatic_selects_gpu_when_ready_and_never_retries_cancellation() {
        let options = InferenceOptions::default();
        let (_, provider, reason) = select_session(options, true, |_| Ok(())).unwrap();
        assert_eq!(provider, Backend::DirectMl);
        assert!(reason.is_none());
        let mut attempts = 0;
        let result = select_session(options, true, |_| {
            attempts += 1;
            Err::<(), _>(Error::Cancelled)
        });
        assert!(matches!(result, Err(Error::Cancelled)));
        assert_eq!(attempts, 1);
        let (_, provider, _) = select_session(options, false, |_| Ok(())).unwrap();
        assert_eq!(provider, Backend::Cpu);
        let (_, provider, _) = select_session(InferenceOptions { backend: Backend::Cpu, ..options }, true, |provider| {
            assert_eq!(provider, Backend::Cpu);
            Ok(())
        })
        .unwrap();
        assert_eq!(provider, Backend::Cpu);
    }
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
