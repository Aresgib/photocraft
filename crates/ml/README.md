# photocraft-ml

Optional local inference at L4. `InferenceBackend::matte_subject` takes finite straight sRGB
RGBA at 1024×1024 and returns soft foreground coverage. The engine converts native document
samples through ICC color management, resamples the matte into an F32 layer mask, and owns
history. No Python process, cloud inference or user-image upload is involved.

The `onnx` feature uses `ort` 2.0.0-rc.10 / ONNX Runtime 1.22 on native x86_64 and aarch64.
The Windows prebuilt also requires `DirectML.dll` beside the executable (Cargo copies it;
the Windows packaging scripts include it and the notices in `licenses/`). Inference supports
Automatic, CPU and Windows DirectML, with ONNX telemetry explicitly disabled.
Automatic prefers the adapter with the most dedicated video memory and reports CPU fallback
when GPU initialization or execution fails. An explicit GPU request returns errors instead.
`infer_with_options` returns the actual provider, DXGI adapter index and optional fallback reason.
One cached session is rebuilt when the policy or adapter changes. Rendering on wgpu does not
select the inference device. Other native platforms use CPU; CUDA is not bundled.
WebAssembly and other architectures return an actionable unsupported error and continue to
provide the existing selection/background tools. Engine builds enable it through the
`ai-background` feature (enabled by default; `--no-default-features` omits it).

BiRefNet lite FP32 is pinned to ONNX Community export `de15b22`, checked against SHA-256 before
loading. Its output is logits: apply sigmoid once, preserve fractional coverage and bilinearly
resample; do not threshold it into a selection. Input uses ImageNet mean/std normalization.
Weights are MIT licensed; the full notice is in `assets/models/LICENSE-BiRefNet.txt` and is
written beside the installed model.

The tool offers a one-time download (224 MB). Engine calls default to `download:false`, so
headless automation never implicitly fetches weights. `PHOTOCRAFT_MODEL_DIR` overrides the cache;
otherwise Windows uses `%LOCALAPPDATA%/PhotoCraft/models`, other native systems use
`$XDG_CACHE_HOME/PhotoCraft/models` or `$HOME/.cache/PhotoCraft/models`. A partial download is
removed on cancellation/error. The model session is reused until the process exits.
An inference cancellation monitor terminates ONNX execution between operators; cancelled
results never become a document edit. Loading the model finishes before releasing its session.

CPU inference is memory intensive and is experimental. Test your images, especially fine hair,
horns, translucent objects and low-contrast edges; use the preview's Keep/Remove brush when
the model misses a region. Additional GPU providers, model management in Preferences, prompt-guided
segmentation and full-resolution edge matting remain future work.

Tests: `cargo test -p photocraft-ml --features onnx`; offline acceptance:
`PHOTOCRAFT_MODEL_DIR=<cache> PHOTOCRAFT_AI_TEST_IMAGE=<image> cargo test -p photocraft-engine
--test ai_background_model -- --ignored --nocapture`.
The acceptance image must contain background. Optional `PHOTOCRAFT_AI_TEST_CROP=[x,y,width,height]`
removes screenshot chrome before the test; no test image or weights are committed.
