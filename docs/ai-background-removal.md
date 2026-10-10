# AI Background Removal

Choose the **AI Background Removal Tool** (sparkles icon directly below Move in the toolbar).
Select an unlocked pixel layer, enable the one-time BiRefNet download if needed, and choose
**Generate preview**. Inference runs locally as a cancellable background job.
The existing Select Subject and Remove Background commands keep their previous behavior.

The preview does not change the document. Switch between Cutout, Overlay and Original, then
paint **Keep** over missed parts or **Remove** over unwanted background. Brush radius is in
document pixels. **Apply mask** installs the preview as an editable layer mask in one undo
step; it retains the layer's pixels, replacing its previous raster mask if present. Undo
restores the previous mask too. Closing the window or Cancel discards the preview and cancels
this tool's running job. An edited image requires a new preview before applying.

The first model download is approximately 224 MB. Nothing about the open image is sent to a
server. The weight export is pinned and SHA-256 verified; see `crates/ml/README.md` for cache,
licensing, build feature and supported architecture details. The browser retains the existing
tools and reports that this native ONNX tool is unavailable.

## AI processing device

The tool window and **Edit → Preferences → Performance → AI processing** share a saved setting:

- **Automatic (recommended)**: on Windows, try DirectML on the DXGI adapter with the most dedicated
  video memory; fall back to CPU if GPU session creation or execution fails. The preview reports
  the device used and the fallback reason. On other native platforms, use CPU.
- **CPU**: always run local inference on the CPU.
- **GPU (DirectML)**: on Windows, use the chosen adapter. Expand **AI GPU settings** to select
  its name/index. An unavailable GPU returns an error; it does not silently select CPU.

Changes apply to the next generated preview, without a restart. Existing previews retain their
mask and device information until regenerated. The canvas renderer is configured independently.
DirectML can execute unsupported model operators on CPU; selecting GPU does not guarantee that
every operator runs there. CUDA and other native GPU providers are not included in this build.

Agents can use `prefs.set {path:"performance.aiBackend", value:"auto|cpu|directml"}` and
`prefs.set {path:"performance.aiDeviceId", value:0}`. `layer.aiBackground.preview` also accepts
`backend` and `deviceId` overrides without changing the saved settings. `layer.aiBackground.info`
reports settings, DXGI adapter names/indices, and the preview's actual provider and fallback reason.

## Build and test on another computer

Use Rust 1.95 or newer and the platform prerequisites in `docs/development.md`.
On Windows, install the Visual Studio C++ build tools and Windows SDK. From the
repository root, run:

```powershell
cargo run -p photocraft
```

This compiles a development build and opens PhotoCraft; a release build or installer
is not needed to try the tool. Select the sparkles icon below Move and choose
**Generate preview**. The model is downloaded on that computer the first time you
permit it. Keep `DirectML.dll` beside the executable if moving a compiled Windows
build; the Windows packaging script includes the runtime and its license notices.

To validate the feature before use:

```powershell
cargo test -p photocraft-ml --features onnx
cargo test -p photocraft-engine ai_background_cmds
cargo test -p photocraft-ui-egui
cargo clippy -p photocraft-ml -p photocraft-engine -p photocraft-ui-egui --all-targets -- -D warnings
cargo xtask layers
cargo xtask wasm
cargo xtask parity
cargo test -p photocraft-engine --test panic_hunt -- --ignored
```

Engine tests and a real-model image test passed during development. The complete
desktop executable and visual UI verification remain pending after the development
machine ran out of disk space. The latest small changes also need these checks on
the destination computer.

All workflow operations are engine commands, available from CLI, control and MCP:

| Command | Parameters/result |
|---|---|
| `layer.aiBackground.preview` | `{download:false, layer:id?}` → editable preview (no document edit) |
| `layer.aiBackground.info` | `{}` → supported, ready, layer/dimensions/version/model/provider |
| `layer.aiBackground.correct` | `{points:[[x,y],…], radius:20, mode:"keep"}`; mode can be `remove` |
| `layer.aiBackground.apply` | `{}` → one reversible mask edit |
| `layer.aiBackground.cancel` | `{}` → discard preview and cancel this tool's jobs |

For the UI, `ui.set {tool:"AiBackground", aiBackground:{mode:"keep", radius:20,
view:"cutout", download:false}}` selects the tool and its settings. `ui.inspect.aiBackground`
reports them. Views are `cutout`, `overlay`, `original`. Settings affect the preview interface;
actual mask corrections and application still dispatch the engine commands above.
