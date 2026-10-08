//! Independent AI cutout workflow: prepare/correct/cancel do not enter document history;
//! applying the prepared F32 layer mask is one reversible document edit.
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use photocraft_cms::{Transform, builtin::Builtin};
use photocraft_color::{PixelFormat, SampleType};
use photocraft_doc::{Document, LayerContent, LayerId, LayerMask};
use photocraft_geom::Rect;
use photocraft_ml::{MODEL_SIDE, Matte, ModelInput, Progress};
use photocraft_raster::Surface;
use serde_json::{Value, json};

use crate::commands::{CommandSpec, layer_param};
use crate::{EngineError, Result, Session};

pub const PREPARE: &str = "layer.aiBackground.preview";
pub const APPLY: &str = "layer.aiBackground.apply";
pub const CORRECT: &str = "layer.aiBackground.correct";
pub const CANCEL: &str = "layer.aiBackground.cancel";
pub const INFO: &str = "layer.aiBackground.info";
static NEXT_VERSION: AtomicU64 = AtomicU64::new(1);

fn next_version() -> Result<u64> {
    NEXT_VERSION.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1)).map_err(|_| error("AI preview version limit reached"))
}

pub struct Preview {
    pub document: Arc<Document>,
    pub layer: LayerId,
    pub revision: u64,
    pub mask: Surface,
    /// The color-managed source used by inference also serves as a bounded UI thumbnail.
    pub input: ModelInput,
    pub version: u64,
}

fn error(s: impl Into<String>) -> EngineError {
    EngineError::Other(s.into())
}
fn ml_error(e: photocraft_ml::Error) -> EngineError {
    match e {
        photocraft_ml::Error::Cancelled => EngineError::Cancelled,
        e => error(e.to_string()),
    }
}
impl Progress for crate::jobs::JobCtx {
    fn cancelled(&self) -> bool {
        self.cancelled()
    }
    fn report(&self, f: f32, message: &str) {
        self.progress(f, message);
    }
}

fn check_layer(doc: &Document, id: LayerId) -> Result<()> {
    let layer = doc.layer(id).ok_or(EngineError::NoLayer(id))?;
    if !matches!(layer.content, LayerContent::Raster(_)) {
        return Err(error("AI background removal needs a pixel layer"));
    }
    let locks = doc.effective_locks(id);
    if locks.all || locks.pixels {
        return Err(error("The layer is locked"));
    }
    if doc.size.area() == 0 || doc.size.area() > photocraft_ml::MAX_IMAGE_PIXELS as u64 {
        return Err(error("AI background removal supports images up to 64 megapixels"));
    }
    Ok(())
}

fn enabled(s: &Session) -> std::result::Result<(), String> {
    if !photocraft_ml::supported() {
        return Err("AI background removal requires the native ai-background build".into());
    }
    if s.jobs().iter().any(|j| j.command == PREPARE) {
        return Err("AI background preview is already running".into());
    }
    let st = s.active().ok_or("no document open")?;
    check_layer(&st.doc, st.active_layer.ok_or("no active layer")?).map_err(|e| e.to_string())
}
fn ready(s: &Session) -> std::result::Result<(), String> {
    current_preview(s).map(|_| ()).map_err(|e| e.to_string())
}
fn always(_: &Session) -> std::result::Result<(), String> {
    Ok(())
}

pub fn current_preview(s: &Session) -> Result<&Preview> {
    let p = s.ai_background.as_ref().ok_or_else(|| error("Generate an AI background preview first"))?;
    let st = s.active().ok_or(EngineError::NoDocument)?;
    if st.doc.id != p.document.id || st.revision != p.revision || !Arc::ptr_eq(&st.doc, &p.document) || st.active_layer != Some(p.layer) {
        return Err(error("The image or active layer changed; generate a new AI preview"));
    }
    check_layer(&st.doc, p.layer)?;
    Ok(p)
}

/// Resample only the model input (1 MP), rather than allocating the entire document image.
/// Work in native channel values before the ICC transform, with straight alpha kept separate.
pub fn sample_input(doc: &Document, id: LayerId, side: usize, progress: &dyn Progress) -> Result<ModelInput> {
    if side == 0 || side > MODEL_SIDE {
        return Err(error("invalid model input size"));
    }
    check_layer(doc, id)?;
    let src = doc.layer(id).and_then(|l| l.surface()).ok_or(EngineError::NoLayer(id))?;
    let transform = Transform::new(&crate::color_cmds::document_profile(doc), Builtin::Srgb.profile(), photocraft_cms::Intent::RelativeColorimetric, true)
        .map_err(|e| error(e.to_string()))?;
    let channels = transform.inputs();
    if channels > src.channels() || channels > 8 {
        return Err(error("Unsupported layer color channels for AI inference"));
    }
    let mut rgba = Vec::with_capacity(side * side);
    let w = doc.size.width as f64;
    let h = doc.size.height as f64;
    for y in 0..side {
        progress.check().map_err(ml_error)?;
        let sy = ((y as f64 + 0.5) * h / side as f64 - 0.5).clamp(0.0, h - 1.0);
        let y0 = sy.floor() as i32;
        let ty = sy - y0 as f64;
        for x in 0..side {
            let sx = ((x as f64 + 0.5) * w / side as f64 - 0.5).clamp(0.0, w - 1.0);
            let x0 = sx.floor() as i32;
            let tx = sx - x0 as f64;
            let mut raw = [0.0; 8];
            let mut alpha: f32 = 0.0;
            for (xx, yy, weight) in [
                (x0, y0, (1.0 - tx) * (1.0 - ty)),
                ((x0 + 1).min(doc.size.width as i32 - 1), y0, tx * (1.0 - ty)),
                (x0, (y0 + 1).min(doc.size.height as i32 - 1), (1.0 - tx) * ty),
                ((x0 + 1).min(doc.size.width as i32 - 1), (y0 + 1).min(doc.size.height as i32 - 1), tx * ty),
            ] {
                let pixel = src.pixel(xx, yy);
                if pixel.iter().any(|v| !v.is_finite()) {
                    return Err(error("Image has non-finite samples"));
                }
                for (c, out) in raw.iter_mut().take(channels).enumerate() {
                    *out += pixel.get(c).copied().unwrap_or(0.0) * weight as f32;
                }
                alpha += if src.format().alpha { pixel.last().copied().unwrap_or(0.0) } else { 1.0 } * weight as f32;
            }
            let mut rgb = [0.0; 3];
            transform.eval(raw.get(..channels).ok_or_else(|| error("invalid channel count"))?, &mut rgb);
            let [r, g, b] = rgb;
            rgba.push([r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0), alpha.clamp(0.0, 1.0)]);
        }
    }
    Ok(ModelInput { rgba })
}

pub fn mask_from_matte(matte: &Matte, bounds: Rect, progress: &dyn Progress) -> Result<Surface> {
    let count = bounds.width().checked_mul(bounds.height()).ok_or_else(|| error("invalid mask size"))?;
    if bounds.is_empty() || count as usize > photocraft_ml::MAX_IMAGE_PIXELS {
        return Err(error("invalid or oversized mask"));
    }
    let mut fmt = PixelFormat::GRAY8;
    fmt.sample = SampleType::F32;
    let mut mask = Surface::new(fmt);
    for rect in bounds.tiles().map(|t| t.rect().intersect(&bounds)) {
        progress.check().map_err(ml_error)?;
        let mut data = Vec::with_capacity(rect.width() as usize * rect.height() as usize);
        for y in rect.y0..rect.y1 {
            for x in rect.x0..rect.x1 {
                let mx = ((x - bounds.x0) as f32 + 0.5) * matte.width as f32 / bounds.width() as f32 - 0.5;
                let my = ((y - bounds.y0) as f32 + 0.5) * matte.height as f32 / bounds.height() as f32 - 0.5;
                data.push(matte.sample(mx, my));
            }
        }
        mask.write_region(rect, &data);
    }
    Ok(mask)
}

fn prepare(s: &mut Session, params: &Value) -> Result<Value> {
    if !params.is_object() {
        return Err(error("AI preview params must be an object"));
    }
    let download = match params.get("download") {
        None => false,
        Some(Value::Bool(b)) => *b,
        _ => return Err(error("download must be a boolean")),
    };
    if !photocraft_ml::supported() {
        return Err(error(photocraft_ml::Error::Unsupported.to_string()));
    }
    if !download && !photocraft_ml::model_available() {
        return Err(error("BiRefNet is not installed; enable Download model once (about 224 MB)"));
    }
    let id = layer_param(s, params)?;
    let st = s.active().ok_or(EngineError::NoDocument)?;
    check_layer(&st.doc, id)?;
    let doc = st.doc.clone();
    let revision = st.revision;
    let version = next_version()?;
    crate::jobs::run(
        s,
        "AI Background Removal",
        true,
        move |ctx| {
            ctx.progress(0.01, "Preparing color-managed image");
            let input = sample_input(&doc, id, MODEL_SIDE, ctx)?;
            let matte = photocraft_ml::infer(&input, download, ctx).map_err(ml_error)?;
            ctx.progress(0.9, "Preparing editable mask");
            let mask = mask_from_matte(&matte, doc.bounds(), ctx)?;
            Ok(Preview { document: doc, layer: id, revision, mask, input, version })
        },
        |s, preview| {
            let result = json!({"preview":true,"layer":preview.layer.0,"model":"BiRefNet","provider":"CPU"});
            s.ai_background = Some(preview);
            Ok(result)
        },
    )
}

fn apply(s: &mut Session, p: &Value) -> Result<Value> {
    if !p.is_object() {
        return Err(error("AI apply params must be an object"));
    }
    let preview = current_preview(s)?;
    let id = preview.layer;
    let mask = preview.mask.clone();
    s.edit("AI Background Removal", move |doc, _| {
        crate::extra_cmds::background_to_layer_for_mask(doc, id);
        doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?.mask = Some(LayerMask { surface: mask, ..LayerMask::reveal_all() });
        Ok(())
    })?;
    s.ai_background = None;
    Ok(json!({"layer":id.0,"mask":true}))
}

fn correct(s: &mut Session, p: &Value) -> Result<Value> {
    current_preview(s)?;
    let radius = p
        .get("radius")
        .and_then(Value::as_f64)
        .filter(|v| v.is_finite() && (0.5..=500.0).contains(v))
        .ok_or_else(|| error("radius must be between 0.5 and 500 pixels"))?;
    let value = match p.get("mode").and_then(Value::as_str) {
        Some("keep") => 1.0,
        Some("remove") => 0.0,
        _ => return Err(error("mode must be keep or remove")),
    };
    let points = p
        .get("points")
        .and_then(Value::as_array)
        .filter(|a| !a.is_empty() && a.len() <= 256)
        .ok_or_else(|| error("points must contain 1..256 coordinate pairs"))?;
    let mut parsed = Vec::with_capacity(points.len());
    let bounds = current_preview(s)?.document.bounds();
    for point in points {
        let a = point.as_array().filter(|a| a.len() == 2).ok_or_else(|| error("invalid correction point"))?;
        let x = a.first().and_then(Value::as_f64).filter(|v| v.is_finite() && *v >= bounds.x0 as f64 && *v < bounds.x1 as f64);
        let y = a.get(1).and_then(Value::as_f64).filter(|v| v.is_finite() && *v >= bounds.y0 as f64 && *v < bounds.y1 as f64);
        parsed.push((x.ok_or_else(|| error("correction point outside image"))?, y.ok_or_else(|| error("correction point outside image"))?));
    }
    let segments: Vec<_> =
        if parsed.len() == 1 { parsed.iter().map(|v| (*v, *v)).collect() } else { parsed.windows(2).filter_map(|a| Some((*a.first()?, *a.get(1)?))).collect() };
    let stroke_rect = |ax: f64, ay: f64, bx: f64, by: f64| {
        Rect::new(
            (ax.min(bx) - radius).floor() as i32,
            (ay.min(by) - radius).floor() as i32,
            (ax.max(bx) + radius).ceil() as i32 + 1,
            (ay.max(by) + radius).ceil() as i32 + 1,
        )
        .intersect(&bounds)
    };
    let work = segments.iter().fold(0u64, |n, ((ax, ay), (bx, by))| {
        let r = stroke_rect(*ax, *ay, *bx, *by);
        n.saturating_add(r.width() as u64 * r.height() as u64)
    });
    if work > 8_000_000 {
        return Err(error("Correction stroke is too large; use shorter strokes"));
    }
    let version = next_version()?;
    let preview = s.ai_background.as_mut().ok_or_else(|| error("no AI preview"))?;
    for ((ax, ay), (bx, by)) in segments {
        let rect = stroke_rect(ax, ay, bx, by);
        let mut data = preview.mask.read_region(rect);
        let dx = bx - ax;
        let dy = by - ay;
        let length = dx * dx + dy * dy;
        for (i, out) in data.iter_mut().enumerate() {
            let x = rect.x0 as f64 + (i % rect.width() as usize) as f64 + 0.5;
            let y = rect.y0 as f64 + (i / rect.width() as usize) as f64 + 0.5;
            let t = if length > 0.0 { ((x - ax) * dx + (y - ay) * dy) / length } else { 0.0 }.clamp(0.0, 1.0);
            if (x - ax - t * dx).hypot(y - ay - t * dy) <= radius {
                *out = value;
            }
        }
        preview.mask.write_region(rect, &data);
    }
    preview.version = version;
    Ok(json!({"corrected":true,"version":preview.version}))
}

fn cancel(s: &mut Session, _: &Value) -> Result<Value> {
    for job in s.jobs().into_iter().filter(|j| j.command == PREPARE) {
        s.cancel_job(job.id);
    }
    s.ai_background = None;
    Ok(json!({"cancelled":true}))
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: INFO,
            label: "Inspect AI Background Preview",
            menu: &[],
            shortcut: None,
            params: "{} → {supported,ready,preview:{layer,width,height,version,model,provider}?}",
            enabled: always,
            run: |s, _| {
                Ok(json!({
                    "supported":photocraft_ml::supported(),
                    "ready":current_preview(s).is_ok(),
                    "preview":s.ai_background.as_ref().map(|p| json!({"layer":p.layer.0,"width":p.document.size.width,"height":p.document.size.height,"version":p.version,"model":"BiRefNet","provider":"CPU"}))
                }))
            },
            journal: false,
        },
        CommandSpec {
            id: PREPARE,
            label: "AI Background Removal",
            menu: &[],
            shortcut: None,
            params: r##"{"layer":id?,"download":bool=false} → {preview,layer,model,provider}; prepares an editable mask without changing the document"##,
            enabled,
            run: prepare,
            journal: false,
        },
        CommandSpec {
            id: CORRECT,
            label: "Correct AI Background Mask",
            menu: &[],
            shortcut: None,
            params: r##"{"points":[[x,y],…],"radius":px,"mode":"keep|remove"}; radius and mode are required"##,
            enabled: ready,
            run: correct,
            journal: false,
        },
        CommandSpec {
            id: APPLY,
            label: "Apply AI Background Mask",
            menu: &[],
            shortcut: None,
            params: "{} → {layer,mask}",
            enabled: ready,
            run: apply,
            journal: true,
        },
        CommandSpec {
            id: CANCEL,
            label: "Cancel AI Background Preview",
            menu: &[],
            shortcut: None,
            params: "{}",
            enabled: always,
            run: cancel,
            journal: false,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jobs::JobCtx;

    fn fixture(depth: u32) -> Session {
        let mut s = Session::new();
        s.execute("file.new", json!({"width":12,"height":10,"depth":depth})).unwrap();
        let st = s.active().unwrap();
        let id = st.active_layer.unwrap();
        let mask = mask_from_matte(&Matte::new(2, 2, vec![0.25; 4]).unwrap(), st.doc.bounds(), &JobCtx::new()).unwrap();
        s.ai_background = Some(Preview {
            document: st.doc.clone(),
            layer: id,
            revision: st.revision,
            mask,
            input: ModelInput { rgba: vec![[0.8, 0.2, 0.1, 1.0]; MODEL_SIDE * MODEL_SIDE] },
            version: 0,
        });
        s
    }

    #[test]
    fn preview_correction_apply_and_undo_at_each_depth() {
        for depth in [8, 16, 32] {
            let mut s = fixture(depth);
            let id = s.ai_background.as_ref().unwrap().layer;
            let original = s.active().unwrap().doc.clone();
            s.execute(CORRECT, json!({"points":[[2,2],[7,2]],"radius":1.5,"mode":"keep"})).unwrap();
            assert!(Arc::ptr_eq(&original, &s.active().unwrap().doc), "preview must not edit the document");
            assert_eq!(current_preview(&s).unwrap().mask.sample_channel(5, 2, 0), 1.0);
            assert_eq!(current_preview(&s).unwrap().mask.sample_channel(11, 9, 0), 0.25);
            s.execute(APPLY, json!({})).unwrap();
            let layer = s.active().unwrap().doc.layer(id).unwrap();
            assert_eq!(layer.mask.as_ref().unwrap().surface.format().sample, SampleType::F32);
            assert_eq!(layer.mask.as_ref().unwrap().value(11, 9), 0.25);
            assert_eq!(layer.surface(), original.layer(id).unwrap().surface());
            let (bytes, _) = photocraft_format::PcraftWriter::new().save_zip(&s.active().unwrap().doc, &Default::default()).unwrap();
            let restored = photocraft_io::import("test.pcraft", &bytes).unwrap().document;
            assert_eq!(restored.layer(id).unwrap().mask.as_ref().unwrap().value(11, 9), 0.25);
            assert!(s.ai_background.is_none());
            s.execute("edit.undo", json!({})).unwrap();
            assert_eq!(&*s.active().unwrap().doc, &*original);
            s.execute("edit.redo", json!({})).unwrap();
            assert!(s.active().unwrap().doc.layer(id).unwrap().mask.is_some());
        }
    }

    #[test]
    fn cancel_and_invalid_corrections_leave_pixels_and_history_alone() {
        let mut s = fixture(8);
        let before = s.active().unwrap().doc.clone();
        for p in [
            Value::Null,
            json!({}),
            json!({"points":[[999,0]],"radius":20,"mode":"keep"}),
            json!({"points":[[1,1]],"radius":-1,"mode":"keep"}),
            json!({"points":[[1,1]],"radius":1,"mode":"bogus"}),
        ] {
            assert!(s.execute(CORRECT, p).is_err());
        }
        s.execute(CANCEL, json!({})).unwrap();
        assert!(Arc::ptr_eq(&before, &s.active().unwrap().doc));
        assert!(s.ai_background.is_none());
        assert!(s.execute(APPLY, json!({})).is_err());
        assert!(s.execute(PREPARE, json!({"download":"yes"})).is_err());
    }

    #[test]
    fn rejects_stale_image_or_target_before_applying() {
        let mut s = fixture(16);
        s.execute("layer.new.layer", json!({"name":"other"})).unwrap();
        assert!(s.execute(APPLY, json!({})).is_err());
        assert!(s.active().unwrap().doc.walk().iter().all(|(_, _, l)| l.mask.is_none()));
    }

    #[test]
    fn resampling_keeps_soft_coverage_and_cancellation() {
        let matte = Matte::new(2, 1, vec![0.0, 1.0]).unwrap();
        let ctx = JobCtx::new();
        let mask = mask_from_matte(&matte, Rect::new(0, 0, 4, 1), &ctx).unwrap();
        assert_eq!(mask.sample_channel(1, 0, 0), 0.25);
        assert_eq!(mask.sample_channel(2, 0, 0), 0.75);
        ctx.cancel();
        assert!(matches!(mask_from_matte(&matte, Rect::new(0, 0, 4, 1), &ctx), Err(EngineError::Cancelled)));
    }

    #[test]
    fn prepares_color_managed_input_without_modifying_source() {
        let s = fixture(32);
        let st = s.active().unwrap();
        let input = sample_input(&st.doc, st.active_layer.unwrap(), 2, &JobCtx::new()).unwrap();
        assert_eq!(input.rgba.len(), 4);
        assert!(input.rgba.iter().flatten().all(|v| v.is_finite()));
        assert!(sample_input(&st.doc, st.active_layer.unwrap(), usize::MAX, &JobCtx::new()).is_err());
    }

    #[test]
    fn preprocessing_supports_gray_and_cmyk_and_preserves_documents() {
        for mode in ["image.mode.grayscale", "image.mode.cmyk", "image.mode.lab"] {
            let mut s = fixture(16);
            s.execute(mode, json!({})).unwrap();
            let st = s.active().unwrap();
            let before = st.doc.clone();
            let input = sample_input(&st.doc, st.active_layer.unwrap(), 2, &JobCtx::new()).unwrap();
            assert!(input.rgba.iter().flatten().all(|v| v.is_finite()));
            assert!(Arc::ptr_eq(&before, &s.active().unwrap().doc));
        }
    }
}
