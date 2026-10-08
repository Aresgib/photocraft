//! Opt-in real-model acceptance and timing; the default test suite needs neither network nor weights.
use photocraft_engine::ai_background_cmds::{APPLY, PREPARE, current_preview, mask_from_matte};
use photocraft_engine::{Session, jobs::JobCtx};
use photocraft_geom::Rect;
use photocraft_ml::Matte;
use serde_json::json;

#[test]
#[ignore = "requires cached pinned BiRefNet and PHOTOCRAFT_AI_TEST_IMAGE; no network download"]
fn local_birefnet_preview_apply_and_export() {
    let image = std::env::var("PHOTOCRAFT_AI_TEST_IMAGE").expect("set PHOTOCRAFT_AI_TEST_IMAGE");
    let bytes = std::fs::read(&image).unwrap();
    let imported = photocraft_io::import(&image, &bytes).unwrap();
    let mut session = Session::new();
    session.add_document(imported.document, None);
    if let Ok(crop) = std::env::var("PHOTOCRAFT_AI_TEST_CROP") {
        let rect: Vec<i32> = serde_json::from_str(&crop).expect("crop must be [x,y,width,height]");
        assert_eq!(rect.len(), 4);
        session.execute("image.crop", json!({"x":rect[0],"y":rect[1],"width":rect[2],"height":rect[3]})).unwrap();
    }
    let before = session.active().unwrap().doc.clone();
    let start = std::time::Instant::now();
    let result = session.execute(PREPARE, json!({"download":false})).unwrap();
    eprintln!("BiRefNet cold preview: {:.2}s; {result}", start.elapsed().as_secs_f64());
    assert!(std::sync::Arc::ptr_eq(&before, &session.active().unwrap().doc));
    let p = current_preview(&session).unwrap();
    let bounds = before.bounds();
    eprintln!(
        "coverage corner={}, center={}, near horn={}",
        p.mask.sample_channel(2, 2, 0),
        p.mask.sample_channel(bounds.x1 / 2, bounds.y1 / 2, 0),
        p.mask.sample_channel(60, 110, 0)
    );
    let mask = photocraft_doc::LayerMask { surface: p.mask.clone(), ..photocraft_doc::LayerMask::reveal_all() };
    assert!(p.mask.sample_channel(0, 0, 0).is_finite());
    session.execute(APPLY, json!({})).unwrap();
    let id = session.active().unwrap().active_layer.unwrap();
    assert_eq!(session.active().unwrap().doc.layer(id).unwrap().mask.as_ref(), Some(&mask));
    let composite = photocraft_compose::render(&session.active().unwrap().doc, session.active().unwrap().doc.bounds());
    let min_alpha = composite.px.iter().map(|px| px[3]).fold(1.0f32, f32::min);
    eprintln!("composite minimum alpha={min_alpha}");
    assert!(min_alpha < 0.1, "acceptance image needs a removed background");
    let output = photocraft_io::export(&session.active().unwrap().doc, "cutout.png", &Default::default()).unwrap();
    let cutout = photocraft_io::import("cutout.png", &output.bytes).unwrap();
    assert_eq!(cutout.document.size, before.size);
    assert!(
        photocraft_compose::render(&cutout.document, cutout.document.bounds()).px.iter().any(|p| p[3] < 0.1),
        "PNG export must keep the removed background transparent"
    );
    if let Some(path) = std::env::var_os("PHOTOCRAFT_AI_TEST_OUTPUT") {
        std::fs::write(path, &output.bytes).unwrap();
    }
    session.execute("edit.undo", json!({})).unwrap();
    assert_eq!(&*session.active().unwrap().doc, &*before);

    let start = std::time::Instant::now();
    session.execute(PREPARE, json!({"download":false})).unwrap();
    eprintln!("BiRefNet warm preview: {:.2}s", start.elapsed().as_secs_f64());
    let matte = Matte::new(1024, 1024, vec![0.75; 1024 * 1024]).unwrap();
    let start = std::time::Instant::now();
    let mask = mask_from_matte(&matte, Rect::new(0, 0, 6000, 4000), &JobCtx::new()).unwrap();
    eprintln!("24MP mask resampling: {:.2}s", start.elapsed().as_secs_f64());
    assert_eq!(mask.sample_channel(5999, 3999, 0), 0.75);
}
