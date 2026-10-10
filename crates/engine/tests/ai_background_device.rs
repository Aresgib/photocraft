//! Opt-in 24 MP device switching benchmark with a synthetic document and cached weights.
use photocraft_engine::Session;
use photocraft_engine::ai_background_cmds::{CANCEL, INFO, PREPARE, current_preview};
use photocraft_ml::Backend;
use serde_json::json;

#[test]
#[ignore = "requires cached BiRefNet and a DirectML GPU; no network or personal images"]
fn ai_device_switching_on_24mp_document() {
    let mut s = Session::new();
    s.execute("file.new", json!({"width":6000,"height":4000})).unwrap();
    let original = s.active().unwrap().doc.clone();
    let adapter = photocraft_ml::gpu_adapters().iter().max_by_key(|a| a.dedicated_memory_bytes).expect("DirectML adapter");
    for backend in ["cpu", "directml", "auto"] {
        s.execute("prefs.set", json!({"values":{"performance.aiBackend":backend,"performance.aiDeviceId":adapter.device_id}})).unwrap();
        for attempt in 0..2 {
            let started = std::time::Instant::now();
            let result = s.execute(PREPARE, json!({"download":false})).unwrap();
            eprintln!("24MP {backend} attempt {attempt}: {:.3}s; {result}", started.elapsed().as_secs_f64());
            let preview = current_preview(&s).unwrap();
            assert!(preview.mask.sample_channel(5999, 3999, 0).is_finite());
            assert!(std::sync::Arc::ptr_eq(&original, &s.active().unwrap().doc));
            if backend != "auto" {
                assert_eq!(preview.provider, Backend::parse(backend).unwrap());
                assert!(preview.fallback_reason.is_none());
            }
            assert_eq!(s.execute(INFO, json!({})).unwrap()["backend"], backend);
            s.execute(CANCEL, json!({})).unwrap();
        }
    }
}
