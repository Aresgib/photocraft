//! A separate toolbar tool. Native engine commands own inference, corrections and the commit.
use egui::{Color32, Sense, TextureHandle, TextureOptions, vec2};
use photocraft_engine::ai_background_cmds::{self, APPLY, CANCEL, CORRECT, PREPARE};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::{PhotocraftApp, Tool, theme::Tokens, widgets};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
pub struct Settings {
    pub mode: String,
    pub radius: f32,
    pub view: String,
    pub download: bool,
    #[serde(skip)]
    pub last_point: Option<[f64; 2]>,
}
impl Default for Settings {
    fn default() -> Self {
        Self { mode: "keep".into(), radius: 20.0, view: "cutout".into(), download: true, last_point: None }
    }
}

pub struct PreviewTexture {
    key: (u64, u64, u64, u64, String, Color32, Color32, Color32),
    texture: TextureHandle,
    size: egui::Vec2,
}

fn run(app: &mut PhotocraftApp, id: &str, p: serde_json::Value) {
    if let Err(e) = app.run(id, p) {
        app.ui.status = e;
        app.ui.status_error = true;
    }
}

pub fn options_bar(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    ui.label(tl!("AI Background Removal"));
    ui.add_enabled_ui(app.session.is_enabled(PREPARE) && !app.session.has_jobs(), |ui| {
        if widgets::secondary_button(ui, tl!("Generate preview"), 0.0).clicked() {
            run(app, PREPARE, json!({"download":app.ui.ai_background.download}));
        }
    });
}

/// Bounded preview; only regenerate when the mask/view/theme changes. Never scan the source
/// document per frame, and use the same mask that Apply will commit.
fn texture(app: &mut PhotocraftApp, ctx: &egui::Context) -> Option<(TextureHandle, egui::Vec2)> {
    let p = ai_background_cmds::current_preview(&app.session).ok()?;
    let t = Tokens::get(ctx);
    let key = (p.document.id.0, p.layer.0, p.revision, p.version, app.ui.ai_background.view.clone(), t.canvas, t.card, t.danger);
    if let Some(cached) = &app.ai_background_texture
        && cached.key == key
    {
        return Some((cached.texture.clone(), cached.size));
    }
    let scale = (420.0 / p.document.size.width.max(p.document.size.height) as f32).min(1.0);
    let w = (p.document.size.width as f32 * scale).round().max(1.0) as usize;
    let h = (p.document.size.height as f32 * scale).round().max(1.0) as usize;
    let mut pixels = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let mx = ((x as f64 + 0.5) * 1024.0 / w as f64).floor() as usize;
            let my = ((y as f64 + 0.5) * 1024.0 / h as f64).floor() as usize;
            let px = p.input.rgba.get(my.min(1023).saturating_mul(1024).saturating_add(mx.min(1023))).copied().unwrap_or([0.0; 4]);
            let sx = ((x as f64 + 0.5) * p.document.size.width as f64 / w as f64).floor() as i32;
            let sy = ((y as f64 + 0.5) * p.document.size.height as f64 / h as f64).floor() as i32;
            let a = p.mask.sample_channel(sx, sy, 0).clamp(0.0, 1.0);
            let checker = if (x / 8 + y / 8) % 2 == 0 { t.canvas } else { t.card };
            let mut rgb = [px[0], px[1], px[2]];
            if app.ui.ai_background.view == "overlay" {
                // Use the theme's error color so the overlay belongs to the active design system.
                let c = t.danger;
                for (v, b) in rgb.iter_mut().zip([c.r(), c.g(), c.b()]) {
                    *v = *v * (0.4 + 0.6 * a) + b as f32 / 255.0 * 0.6 * (1.0 - a);
                }
            } else if app.ui.ai_background.view != "original" {
                for (v, b) in rgb.iter_mut().zip([checker.r(), checker.g(), checker.b()]) {
                    *v = *v * a * px[3] + b as f32 / 255.0 * (1.0 - a * px[3]);
                }
            }
            pixels.push(Color32::from_rgb(
                (rgb[0].clamp(0.0, 1.0) * 255.0).round() as u8,
                (rgb[1].clamp(0.0, 1.0) * 255.0).round() as u8,
                (rgb[2].clamp(0.0, 1.0) * 255.0).round() as u8,
            ));
        }
    }
    let image = egui::ColorImage::new([w, h], pixels);
    let texture = ctx.load_texture("ai-background-preview", image, TextureOptions::LINEAR);
    let size = vec2(w as f32, h as f32);
    app.ai_background_texture = Some(PreviewTexture { key, texture: texture.clone(), size });
    Some((texture, size))
}

pub fn show(app: &mut PhotocraftApp, ctx: &egui::Context) {
    if app.ui.tool != Tool::AiBackground {
        return;
    }
    let mut open = true;
    egui::Window::new(tl!("AI Background Removal")).id(egui::Id::new("ai-background-tool")).open(&mut open).resizable(false).default_width(440.0).show(
        ctx,
        |ui| {
            ui.label(tl!("BiRefNet runs locally. Your image stays on this computer."));
            widgets::checkbox(ui, &mut app.ui.ai_background.download, tl!("Download model once (about 224 MB)"));
            let ready = ai_background_cmds::current_preview(&app.session).is_ok();
            let busy = app.session.has_jobs();
            ui.add_enabled_ui(!busy && app.session.is_enabled(PREPARE), |ui| {
                if widgets::secondary_button(ui, tl!("Generate preview"), 0.0).clicked() {
                    run(app, PREPARE, json!({"download":app.ui.ai_background.download}));
                }
            });
            if !app.session.is_enabled(PREPARE) {
                let reason = photocraft_engine::commands::find(PREPARE).and_then(|c| (c.enabled)(&app.session).err()).unwrap_or_default();
                ui.label(reason);
            }
            if !ready {
                if app.session.ai_background.is_some() {
                    ui.label(tl!("The image changed. Generate a new preview."));
                }
                ui.label(tl!("Generate a preview, correct missing parts, then apply the mask."));
                return;
            }
            ui.horizontal(|ui| {
                for (id, label) in [("cutout", tl!("Cutout")), ("overlay", tl!("Overlay")), ("original", tl!("Original"))] {
                    ui.selectable_value(&mut app.ui.ai_background.view, id.to_string(), label);
                }
            });
            ui.horizontal(|ui| {
                ui.selectable_value(&mut app.ui.ai_background.mode, "keep".into(), tl!("Keep"));
                ui.selectable_value(&mut app.ui.ai_background.mode, "remove".into(), tl!("Remove"));
                ui.label(tl!("Brush radius"));
                widgets::value_field(ui, &mut app.ui.ai_background.radius, 0.5..=500.0, "px", 64.0);
            });
            ui.label(tl!("Paint on the preview to keep or remove areas."));
            if let Some((texture, size)) = texture(app, ctx) {
                let available = (ctx.content_rect().height() - 260.0).max(100.0);
                let display = size * (available / size.y).min(1.0);
                let response = ui.add(egui::Image::new((texture.id(), display)).sense(Sense::click_and_drag()));
                if let Some(pos) = response.hover_pos() {
                    let radius = app.session.ai_background.as_ref().map_or(1.0, |p| app.ui.ai_background.radius * display.x / p.document.size.width as f32);
                    ui.painter().circle_stroke(pos, radius.max(1.0), egui::Stroke::new(1.0, Tokens::get(ctx).text));
                }
                if !busy
                    && (response.dragged() || response.clicked() || response.drag_started())
                    && let Some(pos) = response.interact_pointer_pos()
                {
                    if response.rect.contains(pos) {
                        let point = app.session.ai_background.as_ref().map(|p| {
                            [
                                ((pos.x - response.rect.left()) / response.rect.width() * p.document.size.width as f32) as f64,
                                ((pos.y - response.rect.top()) / response.rect.height() * p.document.size.height as f32) as f64,
                            ]
                        });
                        if let Some(point) = point {
                            let last = app.ui.ai_background.last_point.unwrap_or(point);
                            run(app, CORRECT, json!({"points":[last,point],"radius":app.ui.ai_background.radius,"mode":app.ui.ai_background.mode}));
                            app.ui.ai_background.last_point = Some(point);
                        }
                    } else {
                        app.ui.ai_background.last_point = None;
                    }
                }
                if !response.dragged() {
                    app.ui.ai_background.last_point = None;
                }
            }
            ui.horizontal(|ui| {
                ui.add_enabled_ui(!busy, |ui| {
                    if widgets::secondary_button(ui, tl!("Apply mask"), 0.0).clicked() {
                        run(app, APPLY, json!({}));
                        app.ai_background_texture = None;
                    }
                    if widgets::secondary_button(ui, tl!("Cancel"), 0.0).clicked() {
                        run(app, CANCEL, json!({}));
                        app.ai_background_texture = None;
                    }
                });
            });
        },
    );
    if !open {
        // Closing during inference cancels that tool's job as well as discarding the preview.
        for job in app.session.jobs().into_iter().filter(|j| j.command == PREPARE && j.state == "running") {
            app.session.cancel_job(job.id);
        }
        run(app, CANCEL, json!({}));
        app.ai_background_texture = None;
        app.ui.tool = Tool::Move;
    }
}
