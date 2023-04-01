use csv;

/// We derive Deserialize/Serialize so we can persist app state on shutdown.
#[derive(serde::Deserialize, serde::Serialize)]
#[serde(default)] // if we add new fields, give them default values when deserializing old state
pub struct BrowserBNSL {
    // Example stuff:
    label: String,

    // this how you opt-out of serialization of a member
    #[serde(skip)]
    value: f32,
    dropped_files: Vec<egui::DroppedFile>,
}

impl Default for BrowserBNSL {
    fn default() -> Self {
        Self {
            // Example stuff:
            label: "Hello World!".to_owned(),
            value: 2.7,
            dropped_files: Default::default(),
        }
    }
}

impl BrowserBNSL {
    /// Called once before the first frame.
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // This is also where you can customize the look and feel of egui using
        // `cc.egui_ctx.set_visuals` and `cc.egui_ctx.set_fonts`.

        // Load previous app state (if any).
        // Note that you must enable the `persistence` feature for this to work.
        if let Some(storage) = cc.storage {
            return eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default();
        }

        Default::default()
    }
    fn ui_file_drag_and_drop(&mut self, ctx: &egui::Context) {
        use egui::*;

        // Preview hovering files:
        if !ctx.input(|x| x.raw.hovered_files.is_empty()) {
            let mut text = "Dropping files:\n".to_owned();
            for file in &ctx.input(|x| x.raw.hovered_files.clone()) {
                if let Some(path) = &file.path {
                    text += &format!("\n{}", path.display());
                } else if !file.mime.is_empty() {
                    text += &format!("\n{}", file.mime);
                } else {
                    text += "\n???";
                }
            }

            let painter =
                ctx.layer_painter(LayerId::new(Order::Foreground, Id::new("file_drop_target")));

            let screen_rect = ctx.input(|x| x.screen_rect());
            painter.rect_filled(screen_rect, 0.0, Color32::from_black_alpha(192));
            painter.text(
                screen_rect.center(),
                Align2::CENTER_CENTER,
                text,
                TextStyle::Heading.resolve(&ctx.style()),
                Color32::WHITE,
            );
        }

        // Collect dropped files:
        if !ctx.input(|i| i.raw.dropped_files.is_empty()) {
            self.dropped_files = ctx.input(|i| i.raw.dropped_files.clone());
        }

        // Show dropped files (if any):
        if !self.dropped_files.is_empty() {
            let mut open = true;
            egui::Window::new("Data")
                .open(&mut open)
                .show(ctx, |ui| {
                    for file in &self.dropped_files {
                        let mut info = if let Some(path) = &file.path {
                            path.display().to_string()
                        } else if !file.name.is_empty() {
                            file.name.clone()
                        } else {
                            "???".to_owned()
                        };
                        let mut csv_info = String::new();
                        if let Some(bytes) = &file.bytes { // .bytes in browser, open from path if native
                            info += &format!(" ({} bytes)", bytes.len());
                            let mut rdr = csv::Reader::from_reader(std::str::from_utf8(bytes).unwrap().as_bytes());
                            csv_info += &format!("vars: {:?}", rdr.headers().unwrap());
                        } else if let Some(path) = &file.path {
                            let mut rdr = csv::Reader::from_path(path);
                        }
                        ui.label(info);
                        ui.label(csv_info);
                    }
                });
            if !open {
                self.dropped_files.clear();
            }
        }
    }
}

impl eframe::App for BrowserBNSL {
    /// Called by the frame work to save state before shutdown.
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, self);
    }

    /// Called each time the UI needs repainting, which may be many times per second.
    /// Put your widgets into a `SidePanel`, `TopPanel`, `CentralPanel`, `Window` or `Area`.
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        //let Self { label, value , dropped_files} = self;

        // Examples of how to create different panels and windows.
        // Pick whichever suits you.
        // Tip: a good default choice is to just keep the `CentralPanel`.
        // For inspiration and more examples, go to https://emilk.github.io/egui

        #[cfg(not(target_arch = "wasm32"))] // no File->Quit on web pages!
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            // The top panel is often a good place for a menu bar:
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Quit").clicked() {
                        _frame.close();
                    }
                });
            });
        });

        // egui::SidePanel::left("side_panel").show(ctx, |ui| {
        //     ui.heading("Side Panel");

        //     ui.horizontal(|ui| {
        //         ui.label("Write something: ");
        //         ui.text_edit_singleline(label);
        //     });

        //     ui.add(egui::Slider::new(value, 0.0..=10.0).text("value"));
        //     if ui.button("Increment").clicked() {
        //         *value += 1.0;
        //     }

        //     ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
        //         ui.horizontal(|ui| {
        //             ui.spacing_mut().item_spacing.x = 0.0;
        //             ui.label("powered by ");
        //             ui.hyperlink_to("egui", "https://github.com/emilk/egui");
        //             ui.label(" and ");
        //             ui.hyperlink_to(
        //                 "eframe",
        //                 "https://github.com/emilk/egui/tree/master/crates/eframe",
        //             );
        //             ui.label(".");
        //         });
        //     });
        // });
        self.ui_file_drag_and_drop(ctx);

        // egui::CentralPanel::default().show(ctx, |ui| {
        //     // The central panel the region left after adding TopPanel's and SidePanel's

        //     ui.heading("eframe template");
        //     ui.hyperlink("https://github.com/emilk/eframe_template");
        //     ui.add(egui::github_link_file!(
        //         "https://github.com/emilk/eframe_template/blob/master/",
        //         "Source code."
        //     ));
        //     egui::warn_if_debug_build(ui);
        // });

        if false {
            egui::Window::new("Window").show(ctx, |ui| {
                ui.label("Windows can be moved by dragging them.");
                ui.label("They are automatically sized based on contents.");
                ui.label("You can turn on resizing and scrolling if you like.");
                ui.label("You would normally choose either panels OR windows.");
            });
        }
    }
    
}
