use csv;
use crate::sl;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;
use chrono::prelude::*;
use egui_extras::{Column, TableBuilder};


/// We derive Deserialize/Serialize so we can persist app state on shutdown.
#[derive(serde::Deserialize, serde::Serialize)]
#[serde(default)] // if we add new fields, give them default values when deserializing old state
pub struct BrowserBNSL {
    // this how you opt-out of serialization of a member
    #[serde(skip)]
    sl_states: Vec<sl::SLState>,
    //dropped_files: Vec<egui::DroppedFile>,
    #[serde(skip)]
    busy: bool,
    #[serde(skip)]
    run_switch: bool,
    #[serde(skip)]
    learn_switch: bool,
    #[serde(skip)]
    prune_switch: bool,
    #[serde(skip)]
    tx: Sender<sl::SLState>, //<egui::DroppedFile>,
    #[serde(skip)]
    rx: Receiver<sl::SLState>, //<Option<String>>,
}

impl Default for BrowserBNSL{
    fn default() -> Self {
        let (tx, rx) = channel();
        Self {
            // Example stuff:
            //dropped_files: Default::default(),
            sl_states: Default::default(),
            busy: false,
            run_switch: false,
            learn_switch: true,
            prune_switch: true,
            tx,
            rx,
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
                // } else if !file.mime.is_empty() {
                //     text += &format!("\n{}", file.mime);
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
            for dropped_file in ctx.input(|i| i.to_owned().raw.take()).dropped_files {
                if let Some(data_info) = sl::DataInfo::from_file(dropped_file.clone()){
                    self.sl_states.push(sl::SLState::Queued(data_info, dropped_file))
                }
            };
            // self.dropped_files = ctx.input(|i| i.raw.dropped_files.clone());
        }

        // Show dropped files (if any):
        if !self.sl_states.is_empty() {
            let mut open = true;
            egui::Window::new("Data")
                .open(&mut open)
                .show(ctx, |ui| {
                    #[cfg(target_arch = "wasm32")]
                    self.do_work_wasm(ctx);
                    // ui.checkbox(&mut self.run_switch, "Running");
                    ui.checkbox(&mut self.prune_switch, "Enable pruning");
                    ui.checkbox(&mut self.learn_switch, "Learn structure");
                    if self.run_switch {
                        // Running, so stop button:
                        if ui.button("⏸").clicked() {
                            self.run_switch = false;
                        };
                    } else {
                        // Paused, so play button:
                        if ui.button("Run").clicked() {
                            self.run_switch = true;
                        };
                    }
                    //for state_vec in vec![&self.sl_queue, &self.sl_states]{

                    let mut table = TableBuilder::new(ui)
                        .striped(true)
                        .resizable(true)
                        .cell_layout(egui::Layout::left_to_right(egui::Align::Center))
                        //.column(Column::auto_with_initial_suggestion(10.0))
                        .column(Column::auto())
                        .column(Column::auto())
                        .column(Column::auto())
                        .column(Column::auto())
                        .column(Column::auto().at_most(ctx.available_rect().width()).clip(true))
                        //.column(Column::initial(100.0).range(40.0..=300.0))
                        //.column(Column::initial(100.0).at_least(40.0).clip(true))
                        //.column(Column::remainder())
                        .min_scrolled_height(0.0);
                    table
                        .header(20.0, |mut header| {
                            header.col(|ui| {
                                ui.strong("File");
                            });
                            header.col(|ui| {
                                ui.strong("Variables");
                            });
                            header.col(|ui| {
                                ui.strong("Samples");
                            });
                            header.col(|ui| {
                                ui.strong("Running time");
                            });
                            header.col(|ui| {
                                ui.strong("Result");
                            });
                        })
                        .body(|mut body|  {
                            body.rows(10.0, self.sl_states.len(), |row_index, mut row| {
                                let state = self.sl_states.get(row_index).unwrap();
                                let data_info: sl::DataInfo = state.info();
                                row.col(|ui| {
                                    ui.label(data_info.name);
                                });
                                row.col(|ui| {
                                    ui.label(format!("{}", data_info.n_vars));
                                });
                                row.col(|ui| {
                                    ui.label(format!("{}", data_info.n_samples));
                                });
                                match state {
                                    sl::SLState::Queued(_, _) => {
                                        row.col(|ui| {
                                            ui.label("Queued".to_owned());
                                        });
                                    }
                                    // sl::SLState::Waiting => "SL starting".to_owned(),
                                    #[cfg(not(target_arch = "wasm32"))]
                                    sl::SLState::Running(_, _, start_time) => {
                                        row.col(|ui| {
                                            ui.label(format!("Running for {}", format_duration(Utc::now() - *start_time)).to_owned());
                                        });
                                        row.col(|ui| {
                                            ui.label("".to_owned());
                                        });
                                    }
                                    #[cfg(target_arch = "wasm32")]
                                    sl::SLState::Running(_, _, start_time) => {
                                        row.col(|ui| {
                                            ui.label("Running...".to_owned());
                                        });
                                        row.col(|ui| {
                                            ui.label("".to_owned());
                                        });
                                    }
                                    sl::SLState::Done(_, duration, res) => {
                                        row.col(|ui| {
                                            ui.label(format_duration(duration.clone()));
                                        });
                                        if let Some(modelstring) = res {
                                            row.col(|ui| {
                                                ui.label(modelstring.clone());
                                            });
                                        } else {
                                            row.col(|ui| {
                                                ui.label("".to_owned());
                                            });
                                        }
                                    }
                                    sl::SLState::Failed(..) => {
                                        row.col(|ui| {
                                            ui.label("Run failed".to_owned());
                                        });
                                    }
                                }
                            });
                        });

                    
                    // for sl_state in &self.sl_states {
                    //     let info = match sl_state {
                    //         sl::SLState::Queued(data_info, _) => "Queued".to_owned(),
                    //         // sl::SLState::Waiting => "SL starting".to_owned(),
                    //         #[cfg(not(target_arch = "wasm32"))]
                    //         sl::SLState::Running(data_info, _, start_time) => format!("Running for {}", format_duration(Utc::now() - *start_time)).to_owned(),
                    //         #[cfg(target_arch = "wasm32")]
                    //         sl::SLState::Running(data_info, _, start_time) => "Running...".to_string(),
                    //         sl::SLState::Done(data_info, duration, res) => {
                    //             if let Some(modelstring) = res {
                    //                 format!("SL finished in {}, modelstring: {:?}", format_duration(duration.clone()), modelstring).to_owned()
                    //             } else {
                    //                 format!("Scoring finished in {}", format_duration(duration.clone()))
                    //             }
                    //         }
                    //     };
                    //     ui.label(info);
                    //     ctx.request_repaint();
                    // }
                    #[cfg(not(target_arch = "wasm32"))] // not browser so we can use threads
                    for i in 0..(self.sl_states.len()) {
                        match &self.sl_states[i] {
                            sl::SLState::Queued(data_info, file) => {
                                if !self.busy & self.run_switch {
                                    //self.sl_states[i] = sl::SLState::Running(Instant::now());
                                    let tx_clone = self.tx.clone();
                                    let f = file.clone();
                                    let data_info_clone = data_info.clone();
                                    let p_switch = self.prune_switch.clone();
                                    let l_switch = self.learn_switch.clone();
                                    //thread::spawn(move || {
                                    thread::Builder::new().name("structure learning".to_string()).spawn(move || {
                                        //tx_clone.send(sl::sl_wrapper(file.clone())); //sl::SLState::Queued(file)));
                                        if let Some(res) = sl::sl_wrapper(data_info_clone.clone(), f, p_switch, l_switch) {
                                            tx_clone.send(res)
                                        } else {tx_clone.send(sl::SLState::Failed(data_info_clone))}
                                        //tx_clone.send(sl::sl_wrapper(f)).unwrap();
                                    }).unwrap(); // OS-level thread spawning failures panic for now
                                    self.sl_states[i] = sl::SLState::Running(data_info.clone(), file.clone(), Utc::now());
                                    self.busy = true;
                                    ctx.request_repaint();
                                } else {
                                    ctx.request_repaint();
                                }
                            },
                            // sl::SLState::Waiting => {},
                            sl::SLState::Running(..) => {
                                if let Ok(done_state) = self.rx.try_recv() {
                                    self.sl_states[i] = done_state;
                                    self.busy = false;
                                }
                                ctx.request_repaint();
                            },
                            sl::SLState::Done(..) => {},
                            sl::SLState::Failed(..) => {},
                        }
                    }

                    


                    // for file in &self.dropped_files {
                    //     let mut info = if let Some(path) = &file.path {
                    //         path.display().to_string()
                    //     } else if !file.name.is_empty() {
                    //         file.name.clone()
                    //     } else {
                    //         "???".to_owned()
                    //     };
                    //     // let mut csv_info = String::new();
                    //     if let Some(bytes) = &file.bytes { // .bytes in browser, open from path if native
                    //         info += &format!(" ({} bytes)", bytes.len());
                    //     //     let rdr = csv::Reader::from_reader(std::str::from_utf8(bytes).unwrap().as_bytes());
                    //     //     // csv_info += &format!("vars: {:?}", rdr.headers().unwrap());
                    //     //     let mut score_table = sl::ScoreTable::from_csv_reader(rdr);
                    //     //     let modelstring = score_table.compute(false, true);
                    //     //     if let Some(modelstr) = modelstring {
                    //     //         csv_info += &modelstr;
                    //     //     }
                    //     // } else if let Some(path) = &file.path {
                    //     //     // let mut rdr = csv::Reader::from_path(path).unwrap();
                    //     //     // csv_info += &format!("vars: {:?}", rdr.headers().unwrap());
                    //     //     // let mut score_table = sl::ScoreTable::from_csv(std::fs::File::open(path).unwrap());
                    //     //     let mut score_table = sl::ScoreTable::from_csv_reader(csv::Reader::from_path(path).unwrap());
                    //     //     let modelstring = score_table.compute(false, true);
                    //     //     if let Some(modelstr) = modelstring {
                    //     //         csv_info += &modelstr;
                    //     //     }
                    //     }
                    //     ui.label(info);
                    //     // ui.label(csv_info);
                    // }
                });
            if !open {
                self.sl_states.clear();
            }
        }
    }
    #[cfg(target_arch = "wasm32")]
    fn do_work_wasm(&mut self, ctx: &egui::Context) {
        self.busy = false;
        //#[cfg(target_arch = "wasm32")]
        for i in 0..(self.sl_states.len()) {
            match &self.sl_states[i] {
                sl::SLState::Queued(data_info, file) => {
                    if !self.busy & self.run_switch{
                    self.sl_states[i] = sl::SLState::Running(data_info.clone(), file.clone(), Utc::now());
                    self.busy = true;
                    }
                },
                // sl::SLState::Waiting => {},
                sl::SLState::Running(data_info, file, _) => {// actually start running, frame after taking from queue
                if !self.busy & self.run_switch{
                    self.sl_states[i] = {
                        if let Some(res) = sl::sl_wrapper(data_info.clone(), file.clone(), self.prune_switch, self.learn_switch){
                            res
                        } else {sl::SLState::Failed(data_info.clone())}
                    };
                    ctx.request_repaint();
                }
                },
                sl::SLState::Done(..) => {},
                sl::SLState::Failed(..) => {},
            }
        }
    }

    fn sl_window(&mut self, ctx: &egui::Context) {
        unimplemented!()
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

    }
    
}

fn format_duration(duration: chrono::Duration) -> String {
    return format!("{:03}s", duration.num_milliseconds() as f64/1000f64)
}
