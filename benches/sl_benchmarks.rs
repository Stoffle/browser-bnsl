use browser_bnsl::sl::ScoreTable;

use std::path::{Path, PathBuf};
use std::fs::File;
use std::time::Duration;
use criterion::{black_box, criterion_group, criterion_main, Criterion, SamplingMode};

fn make_score_table(path: PathBuf) -> ScoreTable {
    let file = match File::open(&path){
        Err(why) => panic!("couldn't open {:#?}: {}", path, why),
        Ok(file) => file,
    };
    ScoreTable::from_csv(file)
}

fn sl_benchmark(c: &mut Criterion) {
    let data_files = Path::new(file!()).parent().unwrap().join("data").read_dir().unwrap();
    let mut group = c.benchmark_group("bn repo SL benchmarks");
    // Configure Criterion.rs to detect smaller differences and increase sample size to improve
    // precision and counteract the resulting noise.
    group.sample_size(10).sampling_mode(SamplingMode::Flat).confidence_level(0.9).measurement_time(Duration::new(300, 0));
    // For each dataset
    for data_path in data_files {
        let data_path = data_path.unwrap();
        for (pruning, learning) in vec![
            (true, false),
            (true, true),
            (false, false),
            (false, true),
            ] {
                let mut score_table = make_score_table(data_path.path());
                group.bench_function(
                    &format!("{:#?}, pruning={:#?}, learning={:#?}", data_path.file_name(), pruning, learning),
                    |b| b.iter(|| score_table.clone().compute(pruning, false, learning))
                );
                
            }
        }
        group.finish();

    // c.bench_function(score_table.clone().compute(pruning, false, learn));
}
criterion_group!(benches, sl_benchmark);
criterion_main!(benches);

