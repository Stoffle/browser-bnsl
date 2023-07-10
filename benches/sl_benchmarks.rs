use browser_bnsl::sl::ScoreTable;

use std::fs::read_dir;
use std::path::Path;
use criterion::{black_box, criterion_group, criterion_main, Criterion};


fn sl_benchmark(c: &mut Criterion) {
    let data_folder = Path::new("../data/");
    let data_files = read_dir(data_folder).unwrap();

    // For each dataset
    for data_path in data_files {
        println!("{:#?}", data_path.unwrap());
    }

    // let mut score_table = ScoreTable::from_csv(data_path);
    // c.bench_function(score_table.clone().compute(pruning, false, learn));
}
criterion_group!(benches, sl_benchmark);
criterion_main!(benches);

