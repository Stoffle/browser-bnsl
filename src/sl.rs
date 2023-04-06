use crate::int_utils::IndexSet;
use crate::scoring::{Score, VariableMap, ScoreLookup};
use csv;
use itertools::Itertools;
use std::{fs::File, rc::Rc};
//use std::time::{Duration, Instant};
use egui::DroppedFile;
use chrono::prelude::*;

#[derive(Clone)]
pub struct DataInfo {
    pub name: String,
    pub n_vars: usize,
    pub n_samples: usize
}

impl DataInfo {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_file(file: egui::DroppedFile) -> Option<DataInfo> {
        let path = file.path.clone();        
        let name = path?.file_name()?.to_string_lossy().to_string();
        if let Ok(rdr) = csv::Reader::from_path(file.path?) {
            let (n_vars, n_samples) = csv_info(rdr)?;
            return Some(DataInfo { name: name, n_vars: n_vars, n_samples: n_samples })
        } else {return None}
    }

    #[cfg(target_arch = "wasm32")]
    pub fn from_file(file: egui::DroppedFile) -> Option<DataInfo> {
        let name = file.name.clone();
        let binding = file.bytes?;
        let rdr = csv::Reader::from_reader(std::str::from_utf8(&binding).unwrap().as_bytes());
        let (n_vars, n_samples) = csv_info(rdr)?;
        return Some(DataInfo { name: name, n_vars: n_vars, n_samples: n_samples })
    }
}

fn csv_info<R: std::io::Read>(mut rdr: csv::Reader<R>) -> Option<(usize, usize)> {
    // returns (n_vars, n_samples)
    if rdr.headers().is_err() {return None;}
    let n_vars = (if let Ok(headers) = rdr.headers() {
        Some(headers.len())
    } else {None})?; // hacky way to get rid of the u-word
    let n_samples = rdr.records().count();
    if (n_vars <= 1) || (n_samples <= 1) {return None;} // some non-csv files get read in, this should catch most
    Some((n_vars, n_samples))
}

#[derive(Clone)]
pub enum SLState {
    Queued(DataInfo, egui::DroppedFile),
    // Waiting,
    // Running(Instant),
    Running(DataInfo, egui::DroppedFile, DateTime<Utc>),
    // Done(Duration, Option<String>),
    Done(DataInfo, chrono::Duration, Option<String>),
    Failed(DataInfo),
}
impl SLState {
    pub fn info(&self) -> DataInfo {
        match self {
            SLState::Queued (data_info, ..) => {data_info.clone()}
            SLState::Running(data_info, ..) => {data_info.clone()}
            SLState::Done   (data_info, ..) => {data_info.clone()}
            SLState::Failed (data_info, ..) => {data_info.clone()}
        }
    }
}

#[cfg(not(target_arch = "wasm32"))] // not browser, so should have file path
pub fn sl_wrapper(data_info: DataInfo, file: DroppedFile, pruning: bool, learn: bool) -> Option<SLState> {
    //if let Some(bytes) =  { // .bytes in browser, open from path if native
        if let Ok(rdr) = csv::Reader::from_path(file.path?){
            let mut score_table = ScoreTable::from_csv_reader(rdr);
            let start = Utc::now();
            let modelstring = score_table.compute(pruning, false, learn);
            let duration = Utc::now() - start;
            return Some(SLState::Done(data_info, duration, modelstring))
        } else {None}
        // let mut score_table = ;
        // let start = Instant::now();
    // } else {
    //     return SLState::Done(, None)
    // }
}

#[cfg(target_arch = "wasm32")]
pub fn sl_wrapper(data_info: DataInfo, file: DroppedFile, pruning: bool, learn: bool) -> Option<SLState> {
    let binding = file.bytes.unwrap();
    if let Ok(s) = std::str::from_utf8(&binding) {
        let rdr = csv::Reader::from_reader(s.as_bytes());
        let mut score_table = ScoreTable::from_csv_reader(rdr);
        let start = Utc::now();
        let modelstring = score_table.compute(pruning, false, learn);
        let duration = Utc::now() - start;
        return Some(SLState::Done(data_info, duration, modelstring))
    } else {return None}
    // let rdr = csv::Reader::from_reader(std::str::from_utf8(&binding).unwrap().as_bytes());
    // if let Some(path) = &file.path {
    //     let mut score_table = ScoreTable::from_csv(std::fs::File::open(path).unwrap());
    //     return Some(score_table.compute(false, true))
    // } else {
    //     return None
    // }
}


fn transpose<T>(mut v: Vec<Vec<T>>) -> Vec<Vec<T>> {
    // from https://stackoverflow.com/questions/64498617/how-to-transpose-a-vector-of-vectors-in-rust
    assert!(!v.is_empty());
    for inner in &mut v {
        inner.reverse();
    }
    (0..v[0].len())
        .map(|_| {
            v.iter_mut()
                .map(|inner| inner.pop().unwrap())
                .collect::<Vec<T>>()
        })
        .collect()
}

#[derive(Debug)]
pub struct VectorSet {
    data: Rc<Vec<Vec<u8>>>, // Data encoded as u8, array[variable_index][row_index]
    index_vec: Vec<usize>, // Vec of indices, split by configuration of variables in index_set (splits, sizes)
    data_dimensions: (usize, usize), // (variables, samples)
    levels: Vec<usize>, // number of levels per variable, i.e. max+1
    // splits: VectorSetNode,                  // Tree storing indices and lengths of each sub-vector
    splits: Vec<usize>, // start indices of sub-vectors
    sizes: Vec<usize>,  // sizes of sub-vectors
    index_set: IndexSet, // variables in set, with int representation
    // score computation stuff:
    entropy: Option<f64>,
    //entropy_index: usize,
    cardinality: usize
}
impl VectorSet {
    // fn new_from_self(&self, new_data: Vec<Vec<u8>>, new_spilts: Vec<usize>, new_sizes: Vec<usize>, new_index_set: int_utils::IndexSet) -> VectorSet {
    //     VectorSet{
    //         data: new_data,
    //         data_dimensions: self.data_dimensions,
    //         levels: self.levels.clone(),
    //         splits: new_spilts,
    //         sizes: new_sizes,
    //         index_set: new_index_set
    //     }
    // }
    pub fn from_data(data: Rc<Vec<Vec<u8>>>) -> VectorSet {
        let n_samples = data[0].len();
        let n_vars = data.len();
        let levels = data
            .iter()
            .map(|d| {
                d.iter()
                    .max()
                    .expect("failed to get number of levels")
                    .clone() as usize
                    + 1
            })
            .collect();
        VectorSet {
            data: data,
            index_vec: (0..n_samples).collect_vec(),
            data_dimensions: (n_vars, n_samples),
            levels,
            splits: vec![0],
            sizes: vec![n_samples],
            index_set: IndexSet::new(),
            entropy: Some(0f64),
            cardinality: 1usize,
        }
    }
    // pub fn from_csv(file: File) -> VectorSet {
    //     let mut rdr = csv::ReaderBuilder::new()
    //         .has_headers(true)
    //         .delimiter(b',')
    //         .escape(Some(b'\\'))
    //         .flexible(false)
    //         // .comment(Some(b'#'))
    //         .from_reader(file);
    //     let headers = rdr.headers().expect("unable to read csv headers");
    //     let n_vars: usize = headers.len();
    //     let mut data: Vec<Vec<u8>> = Vec::new();
    //     let mut levels: Vec<usize> = vec![0; n_vars];
    //     let mut n_samples: usize = 0;
    //     for result in rdr.deserialize() {
    //         // to do: use hashmap to convert arbitrary string to ints
    //         let record: Vec<u8> = result.expect("Failed to parse csv");
    //         for i in 0..n_vars {
    //             if record[i] as usize > levels[i]-1 {levels[i] = record[i] as usize + 1};
    //             // we add 1 to get levels from max (as we start with 0)
    //         }
    //         data.push(record);
    //         n_samples += 1;
    //     }

    //     println!("Loaded data: {} samples of {} variables", n_samples, n_vars);

    //     VectorSet{
    //         data,
    //         data_dimensions: (n_samples, n_vars),
    //         levels,
    //         splits: vec![0],
    //         sizes: vec![n_samples],
    //         index_set: int_utils::IndexSet::new(),
    //     }
    //}

    pub fn add_variable(&self, variable: usize, entropy_required: bool) -> VectorSet {
        if !entropy_required {
            // println!("adding {:#?} to {:#?}, entropy not required", variable, self.index_set.vars);
            return VectorSet {
                data: Rc::new(vec![Vec::new()]),
                index_vec: Vec::new(),
                data_dimensions: self.data_dimensions,
                levels: Vec::new(),
                splits: Vec::new(),
                sizes: Vec::new(),
                index_set: self.index_set.add_variable(variable),
                entropy: None,
                cardinality: 0
            };
        }
        let levels = self.levels[variable];
        //let mut new_data: Vec<Vec<u8>> = Vec::new(); //vec![Vec::with_capacity(self.data_dimensions.1); self.data_dimensions.0];
                                                     // let mut new_tree = self.tree.clone();
        let mut new_index_vec: Vec<usize> = Vec::with_capacity(self.index_vec.len());
        let mut new_spilts: Vec<usize> = Vec::new();
        let mut new_sizes: Vec<usize> = Vec::new();
        //for (split, size) in self.splits.into_iter().zip(self.sizes) {
        let mut split_index = 0;
        let mut entropy = 0f64;
        let data_col = &self.data[variable];
        let mut sub_vecs: Vec<Vec<usize>> = vec![Vec::with_capacity(self.sizes.iter().max().unwrap().clone()); levels];
        for split_idx in 0..self.splits.len() {
            let split = self.splits[split_idx];
            let size = self.sizes[split_idx];
            for row_idx in &self.index_vec[split..split + size] {
                let row_idx_copy = row_idx.clone();
                sub_vecs[data_col[row_idx_copy] as usize].push(row_idx_copy);
            }
            // for ref mut sub_vec in sub_vecs {
            for i in 0..levels {
                // iterate over these sub_vecs
                let size = sub_vecs[i].len();
                new_spilts.push(split_index);
                new_sizes.push(size);
                if size == 0 {
                    continue;
                }
                split_index += size;
                new_index_vec.append(&mut sub_vecs[i]); // leaves sub_vec empty
                let px = size as f64 / self.data_dimensions.1 as f64;
                entropy -= px.log2() * px;
            }
        }
        // println!("splits: {:#?}, sizes: {:#?}, data: {:#?}", new_spilts, new_sizes, new_data);
        //self.new_from_self(new_data, new_spilts, new_sizes, self.index_set.add_variable(variable))
        VectorSet {
            data: self.data.clone(),  // TODO: don't copy data, pass into function?
            index_vec: new_index_vec,
            data_dimensions: self.data_dimensions,      // |
            levels: self.levels.clone(),                // | TODO: remove these copies? can just be argument to compute_recursive()
            splits: new_spilts,
            sizes: new_sizes,
            index_set: self.index_set.add_variable(variable),
            entropy: Some(entropy),
            cardinality: (self.cardinality * levels)
        }
    }

    pub fn compute_recursive(
        &self,
        data: &Vec<Vec<u8>>,
        logn: f64,
        variables: Vec<usize>,
        entropies: &mut Vec<f64>,
        scores: &mut Vec<Vec<(f64, usize)>>,
        counter: &mut usize,
        output_queries: bool,
        pruning: bool,
        path_graph: &mut Option<PathGraph>,
        entropy_required_flags: &mut Vec<bool>,
    ) {
        // no need to return children, we want them to be owned by the function
        // check this entropy wasn't pruned
        // if it was, skip to filling in scores
        // change self to represent new set
        let entropy_required = if !pruning {
            true
        } else {
            entropy_required_flags[self.index_set.index]
        };

        if entropy_required {
            entropies[self.index_set.index] = self.entropy.unwrap_or_else(|| panic!("entropy expected but not computed: {:#?}", self.index_set));
        };
        // compute all scores which use last computed entropy (self.entropy):
        let n = self.data_dimensions.1 as f64;
        for (var, score_idx, entropy_idx) in self.index_set.previous_scores() {
            let mut score = f64::MIN;
            if entropy_required {
                let levels = self.levels[var];
                let parent_levels: usize = entropy_idx.vars.iter().map(|x| self.levels[*x]).product();
                let penalty: f64 = 0.5 * logn * ((levels - 1) * parent_levels) as f64; //(self.cardinality / levels) as f64 * levels as f64;
                score = - (self.entropy.unwrap() - entropies[entropy_idx.index]) * n - penalty
            }
            let mut index = entropy_idx.index;
            for parent_subset_score_index in score_idx.predecessors(){
                let (prev_score, prev_parset_idx) = scores[var][parent_subset_score_index.index];
                if prev_score > score {
                    score = prev_score;
                    index = prev_parset_idx;
                }
            }
            if index == entropy_idx.index {
                // score improved so score supersets' entropies can't be pruned
                // so, set flag for the superset entropy and anything on the recursion path to it
                self.index_set.remove_variable(var).require_parent_supersets(var, (0..self.data_dimensions.0).collect_vec(), entropy_required_flags)
            }
            scores[var][score_idx.index] = (score, index);
            // TO DO: if this score was pruned, just set it to the previous one immediately
            *counter += 1;

            // if output_queries && index == entropy_idx.index { // only output locally optimal scores as query
            if output_queries && entropy_required {
                println!("pa_size = {} | {} |{} |", score_idx.vars.len(), score_idx.vars.iter().format(" "), var);
            }
            if let Some(pg) = path_graph {
                pg.try_update(self.index_set.clone(), entropy_idx, var, score)
            }
        }
        let mut next_generation_vars: Vec<usize> = Vec::new();
        if !entropy_required && path_graph.is_none() {
            // no need to keep calling if children were pruned and not performing SL
            return;
        }
        for var in variables {
            let vector_set = self.add_variable(var, entropy_required);
            vector_set.compute_recursive(data, logn, next_generation_vars.clone(), entropies, scores, counter, output_queries, pruning, path_graph, entropy_required_flags);
            next_generation_vars.push(var);
        }
        // fill in scores;
        // for pruned scores, just copy the score+score_index from relevant parent
    }
}

pub struct ScoreTable {
    data: Vec<Vec<u8>>,
    entropies: Vec<f64>, // put this in a Cell or RefRell?
    scores: Vec<Vec<(f64, usize)>>,
    entropy_required_flags: Vec<bool>,
    // required: Vec<bool>, // which entropies cannot be pruned, i.e. prune b 
    variable_map: VariableMap,
    path_graph: Option<PathGraph>
}

impl ScoreTable {
    pub fn from_csv(file: File) -> ScoreTable {
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .delimiter(b',')
            .escape(Some(b'\\'))
            .flexible(false)
            // .comment(Some(b'#'))
            .from_reader(file);
        ScoreTable::from_csv_reader(rdr)
    }

    pub fn from_csv_reader<R: std::io::Read>(mut rdr: csv::Reader<R>) -> ScoreTable {
        let vars = rdr.headers().expect("unable to read csv headers");
        let n_vars: usize = vars.len();
        let mut variable_map = VariableMap::new(n_vars); //HashMap<String, usize> = HashMap::new();
        for var in vars {
            variable_map.insert(var.to_string());
        }
        let mut data: Vec<Vec<u8>> = Vec::new();
        let mut levels: Vec<usize> = vec![0; n_vars];
        let mut n_samples: usize = 0;
        for result in rdr.deserialize() {
            // to do: use hashmap to convert arbitrary string to ints
            let record: Vec<u8> = result.expect("Failed to parse csv");
            for i in 0..n_vars {
                if record[i] as usize + 1 > levels[i] {
                    levels[i] = record[i] as usize + 1
                };
                // we add 1 to get levels from max (as we start with 0)
            }
            data.push(record);
            n_samples += 1;
        }
        // println!("{:#?}", variable_map);
        let col_major_data: Vec<Vec<u8>> = transpose(data);

        // println!("Loaded data: {} samples of {} variables", n_samples, n_vars);

        let mut entropy_required_flags = vec![false; 2usize.pow(n_vars as u32)];
        entropy_required_flags[0] = true;
        for var in 0..n_vars {
            let index_set = IndexSet::new().add_variable(var);
            index_set.require_predecessors(&mut entropy_required_flags);
            for other_var in 0..n_vars {
                if var == other_var {
                    continue;
                }
                index_set.add_variable(other_var).require_predecessors(&mut entropy_required_flags);
            }
        }


        ScoreTable {
            data: col_major_data,
            entropies: vec![0.0; 2usize.pow(n_vars as u32)],
            scores: vec![vec![(0.0, 0); 2usize.pow(n_vars as u32 - 1)]; n_vars],
            entropy_required_flags,
            variable_map,
            path_graph: None
        }
    }

    pub fn compute(&mut self, pruning: bool, output_queries: bool, learn_structure: bool) -> Option<String> {
        let vector_set = VectorSet::from_data(Rc::new(self.data.clone()));
        // println!("initial vector_set {:#?}", vector_set);
        let variables: Vec<usize> = (0..self.data.len()).collect();
        let size = self.data.len() * 2usize.pow(self.data.len() as u32 - 1);
        let mut counter = 0usize;
        self.path_graph = match learn_structure {
            true => {Some(PathGraph::new(self.data.len()))}
            false => {None}
        };

        let logn: f64 = (self.data[0].len() as f64).log2();
        vector_set.compute_recursive(&self.data, logn, variables, &mut self.entropies, &mut self.scores, &mut counter, output_queries, pruning, &mut self.path_graph, &mut self.entropy_required_flags);

        // if let Some(pg) = &self.path_graph {
        //     // println!("{:#?}", pg.nodes);
        //     println!("Learnt structure: {:#?}", pg.get_modelstring(&self.variable_map, &self.scores));
        // }
        // println!("{:#?}", self.entropies);
        // println!("{:#?}", self.scores);
        if let Some(pg) = &self.path_graph {
            Some(format!("{:#?}", pg.get_modelstring(&self.variable_map, &self.scores)))
        } else {
            None
        }
        
    }
}
impl ScoreLookup for ScoreTable {
    fn lookup_score(&self, target: usize, allowed: usize) -> Score {
        let allowed_set = IndexSet::from_index(allowed);
        let parents_score_index = allowed_set.remove_variable_and_shift(target).index;
        let (value, parents) = self.scores[target][parents_score_index];
        let value = value as f32;
        return Score{value, parents};
    }
    fn get_variable_map(&self) -> &VariableMap {
        return &self.variable_map
    }
}

#[derive(Debug)]
pub struct PathGraph {
    nodes: Vec<PathGraphNode>,
    n_vars: usize
}
impl PathGraph {
    pub fn new(n_vars: usize) -> PathGraph {
        let mut nodes = vec![
            PathGraphNode{path_length: f64::MAX, var_added: usize::MAX}
            ; 2usize.pow(n_vars as u32)
            ];
        nodes[0].path_length = 0f64;
        PathGraph { nodes, n_vars }
    }
    pub fn try_update(&mut self, current_node: IndexSet, subset_node: IndexSet, var_added: usize, score: f64) {
        let new_total = self.nodes[subset_node.index].path_length - score; // -ve because score
        if new_total < self.nodes[current_node.index].path_length {
            self.nodes[current_node.index].path_length = new_total;
            self.nodes[current_node.index].var_added = var_added;
        }
    }
    pub fn recover_order(&self) -> Vec<usize> {
        let mut current_node = IndexSet::from_index(self.nodes.len() - 1);
        let mut order: Vec<usize> = Vec::new();
        for _ in 0..self.n_vars {
            let node = self.nodes[current_node.index];
            order.push(node.var_added);
            current_node = current_node.remove_variable(node.var_added);
        }
        order.reverse();
        order
    }
    pub fn get_modelstring(&self, var_map: &VariableMap, score_vec: &Vec<Vec<(f64, usize)>>) -> String {
        let mut modelstring = String::new();
        let mut allowed_parents: Vec<usize> = Vec::new();
        // println!("{:#?}", self.recover_order());
        for var in self.recover_order(){
            let mut current_set = allowed_parents.clone();
            current_set.push(var);
            let current_node = IndexSet::from_vec(current_set);
            let (_, parent_set) = score_vec[var][current_node.remove_variable_and_shift(var).index];
            modelstring += &var_map.get_modelstring_fragment(var, IndexSet::from_index(parent_set).vars);
            allowed_parents.push(var);
        }
        modelstring
    }
}
#[derive(Debug, Clone, Copy)]
pub struct PathGraphNode {
    path_length: f64,
    var_added: usize
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_vectorset() -> VectorSet {
        VectorSet {
            data: Rc::new(transpose(vec![
                vec![0u8, 1u8, 0u8],
                vec![0u8, 1u8, 1u8],
                vec![1u8, 1u8, 2u8],
                vec![1u8, 1u8, 3u8],
                vec![0u8, 0u8, 0u8],
                vec![0u8, 0u8, 1u8],
                vec![1u8, 0u8, 2u8],
                vec![1u8, 0u8, 3u8],
            ])),
            index_vec: (0..8).collect_vec(),
            data_dimensions: (3, 8),
            levels: vec![2, 3, 4],
            splits: vec![0],
            sizes: vec![8],
            index_set: IndexSet::new(),
            entropy: 0f64,
            cardinality: 0usize
        }
    }

    #[test]
    fn test_add_variable() {
        let vs = make_test_vectorset();
        let vs0 = vs.add_variable(0, true);
        // assert_eq!(
        //     vs0.data,
        //     vec![
        //         vec![0u8, 1u8, 0u8],
        //         vec![0u8, 1u8, 1u8],
        //         vec![0u8, 0u8, 0u8],
        //         vec![0u8, 0u8, 1u8],
        //         vec![1u8, 1u8, 2u8],
        //         vec![1u8, 1u8, 3u8],
        //         vec![1u8, 0u8, 2u8],
        //         vec![1u8, 0u8, 3u8],
        //     ],
        // );
        assert_eq!(
            vs0.index_vec,
            vec![0,1,4,5,2,3,6,7]
        );
        assert_eq!(vs0.splits, vec![0, 4]);
        assert_eq!(vs0.sizes, vec![4, 4]);
        assert_eq!(
            vs0.index_set,
            IndexSet {
                index: 1usize,
                vars: vec![0]
            }
        );
        assert_eq!(vs0.entropy.unwrap(), -1f64);
        let vs01 = vs0.add_variable(1, true);
        // assert_eq!(
        //     vs01.data,
        //     vec![
        //         vec![0u8, 0u8, 0u8],
        //         vec![0u8, 0u8, 1u8],
        //         vec![0u8, 1u8, 0u8],
        //         vec![0u8, 1u8, 1u8],
        //         vec![1u8, 0u8, 2u8],
        //         vec![1u8, 0u8, 3u8],
        //         vec![1u8, 1u8, 2u8],
        //         vec![1u8, 1u8, 3u8],
        //     ],
        // );
        assert_eq!(
            vs01.index_vec,
            vec![4,5,0,1,6,7,2,3]
        );
        assert_eq!(vs01.splits, vec![0, 2, 4, 4, 6, 8]);
        assert_eq!(vs01.sizes, vec![2, 2, 0, 2, 2, 0]);
        assert_eq!(
            vs01.index_set,
            IndexSet {
                index: 3usize,
                vars: vec![0, 1]
            }
        );
        assert_eq!(vs01.entropy.unwrap(), -2f64);
        let vs012 = vs01.add_variable(2, true);
        assert_eq!(vs012.splits, vec![0, 1, 2, 2, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 6, 6, 6, 7, 8, 8, 8, 8]);
        assert_eq!(vs012.sizes,    vec![1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0]);
        assert_eq!(vs012.entropy.unwrap(), -3f64);
    }

    // #[test]
    // fn test_path_graph() {
    //     let pg = PathGraph{
    //         nodes: vec![0, 1, 2, 3, 4].iter().map(|x| PathGraphNode{path_length: 2f64 + *x as f64, var_added: 2usize.pow(*x)-1}).collect(),
    //         n_vars: 5
    //     };
    //     assert_eq!(pg.recover_order(), vec![0,1,2,3,4])
    // }
}


