
use crate::int_utils::int_subset;
use std::collections::HashMap;
use itertools::Itertools;


pub trait ScoreLookup {
    fn lookup_score(&self, target: usize, allowed: usize) -> Score;
    fn get_variable_map(&self) -> &VariableMap;
}

#[derive(Debug, Clone)]
pub struct Score {
    pub value: f32,
    // parents: HashSet<String>
    pub parents: usize,
}

fn create_score(s: &str, varmap: &mut VariableMap) -> Score {
    let data: Vec<&str> = s.split(' ').collect();

    let value_fromstr = data[0].parse::<f32>().unwrap();

    // let parents: HashSet<String> = data[2..].iter().copied().map(|x| String::from(x)).collect();
    let parents: usize = varmap.get_set_index(data[2..].to_vec());

    Score {
        value: value_fromstr,
        parents,
    }
}

// #[derive(Debug)]
// // pub struct ScoreTable(pub Vec<Vec<Score>>, pub VariableMap);
// pub struct ScoreTable {
//     pub scores: Vec<Vec<Score>>,
//     // required: Vec<bool>, // which entropies cannot be pruned, i.e. prune b 
//     pub variable_map: VariableMap, //HashMap<String, usize>,
// }

// impl FromStr for ScoreTable {
//     type Err = ParseErr;

//     fn from_str(s: &str) -> Result<ScoreTable, ParseErr> {
//         let data: Vec<&str> = s.split('\n').collect();
//         // let mut score_map: HashMap<String, Vec<Score>> = HashMap::new();
//         let n_vars = data[0].parse::<usize>()?;
//         let mut scores: Vec<Vec<Score>> = (0..n_vars).map(|_| vec![]).collect();
//         // let mut score_table: Vec<Vec<Score>> = Vec::with_capacity(n_vars);
//         // for _ in 0..n_vars {
//         //     score_table.push(vec!());
//         // }
//         let mut variable_map = VariableMap::new(n_vars);

//         let mut pos = 1;
//         for _var_idx in 0..n_vars {
//             let var_row: Vec<&str> = data[pos].split(' ').collect();
//             let var_idx = variable_map.get_idx(var_row[0]);
//             let var_n_scores: usize = var_row[1].parse::<usize>()?;

//             let mut scores_vec: Vec<Score> = data[pos + 1..pos + 1 + var_n_scores]
//                 .iter()
//                 .map(|x| create_score(x, &mut variable_map))
//                 .collect();
//             scores_vec.sort_unstable_by(|a, b| b.value.partial_cmp(&a.value).unwrap()); // sort scores descending (least -ve first)

//             scores[var_idx] = scores_vec;

//             pos += var_n_scores + 1;
//         }
//         Ok(Self{scores, variable_map})
//     }
// }

// impl ScoreLookup for ScoreTable {
//     fn lookup_score(&self, target: usize, allowed: usize) -> Score {
//         match self.scores.get(target) {
//             Some(scores) => {
//                 for score in scores {
//                     // if score.parents.is_subset(allowed) {return score.value}
//                     if int_subset(score.parents, allowed) {
//                         return score.clone();
//                     }
//                 }
//             }
//             None => {
//                 panic!("No score found for target:{}, allowed:{}", target, allowed);
//             }
//         }
//         // if self.0.get(target).is_some() {
//         //     let scores = self.0.get(target).unwrap();
//         //     for score in scores {
//         //         // if score.parents.is_subset(allowed) {return score.value}
//         //         if int_subset(score.parents, allowed) {
//         //             return score;
//         //         }
//         //     }
//         // }
//         panic!("No score found for target:{}, allowed:{}", target, allowed);
//     }
//     fn get_variable_map(&self) -> &VariableMap {
//         return &self.variable_map
//     }
// }
// impl ScoreTable {
//     pub fn heuristic(&self, node_idx: usize) -> f32 {
//         // for each 0 in binary repr of node_idx, add the best score
//         let mut score_total = 0.0;
//         for i in 0..self.variable_map.n_vars {
//             if !int_subset(1usize << i, node_idx) {
//                 score_total += self.scores.get(i).unwrap()[0].value
//             }
//         }
//         score_total
//     }
//     pub fn order_mapdag_modelstring(&self, order: &[usize]) -> String {
//         // Return modelstring representation of best DAG for given order
//         let mut modelstring = String::new();
//         for order_pos in 0..order.len() {
//             let var_name = self.variable_map.get_name(order[order_pos]); // &self.1.vector[order[order_pos]];
//             let allowed_parents_idx = self.variable_map.get_set_index_from_indices(order[..order_pos].to_vec());
//             let parent_idx_vec: Vec<usize> = self.variable_map.idx_to_set(
//                 self.lookup_score(order[order_pos], allowed_parents_idx)
//                     .parents,
//             );
//             if parent_idx_vec.is_empty() {
//                 eprintln!("{} no allowed parents", var_name);
//                 modelstring.push_str(&format!("[{}]", var_name));
//                 continue;
//             }
//             let parent_name_vec: Vec<String> = parent_idx_vec
//                 .into_iter()
//                 .map(|idx| self.variable_map.get_name(idx))
//                 .collect();
//             let parent_name_str: String = parent_name_vec.join(":");
//             eprintln!(
//                 "{} allowed parents {:?}, MAP parents {:?}",
//                 var_name, allowed_parents_idx, parent_name_str
//             );
//             modelstring.push_str(&format!("[{}|{}]", var_name, parent_name_str));
//         }
//         modelstring
//     }
// }

#[derive(Clone, Debug)]
pub struct VariableMap {
    map: HashMap<String, usize>,
    vector: Vec<String>,
    pub n_vars: usize,
}

impl VariableMap {
    pub fn new(n_vars: usize) -> VariableMap {
        let name_map: HashMap<String, usize> = HashMap::new();
        let name_vec: Vec<String> = Vec::with_capacity(n_vars);
        VariableMap {
            map: name_map,
            vector: name_vec,
            n_vars
        }
    }

    pub fn insert(&mut self, var: String) {
        let idx = self.vector.len();
        self.vector.push(var.clone());
        self.map.insert(var, idx);
    }

    pub fn get_idx(&mut self, var: &str) -> usize {
        if !self.map.contains_key(var) {
            self.insert(var.to_string())
        }
        *self.map.get(var).unwrap()
    }

    pub fn get_name(&self, idx: usize) -> String {
        self.vector[idx].clone()
    }

    pub fn get_set_index(&mut self, set: Vec<&str>) -> usize {
        let mut idx = 0usize;
        for var in set {
            idx += 1usize << self.get_idx(var);
        }
        idx
    }

    pub fn get_set_index_from_indices(&self, set: Vec<usize>) -> usize {
        let mut idx = 0usize;
        for var in set {
            idx += 1usize << var;
        }
        idx
    }

    pub fn idx_to_set(&self, set_idx: usize) -> Vec<usize> {
        let mut var_set: Vec<usize> = Vec::new();
        for var_idx in 0..self.n_vars {
            if int_subset(1usize << var_idx, set_idx) {
                var_set.push(var_idx)
            }
        }
        var_set
    }

    pub fn get_modelstring_fragment(&self, target: usize, parent_set: Vec<usize>) -> String {
        if parent_set.len() == 0 {
            format!("[{}]", self.get_name(target))
        } else {
            format!("[{}|{}]", self.get_name(target), parent_set.iter().map(|x| self.get_name(*x)).format(":"))
        }
    }
}

// #[derive(Debug)]
// pub struct DataIntVec {
//     data: Vec<u64>,                 // Data encoded as ints
//     first_bits: Vec<usize>,          // Which bit in the int corresponds to the start of each variable
//     n_values: Vec<usize>
//     // last_bits: Vec<usize>,          // Which bit in the int corresponds to the end of each variable (may be equal to first)
// }
// impl DataIntVec {
//     pub fn from_csv(file: File) -> DataIntVec {
//         let df = CsvReader::new(file)
//             .infer_schema(None)
//             .has_header(true)
//             .finish()
//             .unwrap();
//         let mut last_bit = 0;
//         let ncols = df.get_column_names().len();
//         let mut first_bits = vec![0usize; ncols];
//         let mut n_values = vec![0usize; ncols];
//         // let mut last_bits = vec![0usize; ncols];
//         let mut data: Vec<u64> = vec![0; df.height()];
//         for (i, col) in df.get_columns().iter().enumerate() {
//             let width = usize_ceil_log2!(col.unique().unwrap().len());
//             n_values[i] = width;
//             first_bits[i] = last_bit;
//             // last_bits[i] = last_bit + width - 1;
//             data = data.iter().zip(col.i64().unwrap().into_no_null_iter()).map(|(x,y)| x + ((y as u64) << last_bit)).collect();
//             last_bit += width;
//             assert!(last_bit < 64);
//             // println!("width (bits) {}", width);
//             // println!("shifted to {} bit", 1u64 << last_bit);
//             // println!("{:?}", col.name());
//             // println!("{:?}", usize_ceil_log2!(col.unique().unwrap().len()) );
//             // println!("{:?}", data);
//         }
//         DataIntVec{data, first_bits, n_values}
//     }
//     pub fn get_occurrences(&self, variables: Vec<usize>, values: Vec<u64>) -> Vec<bool> {
//         // get the bool vec (for counting) of where a configuration of a subset of the variables occurs
//         let mut compare_int = 0u64;
//         for (var, val) in variables.iter().zip(values) {
//             compare_int += val << self.first_bits[var.to_owned()];
//         }
//         println!("{}",compare_int);
//         let result: Vec<bool> = self.data.iter().map(|x| (x & compare_int) == compare_int).collect();
//         result
//     }
// }


// #[derive(Debug, Clone)]
// struct Node {
//     children: Option<Vec<Node>>,
//     count: usize,
// }
// impl Node {
//     fn recursive_new(levels: Vec<usize>) -> Node {
//         let mut levels = levels.clone();
//         match levels.pop() {
//             Some(n_children) => return Node{children:Some(vec![Node::recursive_new(levels);n_children]), count:0},
//             None => return Node{children:None, count:0},
//         }
//     }
//     fn increment_recursive(&mut self, mut values: VecDeque<usize>){
//         self.count += 1usize;
//         match values.pop_front() {
//             Some(value) => self.children.as_mut().expect("Value given for non-leaf node")[value].increment_recursive(values),
//             None => ()
//         }
//     }
//     fn counts(&self) -> Vec<usize> {
//         let mut counts: Vec<usize> = Vec::new();
//         if self.count > 0 { // skip branches with only 0 counts
//             match &self.children.as_ref() {
//                 None => counts.push(self.count),
//                 Some(children) => {
//                     // let children: &[Node] = &self.children.as_ref().unwrap()[..];
//                     let mut inner_counts: Vec<usize> = children.iter().map(|c| c.counts()).flatten().collect();
//                     counts.append(inner_counts.as_mut())
//                 }
//                 //running_count += &self.children.unwrap().as_ref().iter().map(|c| c.counts()).sum(),
//             }
//         }
//         counts
//     }
//     // fn increment(&mut self, idx: usize) -> NodeOrCount {
//     //     let child = &self.children.as_mut().unwrap()[idx];
//     //     match child.children {
//     //         Some(children) => NodeOrCount::Node(*child),
//     //         None => NodeOrCount::Count(child.count.expect("no count"))
//     //     }
//     // }
// }

// #[derive(Debug)]
// pub struct Dataset {
//     //df: DataFrame,
//     data: Vec<Var>,
//     len: usize,
//     pub entropies: Vec<f64>,
//     pub scores: Vec<Vec<f64>>,
//     pub var_map: VariableMap,
// }
// #[derive(Debug)]
// struct Var {
//     name: String,
//     data: Vec<i64>,
//     levels: usize,
// }

// impl Dataset {
//     pub fn from_csv(file: File) -> Dataset {
//         let df = CsvReader::new(file)
//             .infer_schema(None)
//             .has_header(true)
//             .finish()
//             .expect("Polars failed to read file as csv");
//         let mut data: Vec<Var> = Vec::new();
//         let n_vars: usize = df.get_column_names().len();
//         let mut var_map: VariableMap = VariableMap::new(n_vars);
//         for col in df.get_columns().iter() {
//             let name = col.name().to_owned();
//             var_map.insert(name.clone());
//             let col_array: Vec<i64> = col.i64().expect("Couldn't get chunkedarray from series").into_no_null_iter().collect();
//             //let levels = col.unique().unwrap().len();
//             let levels: usize = col.max::<i64>().unwrap() as usize + 1;
//             // TODO: map arbitrary strings to integer instead of assuming it's done
//             data.push(Var{name, data:col_array, levels});
//         }
//         let len = df.height();
//         let entropies: Vec<f64> = vec![0f64; 2usize.pow(n_vars as u32)];
//         let scores: Vec<Vec<f64>> = vec![vec![0f64; 2usize.pow(n_vars as u32 - 1)]; data.len()];
//         Dataset{data, len, entropies, scores, var_map}
//     }
//     fn get_counts(&self, variables: Vec<usize>) -> Vec<usize> {
//         // get the bool vec (for counting) of where a configuration of a subset of the variables occurs
//         let mut vars: Vec<&Var> = self.data.iter().enumerate().filter(|(i, _)| variables.contains(i)).map(|(_, x)| x).collect();
//         vars.sort_by_key(|a| a.levels); // ascending number of levels minimises ADTree size

//         let mut levels: Vec<usize> = Vec::new();
//         let mut data: Vec<&[i64]> = Vec::new();
//         for var in vars {
//             levels.push(var.levels.clone());
//             // println!("{} has {} levels", var.name, var.levels);
//             data.push(&var.data[..])
//         }
//         levels.reverse();
//         //data.reverse();

//         // let mut levels: Vec<usize> = vars.clone().iter().map(|v| v.levels).collect(); // don't want to clone everything, just levels
        
//         let mut adtree = Node::recursive_new(levels);
//         //println!("{:#?}", adtree);
//         // let data: Vec<VecDeque<u64>> = vars.iter().map(|v| VecDeque::from(v.data.to_owned())).collect(); // prefer to just read the values than copy

//         // let data = izip(vars.iter().map(|a| a.data)); // doesn't work for arbitrary number of variables?
//         // let mut data = vars.iter().map(|v| &v.data[..]);
//         for i in 0..self.len {
//             //let mut row: VecDeque<usize> = vars.iter().map(|v| v.data[i] as usize).collect();
//             let mut row: VecDeque<usize> = VecDeque::new(); // TODO: replace with vec, unnecessary overhead
//             for col in &data {
//                 row.push_back(col[i] as usize);
//             }
//             //println!("{:#?}", row);
//             adtree.increment_recursive(row);
//         }

//         //println!("{:#?}", adtree);
//         let counts = adtree.counts();
//         // let total_count: usize = counts.iter().sum();
//         // println!("Counts sum: {}, n_samples: {}", total_count, self.len);
//         counts
//     }

//     fn entropy(&mut self, variables: Vec<usize>) {
//         let counts = self.get_counts(variables.clone());
//         let mut entropy = 0f64;
//         for count in counts {
//             if count == 0 {continue;} // we take 0log(0) == 0
//             let frac: f64 = count as f64 / self.len as f64;
//             entropy += frac.log2() * frac;
//         }
//         self.entropies[self.var_map.get_set_index_from_indices(variables)] = entropy;
//     }

//     pub fn compute_entropies(&mut self) {
//         let size = 2usize.pow(self.data.len() as u32);
//         let pb = ProgressBar::new(size as u64);
//         let mut counter = 0usize;
//         for i in 1..size {
//             pb.inc(1);
//             let variables = self.var_map.idx_to_set(i);
//             if variables.len() > 8 {continue;}
//             self.entropy(variables);
//             counter += 1;
//         }
//         pb.finish_with_message("done");
//         println!("{} entropies finished in {:?}", counter, pb.elapsed());
//     }
// }


#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_variablemap() {
        let mut test_map = VariableMap::new(5);
        test_map.insert("a".to_string());
        test_map.insert("b".to_string());
        test_map.insert("c".to_string());
        assert_eq!(test_map.get_idx("a"), 0);
        assert_eq!(test_map.get_name(0), "a".to_string());
        assert_eq!(test_map.get_set_index(vec!(&"a".to_string())), 1);
        assert_eq!(
            test_map.get_set_index(vec!(&"b".to_string(), &"a".to_string())),
            0b11
        );
    }


}

