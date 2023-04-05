pub fn int_diff_index(first: usize, second: usize) -> usize {
    // find first bit which is 1 in second but not first
    let diff = second & !first;
    if diff == 0 {
        panic!()
    }
    let mut idx = 0;
    loop {
        if 1 << idx == diff {
            return idx;
        }
        idx += 1;
    }
}

pub fn int_subset(first: usize, second: usize) -> bool {
    // first <= second
    first & second == first
}

#[derive(PartialEq, Eq, Debug, Clone)]
pub struct IndexSet {
    pub index: usize,
    pub vars: Vec<usize>,
}
impl IndexSet {
    pub fn new() -> IndexSet{
        IndexSet { index: 0usize, vars: Vec::new() }
    }
    pub fn from_vec(vars: Vec<usize>) -> IndexSet{
        let mut index = 0usize;
        for var in &vars {
            index += 1usize << var;
        }
        IndexSet{index, vars}
    }
    pub fn from_index(index: usize) -> IndexSet{
        let mut vars: Vec<usize> = Vec::new();
        for position in 0..(usize::BITS - index.leading_zeros()) {
            let x = 1usize << position;
            if x & index == x {
                vars.push(position as usize);
            }
        }
        IndexSet {index, vars}
    }
    pub fn add_variable(&self, var: usize) -> IndexSet{
        assert!(!self.vars.contains(&var), "Attempted to add duplicate variable {} to set {:#?}", var, self.vars);
        let mut vars = self.vars.clone();
        vars.push(var);
        let index = self.index + (1usize<<var);
        IndexSet {index, vars}
    }
    pub fn remove_variable(&self, var: usize) -> IndexSet{
        assert!(self.vars.contains(&var), "Attempted to remove variable {} not in set {:#?}", var, self.vars);
        let mut vars = self.vars.clone();
        vars.retain(|x| x != &var);
        let index = self.index - (1usize<<var);
        IndexSet {index, vars}
    }
    pub fn remove_variable_and_shift(&self, var: usize) -> IndexSet{
        // remove variable and shift everything past it
        // for indexing score vec, don't store it!
        // assert!(self.vars.contains(&var), "Attempted to remove variable {} not in set {:#?}", var, self.vars);
        let mut vars = self.vars.clone();
        vars.retain(|x| x != &var);
        vars = vars.iter().map(|x| {
            if x > &var {*x-1}
            else {*x}
        }).collect();
        IndexSet::from_vec(vars)
    }
    pub fn predecessors(&self) -> Vec<IndexSet> {
        let mut predecessors = Vec::new();
        for var in &self.vars {
            predecessors.push(self.remove_variable(*var))
        }
        predecessors
    }
    pub fn previous_scores(&self) -> Vec<(usize, IndexSet, IndexSet)> {
        // For each subset of &self:
        // which var was removed, parent set index (for score subvec), full set index (for entropy)
        let mut previous_scores = Vec::new();   // tuple of (var, score parents index, parents entropy index)
        for var in &self.vars {
            previous_scores.push((*var, self.remove_variable_and_shift(*var), self.remove_variable(*var)))
        }
        previous_scores
    }
    pub fn require_parent_supersets(&self, child: usize, all_vars: Vec<usize>, required_flags: &mut Vec<bool>) {
        let combined = self.add_variable(child);
        for var in all_vars {
            if (var == child) || self.vars.contains(&var){
                continue;
            }
            combined.add_variable(var).require_predecessors(required_flags);
        }
    }
    pub fn require_predecessors(&self, required_flags: &mut Vec<bool>) {
        let mut var_set = self.clone();
        var_set.vars.sort();
        while var_set.vars.len() > 0 {
            if required_flags[var_set.index]{
                // if it's already required, its predecessors also are, so stop early
                return;
            }
            required_flags[var_set.index] = true;
            var_set = var_set.remove_variable(var_set.vars[0]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_int_subset() {
        assert_eq!(int_subset(0b0usize, 0b1usize), true);
        assert_eq!(int_subset(0b01usize, 0b11usize), true);
        assert_eq!(int_subset(0b11usize, 0b01usize), false);
    }

    #[test]
    fn test_int_diff_index() {
        assert_eq!(int_diff_index(0b010, 0b011), 0);
        assert_eq!(int_diff_index(0b0, 0b100), 2);
    }
    #[test]
    fn test_indexset_from_vec() {
        assert_eq!(
            IndexSet::from_vec(vec![0,1,2,4]),
            IndexSet{index: 23, vars: vec![0,1,2,4]}
        )
    }
    #[test]
    fn test_indexset_from_empty_vec() {
        assert_eq!(
            IndexSet::from_vec(Vec::new()),
            IndexSet{index: 0, vars: Vec::new()}
        )
    }
    #[test]
    fn test_indexset_from_index() {
        assert_eq!(
            IndexSet::from_index(23),
            IndexSet{index: 23, vars: vec![0,1,2,4]}
        )
    }
    #[test]
    fn test_indexset_add_variable() {
        assert_eq!(
            IndexSet::from_index(23).add_variable(5),
            IndexSet::from_index(55)
        );
    }
    #[test]
    fn test_indexset_rm_variable() {
        assert_eq!(
            IndexSet::from_index(55).remove_variable(5),
            IndexSet::from_index(23)
        );
    }
   #[test]
   fn test_indexset_prev_scores() {
        assert_eq!(
            // IndexSet::from_index(23).remove_variable_and_shift(1),
            IndexSet::from_vec(vec![0,1,2,4]).remove_variable_and_shift(1),
            IndexSet::from_vec(vec![0,1,3])
        )
   }
}
