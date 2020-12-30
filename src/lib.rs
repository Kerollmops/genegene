pub struct Genetic<I, F, H, R> {
    population: Vec<I>,
    heuristics: Vec<H>,
    generation: u64,
    fitnesses: F,
    reproduce: R,
}

impl<I, F, H, R> Genetic<I, F, H, R>
where
    F: Fitnesses<I, H>,
{
    pub fn new(population: Vec<I>, mut fitnesses: F, reproduce: R) -> Genetic<I, F, H, R> {
        let heuristics = fitnesses.fitnesses(&population);
        Genetic { population, heuristics, generation: 0, fitnesses, reproduce }
    }

    pub fn population(&self) -> &[I] {
        &self.population
    }

    pub fn heuristics(&self) -> &[H] {
        &self.heuristics
    }
}

impl<I, F, H, R> Iterator for Genetic<I, F, H, R>
where
    F: Fitnesses<I, H>,
    R: Reproduce<I, H>,
    H: Ord + Clone,
{
    type Item = (Generation, H);

    fn next(&mut self) -> Option<Self::Item> {
        self.population = self.reproduce.reproduce(&self.population, &self.heuristics);
        self.heuristics = self.fitnesses.fitnesses(&self.population);
        self.generation += 1;
        match self.heuristics.iter().max().cloned() {
            Some(heuristic) => Some((self.generation, heuristic)),
            None => None,
        }
    }
}

pub trait Fitnesses<I, H> {
    fn fitnesses(&mut self, population: &[I]) -> Vec<H>;
}

impl<I, H> Fitnesses<I, H> for dyn FnMut(&[I]) -> Vec<H> {
    fn fitnesses(&mut self, population: &[I]) -> Vec<H> {
        (self)(population)
    }
}

pub trait Reproduce<I, H> {
    fn reproduce(&mut self, population: &[I], heuristics: &[H]) -> Vec<I>;
}

impl<I, H> Reproduce<I, H> for dyn FnMut(&[I], &[H]) -> Vec<I> {
    fn reproduce(&mut self, population: &[I], heuristics: &[H]) -> Vec<I> {
        (self)(population, heuristics)
    }
}

pub type Generation = u64;
