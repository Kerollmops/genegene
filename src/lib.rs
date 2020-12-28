use std::iter;

use rand::Rng;
use rand_distr::Distribution;
use rand_distr::weighted_alias::{AliasableWeight, WeightedAliasIndex};
use rayon::prelude::*;

pub struct Genetic<I, F, H, S, O> {
    population: Vec<I>,
    heuristics: Vec<H>,
    generation: u64,
    fitness: F,
    selection: S,
    operator: O,
}

impl<I, F, H, S, O> Genetic<I, F, H, S, O>
where
    F: Fitness<I, H> + Sync,
    I: Sync,
    H: Send,
{
    pub fn new(population: Vec<I>, fitness: F, selection: S, operator: O) -> Genetic<I, F, H, S, O> {
        let heuristics = population.par_iter().map(|i| fitness.fitness(i)).collect();
        Genetic { population, heuristics, generation: 0, fitness, selection, operator }
    }

    pub fn population(&self) -> &[I] {
        &self.population
    }

    pub fn heuristics(&self) -> &[H] {
        &self.heuristics
    }
}

impl<I, F, H, S, O> Iterator for Genetic<I, F, H, S, O>
where
    F: Fitness<I, H> + Sync,
    S: Selection<I, H>,
    O: Operator<I>,
    I: Sync,
    H: Ord + Clone + Send,
{
    type Item = (Generation, H);

    fn next(&mut self) -> Option<Self::Item> {
        let mut selection = self.selection.selection(&self.population, &self.heuristics);
        let remaining = self.population.len().saturating_sub(selection.len());
        let mut new_population = self.operator.complete(&self.population, remaining);

        new_population.append(&mut selection);
        let fitness = &self.fitness;
        self.heuristics = new_population.par_iter().map(|i| fitness.fitness(i)).collect();
        self.population = new_population;

        self.generation += 1;
        match self.heuristics.iter().max() {
            Some(heuristic) => Some((self.generation, heuristic.clone())),
            None => None,
        }
    }
}

pub trait Fitness<I, H> {
    fn fitness(&self, individual: &I) -> H;
}

impl<I, H> Fitness<I, H> for dyn Fn(&I) -> H {
    fn fitness(&self, individual: &I) -> H {
        (self)(individual)
    }
}

pub trait Selection<I, H> {
    fn selection(&mut self, population: &[I], heuristics: &[H]) -> Vec<I>;
}

impl<I, H> Selection<I, H> for dyn FnMut(&[I], &[H]) -> Vec<I> {
    fn selection(&mut self, population: &[I], heuristics: &[H]) -> Vec<I> {
        (self)(population, heuristics)
    }
}

pub trait Operator<I> {
    fn complete(&mut self, population: &[I], limit: usize) -> Vec<I>;
}

impl<I> Operator<I> for dyn FnMut(&[I], usize) -> Vec<I> {
    fn complete(&mut self, population: &[I], limit: usize) -> Vec<I> {
        (self)(population, limit)
    }
}

pub type Generation = u64;

pub struct RouletteWheel<'r, R: ?Sized> {
    pub rng: &'r mut R,
    pub take_limit_ratio: f64,
}

impl<R: Rng + ?Sized, I: Clone, H: AliasableWeight + Clone + Ord> Selection<I, H> for RouletteWheel<'_, R> {
    fn selection(&mut self, population: &[I], heuristics: &[H]) -> Vec<I> {
        let best = heuristics.iter().enumerate()
            .max_by_key(|(_, h)| h.clone())
            .map(|(i, _)| population[i].clone());

        let heuristics = WeightedAliasIndex::new(heuristics.to_vec()).unwrap();
        let limit = (population.len() as f64 * self.take_limit_ratio) as usize;
        let iter = iter::from_fn(|| {
            let i = heuristics.sample(self.rng);
            population.get(i).cloned()
        });

        best.into_iter().chain(iter).take(limit).collect()
    }
}
