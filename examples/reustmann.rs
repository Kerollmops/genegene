use std::cmp::{self, Reverse};
use std::fmt;
use std::io::empty;

use bstr::ByteSlice as _;
use ordered_float::OrderedFloat;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use reustmann::instruction::op_codes;
use reustmann::instruction::{Instruction, OpCode};
use reustmann::Statement;
use reustmann::{Program, Interpreter};

use genegene::{Genetic, Fitnesses, Reproduce};

const ARCH_LENGTH: usize = 32;
const TARGET: &str = "Hello World!";
const INSTRUCTIONS: [Instruction; 46] = [
    Instruction::Nop,
    Instruction::Reset,
    Instruction::Halt,
    Instruction::In,
    Instruction::Out,
    Instruction::Pop,
    Instruction::Dup,
    Instruction::PushPc,
    Instruction::PopPc,
    Instruction::PopSp,
    Instruction::SpTgt,
    Instruction::PushNz,
    Instruction::Swap,
    Instruction::Push0,
    Instruction::Add,
    Instruction::Sub,
    Instruction::Inc,
    Instruction::Dec,
    Instruction::Mul,
    Instruction::Div,
    Instruction::Xor,
    Instruction::And,
    Instruction::Or,
    Instruction::Shl,
    Instruction::Shr,
    Instruction::Not,
    Instruction::Bz,
    Instruction::Bnz,
    Instruction::Beq,
    Instruction::Bgt,
    Instruction::Blt,
    Instruction::Bge,
    Instruction::Loop,
    Instruction::EndL,
    Instruction::BraN,
    Instruction::BraP,
    Instruction::Target,
    Instruction::Skip1,
    Instruction::Skip2,
    Instruction::Skip3,
    Instruction::Skip4,
    Instruction::Skip5,
    Instruction::Skip6,
    Instruction::Skip7,
    Instruction::Skip8,
    Instruction::Skip9,
];

fn random_instruction<R: Rng + ?Sized>(rng: &mut R) -> u8 {
    if rng.gen() {
        OpCode::from(*INSTRUCTIONS.choose(rng).unwrap())
    } else {
        rng.gen_range(32..127)
    }
}

type Heuristic = OrderedFloat<f64>;

#[derive(Clone)]
struct Individual {
    instructions: Vec<OpCode>,
}

impl Individual {
    pub fn from_rng<R: Rng + ?Sized>(rng: &mut R) -> Individual {
        let size = rng.gen_range(10..=ARCH_LENGTH);
        Individual { instructions: (0..=size).map(|_| random_instruction(rng)).collect() }
    }

    fn crossover<R: Rng + ?Sized>(&self, rng: &mut R, other: &Individual) -> Self {
        let parent0 = self.instructions.as_slice();
        let parent1 = other.instructions.as_slice();

        let (p0, p1) = {
            let min = cmp::min(parent0.len(), parent1.len());
            let p0 = rng.gen_range(0..min);
            let p1 = rng.gen_range(0..min);
            if p0 > p1 { (p1, p0) } else { (p0, p1) }
        };

        let mut instructions: Vec<_> = match rng.gen_range(0..4) {
            0 => // AaaaabbbbbaaaaaAaaaa
                parent0[0..p0].iter()
                .chain(&parent1[p0..p1])
                .chain(&parent0[p1..])
                .chain(&parent0[..p0])
                .cloned()
                .collect(),
            1 => // Aaaaabbbbbaaaaa
                parent0[..p0].iter()
                .chain(&parent1[p0..])
                .chain(&parent0[p1..])
                .cloned()
                .collect(),
            2 => // aaaaabbbbbAaaaa
                parent0[p0..p1].iter()
                .chain(&parent1[p1..])
                .chain(&parent0[p0..])
                .cloned()
                .collect(),
            _ => // aaaaabbbbb
                parent0[p0..p1].iter()
                .chain(&parent1[p1..])
                .cloned()
                .collect(),
        };

        instructions.truncate(ARCH_LENGTH);
        Individual { instructions }
    }

    fn mutate<R: Rng + ?Sized>(&mut self, rng: &mut R, mutation_rate: f64) {
        let num_instrs = self.instructions.len();
        if rng.gen::<f64>() < mutation_rate * num_instrs as f64 {
            let num_chars_to_mutate = mutation_rate * num_instrs as f64 + 0.5 + rng.gen::<f64>();
            for _ in 0..num_chars_to_mutate as usize {
                let instr = self.instructions.choose_mut(rng).unwrap();
                *instr = random_instruction(rng);

                // This replaces a char with a random char from the Iota opcode set:
                //ind.mGenome[idx] = programCharset[rand() % programCharset.size()];

                // or, alternatively, the following line replaces a char with any
                // random value that fits in the logical Iota memory width:
                //ind.mGenome[idx] = rand() % (1UL << mArchWidth);

                // or, alternatively, the following line replaces a char with:
                // any random *printable* char for a prettier display during evolution:
                // *instr = rng.gen_range(32..127);
                // } else {
                //     // or incr or decr the instruction by one:
                //     if rng.gen() { instr.wrapping_add(1) } else { instr.wrapping_sub(1) }
                // };
            }
        }
    }
}

impl fmt::Debug for Individual {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.instructions.as_bstr())
    }
}

fn program_output(instructions: &[u8]) -> Vec<u8> {
    let program = Program::from_iter(instructions.iter().cloned());

    let arch_length = ARCH_LENGTH; // memory length
    let arch_width = 8; // word size
    let mut interpreter = Interpreter::new(arch_length, arch_width).unwrap();
    interpreter.copy_program(&program);

    let mut input = empty(); // no input data needed
    let mut output = Vec::new(); // output on the standard

    for _ in 0..200 { // limit
        // each interpreter step return a statement
        // while no `HALT` statement is found, we continue
        match interpreter.step(&mut input, &mut output) {
            Statement(op_codes::HALT, _) => break,
            _ => ()
        }
    }

    output
}

/// Compare string a with "Hello World!".
/// Returns 0..1.0, where 1.0 is a perfect match.
struct ReusmannTextSimilarity;

impl Fitnesses<Individual, Heuristic> for ReusmannTextSimilarity {
    fn fitnesses(&mut self, population: &[Individual]) -> Vec<Heuristic> {
        population.par_iter().map(|individual| {
            let output = program_output(&individual.instructions);

            let mut count = 0.0;
            for (a, b) in TARGET.as_bytes().iter().zip(&output) {
                count += if a == b { 1.0 } else { 0.0 };
                // let diff = if a > b { a - b } else { b - a };
                // count += if diff == 0 { 1.0 } else if diff <= 50 { 0.3 } else { 0.0 };
                // count += 1.0 - diff as f64 / u8::MAX as f64;
            }

            // Optional to evolve an exact-length solution:
            // Give credit for having the correct length;
            // Exact length counts as much as getting one char correct:
            if TARGET.len() == output.len() {
                count += 1.0;
            }

            // normalize to [0.0..1.0 + 1]
            let max = TARGET.len() as f64 + 1.0;
            OrderedFloat(count / max)
        })
        .collect()
    }
}

/// Compare string a with a reference string. Think of them as two abstract vectors.
/// Calculate the root mean square difference between the two.
/// Returns 0..1.0, where 1.0 is a perfect match.
/// Assumes 8-bit chars.
struct ReusmannSimilarityRMS;

impl Fitnesses<Individual, Heuristic> for ReusmannSimilarityRMS {
    fn fitnesses(&mut self, population: &[Individual]) -> Vec<Heuristic> {
        population.par_iter().map(|individual| {
            let output = program_output(&individual.instructions);

            let length = cmp::min(output.len(), TARGET.len());
            if length == 0 || output.len() == 0 || TARGET.len() == 0 {
                return OrderedFloat(0.0);
            }

            // Any missing elements count as maximum error
            let diff = if TARGET.len() > length { TARGET.len() - length }
            else { length - TARGET.len() };
            let mut sum_errors_squared = diff as f64 * 256.0 * 256.0;

            for (a, b) in output.iter().zip(TARGET.as_bytes()) {
                let diff = if a > b { a - b } else { b - a } as f64;
                sum_errors_squared += diff * diff;
            }

            let rms = (sum_errors_squared / TARGET.len() as f64).sqrt();
            let fitness = 1.0 / (1.0 + rms);
            OrderedFloat(fitness)
        })
        .collect()
    }
}

/// Quick and easy who-mates-and-who-dies algorithm using two thresholds.
/// The thresholds are between 0.0 and 1.0, and are percentiles for the
/// current set of fitness measurements after sorting.
/// The diagram below shows two typical threshold values which can be
/// changed with command line options -F and -E.
///
/// ```text
/// +--------------------------------------------+ 1.0 best fitness
/// | Always a parent; lives another generation  |
/// +--------------------------------------------+ ~0.9 <= Elite threshold (-E)
/// |                                            |
/// | May be a parent; lives another generation  |
/// |                                            |
/// +--------------------------------------------+ ~0.5 <= Fitness Deletion Threshold (-F)
/// |                                            |
/// | Does not survive the generation,           |
/// | Replaced by new offspring                  |
/// |                                            |
/// +--------------------------------------------+ 0.0 worst fitness
/// ```
///
/// This function iterates through individuals from the worst fitness to the
/// fitness deletion threshold, replacing each with a new individual. Rather
/// than deleting an object only to have to create a replacement, we just
/// simply overwrite the genome with a new one derived from two parents
/// selected from above the thresholds. A new genome makes a new individual.
struct ReusmannReproduction<R> {
    rng: R,
    select_elite: usize,
    elite_threshold: f64,
    fitness_deletion_threshold: f64,
    mutation_rate: f64,
}

impl<R> Reproduce<Individual, Heuristic> for ReusmannReproduction<R>
where
    R: Rng + Clone + Send + Sync
{
    fn reproduce(&mut self, population: &[Individual], heuristics: &[Heuristic]) -> Vec<Individual> {
        // We sort the individuals by their heuristic.
        let mut population: Vec<_> = population.iter().enumerate().collect();
        population.sort_unstable_by_key(|(i, _)| Reverse(heuristics[*i]));

        let pop_len = population.len() as f64;
        let elite_threshold = (pop_len * (1.0 - self.elite_threshold)) as usize;
        let fitness_threshold = (pop_len * (1.0 - self.fitness_deletion_threshold)) as usize;

        let elite_parents = &population[..elite_threshold];
        let parents = &population[..fitness_threshold];

        // complete the population with some children.
        let remaining = population.len().saturating_sub(self.select_elite);
        let mut new_population: Vec<Individual> = (0..remaining)
            .into_par_iter()
            .map_with(self.rng.clone(), |rng, _| {
                // Select two parents, one from above the elite threshold, the other will be
                // a random selection above the fitness deletion threshold, which might or might
                // not be from the elite section.
                let (_h0, p0) = elite_parents.choose(rng).unwrap();
                let (_h1, p1) = parents.choose(rng).unwrap();
                p0.crossover(rng, p1)
            })
            .collect();

        // This mutates some individual's genomes by randomly changing
        // a single character to a random Iota opcode letter.
        for individual in &mut new_population {
            individual.mutate(&mut self.rng, self.mutation_rate);
        }

        // select a number of parents in the elite population.
        let iter = elite_parents.iter().map(|(_, ind)| (*ind).clone()).take(self.select_elite);
        new_population.extend(iter);

        new_population
    }
}

#[derive(Debug)]
struct Stats {
    p90: f64,
    p75: f64,
    median: f64,
    p25: f64,
    average: f64,
}

impl Stats {
    fn from_heuristics(heuristics: &[Heuristic]) -> Stats {
        let mut heuristics = heuristics.to_vec();
        heuristics.sort_unstable();

        let p90 = heuristics.len() / 100 * 90;
        let p75 = heuristics.len() / 100 * 75;
        let median = heuristics.len() / 2;
        let p25 = heuristics.len() / 100 * 25;

        Stats {
            p90: *heuristics[p90],
            p75: *heuristics[p75],
            median: *heuristics[median],
            p25: *heuristics[p25],
            average: heuristics.iter().fold(0.0, |acc, f| acc + **f) / heuristics.len() as f64,
        }
    }
}

// Here is the perfect program for this task:
// Gp..OOOOOOOOOOOOHTFello World!
//
// Here a program found by this algorithm:
// Fello WormdBz.Zt. h%%O},qiO3d}U,
// $dlko}WnrkdLz(ddd@trO.VrjVkrOr@d
fn main() {
    let seed = rand::random();
    eprintln!("using the seed: 0x{:02x}", seed);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let population_size = 1000;
    let population = (0..population_size).map(|_| Individual::from_rng(&mut rng)).collect();
    let fitnesses = ReusmannTextSimilarity;
    let reproduce = ReusmannReproduction {
        rng: rng.clone(),
        select_elite: 1,
        elite_threshold: 0.9,
        fitness_deletion_threshold: 0.5,
        mutation_rate: 0.01,
    };

    let mut gene = Genetic::new(population, fitnesses, reproduce);
    let mut remaining = 1_000_000_000;
    let mut previous_heuristic = OrderedFloat(0.0);

    while let Some((generation, heuristic)) = gene.next() {
        if remaining % 10_000 == 0 {
            let stats = Stats::from_heuristics(&gene.heuristics());
            eprintln!("generation: {}, best: {:.04}", generation, heuristic);
            eprintln!("{:.04?}", stats);
            if let Some((i, _h)) = gene.heuristics().iter().enumerate().max_by_key(|(_, h)| *h) {
                let individual = &gene.population()[i];
                let output = program_output(&individual.instructions);
                eprintln!("best output: {:?}", output.as_bstr());
                eprintln!();
            }

            if heuristic > previous_heuristic {
                previous_heuristic = heuristic;
                eprint!("\x07");
            }
        }

        remaining -= 1;
        if heuristic == 1.0 || remaining == 0 { break }
    }

    let heuristics = gene.heuristics();
    let population = gene.population();
    if let Some((i, ind)) = population.iter().enumerate().max_by_key(|(i, _)| heuristics[*i]) {
        eprint!("\x07");
        println!("best individual: {:?}", ind);
        println!("best score: {:.04?}", *heuristics[i]);
        println!("best output: {:?}", program_output(&ind.instructions).as_bstr());
    }
}
