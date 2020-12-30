use std::cmp::{self, Reverse};
use std::io::empty;

use bstr::ByteSlice as _;
use rand::distributions::Alphanumeric;
use rand::{Rng, SeedableRng};
use rand::seq::SliceRandom;
use rayon::prelude::*;
use reustmann::instruction::op_codes;
use reustmann::instruction::{Instruction, OpCode};
use reustmann::Statement;
use reustmann::{Program, Interpreter};

use genegene::{Genetic, Fitnesses, Reproduce};

const PROGRAM_LENGTH: usize = 150;
const HELLO_WORLD: &str = "Hello World!";
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
        rng.sample(Alphanumeric) as u8
    }
}

type Heuristic = u32;

#[derive(Clone)]
struct Individual {
    instructions: Vec<OpCode>,
}

impl Individual {
    pub fn from_rng<R: Rng + ?Sized>(rng: &mut R) -> Individual {
        let size = rng.gen_range(10..=50);
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

        let instructions = match rng.gen_range(0..3) {
            0 => // AaaaabbbbbaaaaaAaaaa
                parent0[0..p0].iter()
                .chain(&parent1[p0..p0 + p1 - p0])
                .chain(&parent0[p1..])
                .chain(&parent0[0..p0])
                .take(PROGRAM_LENGTH)
                .cloned()
                .collect(),
            1 => // Aaaaabbbbbaaaaa
                parent0[0..p0].iter()
                .chain(&parent1[p0..])
                .chain(&parent0[p1..])
                .take(PROGRAM_LENGTH)
                .cloned()
                .collect(),
            _ => // aaaaabbbbbAaaaa
                parent0[p0..p0 + p1 - p0].iter()
                .chain(&parent1[p1..])
                .chain(&parent0[p0..])
                .take(PROGRAM_LENGTH)
                .cloned()
                .collect(),
        };

        Individual { instructions }
    }

    fn mutate<R: Rng + ?Sized>(&mut self, rng: &mut R, mutation_rate: f64) {
        let num_chars_to_mutate = (
            mutation_rate * self.instructions.len() as f64
            + 0.5 + rng.gen::<f64>()
        ) as usize;

        for _ in 0..num_chars_to_mutate {
            if let Some(chosen) = self.instructions.choose_mut(rng) {
                *chosen = random_instruction(rng);
            }
        }
    }
}

fn program_output(instructions: &[u8]) -> Vec<u8> {
    let program = Program::from_iter(instructions.iter().cloned());

    let arch_length = PROGRAM_LENGTH; // memory length
    let arch_width = 8; // word size
    let mut interpreter = Interpreter::new(arch_length, arch_width).unwrap();
    interpreter.copy_program(&program);

    let mut input = empty(); // no input data needed
    let mut output = Vec::new(); // output on the standard

    for _ in 0..5_000 { // limit
        // each interpreter step return a statement
        // while no `HALT` statement is found, we continue
        match interpreter.step(&mut input, &mut output) {
            Statement(op_codes::HALT, _) => break,
            _ => ()
        }
    }

    output
}

struct ReusmannFitnesses;

impl Fitnesses<Individual, Heuristic> for ReusmannFitnesses {
    fn fitnesses(&mut self, population: &[Individual]) -> Vec<Heuristic> {
        population.par_iter().map(|individual| {
            let output = program_output(&individual.instructions);

            let mut count = 0.0;
            for (a, b) in HELLO_WORLD.as_bytes().iter().zip(&output) {
                if a == b { count += 1.0 }
            }

            // Optional to evolve an exact-length solution:
            // Give credit for having the correct length;
            // Exact length counts as much as getting one char correct:
            if HELLO_WORLD.len() == output.len() { count += 1.0 }

            // normalize to [0.0..u16::MAX + 1]
            let max = HELLO_WORLD.len() as f64 + 1.0;
            let fitness = (count / max) * u16::MAX as f64;
            fitness as u32
        })
        .collect()
    }
}

struct ReusmannReproduction<R> {
    rng: R,
    elite_threshold: f64,
    fitness_deletion_threshold: f64,
    mutation_rate: f64,
}

impl<R> Reproduce<Individual, Heuristic> for ReusmannReproduction<R>
where
    R: Rng + Clone + Send + Sync
{
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
    fn reproduce(&mut self, population: &[Individual], heuristics: &[Heuristic]) -> Vec<Individual> {
        // We sort the individuals by their heuristic.
        let mut population: Vec<_> = population.iter().enumerate().map(|(i, ind)| (ind, heuristics[i])).collect();
        population.sort_unstable_by_key(|(_, h)| Reverse(*h));

        let elite_threshold = (population.len() as f64 * (1.0 - self.elite_threshold)) as usize;
        let fitness_threshold = (population.len() as f64 * (1.0 - self.fitness_deletion_threshold)) as usize;

        let elite_parents = &population[..elite_threshold];
        let parents = &population[..fitness_threshold];

        (0..population.len()).into_par_iter().map_with(self.rng.clone(), |rng, _| {
            // Select two parents, one from above the elite threshold, the other will be
            // a random selection above the fitness deletion threshold, which might or might
            // not be from the elite section.
            let (p0, _h0) = elite_parents.choose(rng).unwrap();
            let (p1, _h1) = parents.choose(rng).unwrap();

            let mut child = p0.crossover(rng, p1);
            if rng.gen::<f64>() <= self.mutation_rate * child.instructions.len() as f64 {
                child.mutate(rng, self.mutation_rate);
            }
            child
        })
        .collect()
    }
}

// Here is the perfect program for this task:
// Gp..OOOOOOOOOOOOHTFello World!
fn main() {
    let seed = rand::random();
    eprintln!("using the seed: 0x{:02x}", seed);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let population_size = 10_000;
    let population = (0..population_size).map(|_| Individual::from_rng(&mut rng)).collect();
    let fitnesses = ReusmannFitnesses;
    let reproduce = ReusmannReproduction {
        rng: rng.clone(),
        elite_threshold: 0.9,
        fitness_deletion_threshold: 0.5,
        mutation_rate: 0.05,
    };
    let mut gene = Genetic::new(population, fitnesses, reproduce);

    let mut remaining = 10_000;
    while let Some((generation, best_heuristic)) = gene.next() {
        println!("generation: {}, best heuristic: {}", generation, best_heuristic);
        if let Some((i, _h)) = gene.heuristics().iter().enumerate().max_by_key(|(_, h)| *h) {
            let individual = &gene.population()[i];
            let output = program_output(&individual.instructions);
            println!("best individual:");
            println!("{:?}", individual.instructions.as_bstr());
            println!("output: {:?}", output.as_bstr());
            println!();
        }

        remaining -= 1;
        if remaining == 0 { break }
    }
}
