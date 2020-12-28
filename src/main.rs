use std::io::empty;
use std::{cmp, iter};

use bstr::ByteSlice as _;
use rand::distributions::Alphanumeric;
use rand::Rng;
use rand::seq::SliceRandom;
use reustmann::instruction::op_codes;
use reustmann::instruction::{Instruction, OpCode};
use reustmann::Statement;
use reustmann::{Program, Interpreter};

use genegene::{Genetic, RouletteWheel};
use genegene::{Fitness, Operator};

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

struct ReusmannFitness;

impl Fitness<Individual, Heuristic> for ReusmannFitness {
    fn fitness(&self, individual: &Individual) -> Heuristic {
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
    }
}

struct ReusmannCrossoverMutation<'r, R: 'r + ?Sized> {
    rng: &'r mut R,
    mutation_rate: f64,
}

impl<R: Rng + ?Sized> Operator<Individual> for ReusmannCrossoverMutation<'_, R> {
    fn complete(&mut self, population: &[Individual], limit: usize) -> Vec<Individual> {
        let iter = iter::from_fn(|| {
            let mut iter = population.choose_multiple(self.rng, 2);
            match iter.next().zip(iter.next()) {
                Some((mother, father)) => {
                    let mut child = mother.crossover(self.rng, father);
                    if self.rng.gen::<f64>() <= self.mutation_rate * child.instructions.len() as f64 {
                        child.mutate(self.rng, self.mutation_rate);
                    }
                    Some(child)
                },
                None => None
            }
        });

        // TODO complete with randoms
        iter.take(limit).collect()
    }
}

// Gp..OOOOOOOOOOOOHTFello World!
fn main() {
    let mut rng = rand::thread_rng();

    let population_size = 10_000;
    let population = (0..population_size).map(|_| Individual::from_rng(&mut rng)).collect();
    let fitness = ReusmannFitness;
    let selection = RouletteWheel { rng: &mut rng.clone(), take_limit_ratio: 0.05 };
    let operator = ReusmannCrossoverMutation { rng: &mut rng, mutation_rate: 0.02 };
    let mut gene = Genetic::new(population, fitness, selection, operator);

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
