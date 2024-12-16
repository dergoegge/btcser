use btcser::{object::ObjectParser, parser::DescriptorParser};
use btcser_mutator::{sampler::ChaoSampler, ByteArrayMutator, Mutator, StdSerializedValueMutator};
use rand::Rng;
use std::env;
use std::fs;
use std::io::{self, Read};

struct SimpleMutator {
    rng: rand::rngs::StdRng,
}

impl ByteArrayMutator for SimpleMutator {
    fn new(seed: u64) -> Self {
        Self {
            rng: rand::SeedableRng::seed_from_u64(seed),
        }
    }

    fn mutate(&self, bytes: &mut Vec<u8>) {
        self.mutate_in_place(bytes);
    }

    fn mutate_in_place(&self, bytes: &mut [u8]) {
        if bytes.is_empty() {
            return;
        }

        // Choose a random mutation strategy
        match self.rng.clone().gen_range(0..5) {
            0 => self.bit_flip(bytes),
            1 => self.byte_flip(bytes),
            2 => self.random_byte(bytes),
            3 => self.add_or_subtract(bytes),
            4 => self.swap_bytes(bytes),
            _ => unreachable!(),
        }
    }
}

impl SimpleMutator {
    // Flip random bits in random bytes
    fn bit_flip(&self, bytes: &mut [u8]) {
        let num_mutations = self.rng.clone().gen_range(1..=bytes.len());
        for _ in 0..num_mutations {
            let idx = self.rng.clone().gen_range(0..bytes.len());
            let bit = self.rng.clone().gen_range(0..8);
            bytes[idx] ^= 1 << bit;
        }
    }

    // Flip entire bytes
    fn byte_flip(&self, bytes: &mut [u8]) {
        let num_mutations = self.rng.clone().gen_range(1..=bytes.len());
        for _ in 0..num_mutations {
            let idx = self.rng.clone().gen_range(0..bytes.len());
            bytes[idx] ^= 0xFF;
        }
    }

    // Replace bytes with random values
    fn random_byte(&self, bytes: &mut [u8]) {
        let num_mutations = self.rng.clone().gen_range(1..=bytes.len());
        for _ in 0..num_mutations {
            let idx = self.rng.clone().gen_range(0..bytes.len());
            bytes[idx] = self.rng.clone().gen::<u8>();
        }
    }

    // Add or subtract small values from bytes
    fn add_or_subtract(&self, bytes: &mut [u8]) {
        let num_mutations = self.rng.clone().gen_range(1..=bytes.len());
        for _ in 0..num_mutations {
            let idx = self.rng.clone().gen_range(0..bytes.len());
            let delta = self.rng.clone().gen_range(1..=4);
            if self.rng.clone().gen_bool(0.5) {
                bytes[idx] = bytes[idx].wrapping_add(delta);
            } else {
                bytes[idx] = bytes[idx].wrapping_sub(delta);
            }
        }
    }

    // Swap pairs of bytes
    fn swap_bytes(&self, bytes: &mut [u8]) {
        if bytes.len() < 2 {
            return;
        }
        let num_swaps = self.rng.clone().gen_range(1..=bytes.len() / 2);
        for _ in 0..num_swaps {
            let idx1 = self.rng.clone().gen_range(0..bytes.len());
            let idx2 = self.rng.clone().gen_range(0..bytes.len());
            bytes.swap(idx1, idx2);
        }
    }
}

fn main() -> Result<(), String> {
    // Get descriptor file path, name, and optional seed from command line args
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 && args.len() != 4 {
        return Err("Usage: mutator <descriptor_file> <descriptor_name> [seed]".to_string());
    }

    // Read and parse the descriptor file
    let descriptor_content = fs::read_to_string(&args[1])
        .map_err(|e| format!("Failed to read descriptor file: {}", e))?;

    let mut parser = DescriptorParser::new();
    parser.parse_file(&descriptor_content)?;

    // Get the specified descriptor
    let descriptor_name = &args[2];
    let descriptor = parser
        .descriptors
        .get(descriptor_name)
        .ok_or_else(|| format!("Descriptor '{}' not found in file", descriptor_name))?
        .clone();

    // Parse seed if provided, otherwise generate random seed
    let seed = if args.len() == 4 {
        args[3]
            .parse::<u64>()
            .map_err(|e| format!("Invalid seed value: {}", e))?
    } else {
        let random_seed = rand::random::<u64>();
        eprintln!("Using random seed: {}", random_seed);
        random_seed
    };

    // Create mutator
    let mutator = Mutator::new(descriptor.clone(), &parser);

    // Read hex-encoded input from stdin
    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .map_err(|e| format!("Failed to read from stdin: {}", e))?;

    // Trim whitespace and decode hex
    let input = input.trim();
    let bytes = hex::decode(input).map_err(|e| format!("Failed to decode hex input: {}", e))?;

    // Perform mutation
    let mutated =
        mutator.mutate::<ChaoSampler<_>, StdSerializedValueMutator<SimpleMutator>>(&bytes, seed)?;

    // Validate that the mutated bytes can be parsed
    let obj_parser = ObjectParser::new(descriptor, &parser);
    if let Err(e) = obj_parser.parse(&mutated) {
        return Err(format!(
            "Mutation produced invalid bytes that failed to parse: {}",
            e
        ));
    }

    // Output the mutated bytes as hex
    println!("{}", hex::encode(mutated));

    Ok(())
}
