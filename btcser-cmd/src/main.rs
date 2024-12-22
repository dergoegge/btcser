use std::fs;
use std::io::{self, Read};

use btcser::{
    object::{ObjectParser, SerializedValue},
    parser::{DescriptorParser, IntType},
};
use btcser_mutator::{sampler::ChaoSampler, ByteArrayMutator, Mutator, StdSerializedValueMutator};

use clap::{Parser, Subcommand};
use rand::Rng;

#[derive(Parser)]
#[command(
    author = "Niklas Gögge <n.goeggi@gmail.com>",
    version = "1.0",
    about = "btcser command line tool",
    long_about = "A command line tool for parsing and mutating serialized Bitcoin objects."
)]
struct Cmd {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Parse descriptors from stdin
    Parse,
    /// Parse a serialized object from stdin
    ParseObj {
        /// Path to the descriptor file
        #[arg(help = "The path to the descriptor file containing the object definitions.")]
        descriptor_file: String,

        /// Name of the descriptor to parse
        #[arg(help = "The name of the descriptor to parse from the file.")]
        descriptor_name: String,
    },
    /// Mutate a serialized object from stdin
    Mutate {
        /// Path to the descriptor file
        #[arg(help = "The path to the descriptor file containing the object definitions.")]
        descriptor_file: String,

        /// Name of the descriptor to mutate
        #[arg(help = "The name of the descriptor to mutate from the file.")]
        descriptor_name: String,
    },
}

// Define SimpleMutator struct
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
    // Implement mutation strategies
    fn bit_flip(&self, bytes: &mut [u8]) {
        let num_mutations = self.rng.clone().gen_range(1..=bytes.len());
        for _ in 0..num_mutations {
            let idx = self.rng.clone().gen_range(0..bytes.len());
            let bit = self.rng.clone().gen_range(0..8);
            bytes[idx] ^= 1 << bit;
        }
    }

    fn byte_flip(&self, bytes: &mut [u8]) {
        let num_mutations = self.rng.clone().gen_range(1..=bytes.len());
        for _ in 0..num_mutations {
            let idx = self.rng.clone().gen_range(0..bytes.len());
            bytes[idx] ^= 0xFF;
        }
    }

    fn random_byte(&self, bytes: &mut [u8]) {
        let num_mutations = self.rng.clone().gen_range(1..=bytes.len());
        for _ in 0..num_mutations {
            let idx = self.rng.clone().gen_range(0..bytes.len());
            bytes[idx] = self.rng.clone().gen::<u8>();
        }
    }

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

fn main() {
    let cli = Cmd::parse();

    match cli.command {
        Some(Commands::Parse) => {
            let mut parser = DescriptorParser::new();
            let mut input = String::new();

            std::io::stdin()
                .read_to_string(&mut input)
                .expect("Failed to read from stdin");

            match parser.parse_file(&input) {
                Ok(_) => println!("{:#?}", parser.descriptors),
                Err(e) => println!("Error parsing content: {}", e),
            }
        }
        Some(Commands::ParseObj {
            descriptor_file,
            descriptor_name,
        }) => {
            let descriptor_content =
                fs::read_to_string(&descriptor_file).expect("Failed to read descriptor file");

            let mut parser = DescriptorParser::new();
            parser
                .parse_file(&descriptor_content)
                .expect("Failed to parse descriptor file");

            // Get the specified descriptor
            let descriptor = parser
                .descriptors
                .get(&descriptor_name)
                .expect("Descriptor not found in file")
                .clone();

            // Create object parser
            let object_parser = ObjectParser::new(descriptor, &parser);

            // Read hex-encoded input from stdin
            let mut input = String::new();
            io::stdin()
                .read_to_string(&mut input)
                .expect("Failed to read from stdin");

            // Trim whitespace and decode hex
            let input = input.trim();
            let bytes = hex::decode(input).expect("Failed to decode hex input");

            // Parse the input
            let values = object_parser.parse(&bytes).expect("Failed to parse input");

            // Pretty print the result
            for (i, value) in values.iter().enumerate() {
                print_serialized_value(value, 0, &i.to_string());
            }
        }
        Some(Commands::Mutate {
            descriptor_file,
            descriptor_name,
        }) => {
            // Read descriptor file and parse it
            let descriptor_content =
                fs::read_to_string(&descriptor_file).expect("Failed to read descriptor file");

            let mut parser = DescriptorParser::new();
            parser
                .parse_file(&descriptor_content)
                .expect("Failed to parse descriptor file");

            // Get the specified descriptor
            let descriptor = parser
                .descriptors
                .get(&descriptor_name)
                .expect("Descriptor not found in file")
                .clone();

            // Parse seed if provided, otherwise generate random seed
            let seed = rand::random::<u64>();
            eprintln!("Using random seed: {}", seed);

            // Create mutator
            let mutator = Mutator::new(descriptor.clone(), &parser);

            // Read hex-encoded input from stdin
            let mut input = String::new();
            io::stdin()
                .read_to_string(&mut input)
                .expect("Failed to read from stdin");

            // Trim whitespace and decode hex
            let input = input.trim();
            let bytes = hex::decode(input).expect("Failed to decode hex input");

            // Perform mutation
            let mutated = mutator
                .mutate::<ChaoSampler<_>, StdSerializedValueMutator<SimpleMutator>>(&bytes, seed)
                .expect("Failed to mutate input");

            // Validate that the mutated bytes can be parsed
            let obj_parser = ObjectParser::new(descriptor, &parser);
            if let Err(e) = obj_parser.parse(&mutated) {
                panic!(
                    "Mutation produced invalid bytes that failed to parse: {}",
                    e
                );
            }

            // Output the mutated bytes as hex
            println!("{}", hex::encode(mutated));
        }
        None => {
            println!("No valid subcommand was used");
        }
    }
}

fn print_serialized_value(value: &SerializedValue, indent: usize, path: &str) {
    let indent_str = "  ".repeat(indent);

    match &value.field_type {
        btcser::parser::FieldType::Struct(name) => {
            println!("{}[{}] {}", indent_str, path, name);
        }
        btcser::parser::FieldType::Vec(inner_type) => {
            if let btcser::parser::FieldType::Int(IntType::U8) = **inner_type {
                println!(
                    "{}[{}] Vec<u8>: 0x{}",
                    indent_str,
                    path,
                    hex::encode(&value.bytes)
                );
                return;
            } else {
                println!(
                    "{}[{}] Vec<{:?}> (length: {})",
                    indent_str,
                    path,
                    inner_type,
                    value.nested_values.len()
                );
            }
        }
        btcser::parser::FieldType::Slice(inner_type, _) => {
            if let btcser::parser::FieldType::Int(IntType::U8) = **inner_type {
                println!(
                    "{}[{}] Slice<u8>: 0x{}",
                    indent_str,
                    path,
                    hex::encode(&value.bytes)
                );
                return;
            } else {
                println!(
                    "{}[{}] Slice<{:?}> (length: {})",
                    indent_str,
                    path,
                    inner_type,
                    value.nested_values.len()
                );
            }
        }
        _ => {
            let value_str = match value.field_type {
                btcser::parser::FieldType::Int(_) => {
                    if value.bytes.len() <= 8 {
                        let mut bytes = [0u8; 8];
                        bytes[..value.bytes.len()].copy_from_slice(value.bytes);
                        format!("{}", u64::from_le_bytes(bytes))
                    } else {
                        format!("0x{}", hex::encode(value.bytes))
                    }
                }
                _ => format!("0x{}", hex::encode(value.bytes)),
            };
            println!(
                "{}[{}] {:?}: {}",
                indent_str, path, value.field_type, value_str
            );
        }
    }

    // Print nested values with updated path
    for (i, nested) in value.nested_values.iter().enumerate() {
        let new_path = if path.is_empty() {
            i.to_string()
        } else {
            format!("{}.{}", path, i)
        };
        print_serialized_value(nested, indent + 1, &new_path);
    }
}
