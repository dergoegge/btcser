use std::fs;
use std::io::{self, Read};

use btcser::{
    object::{ObjectParser, SerializedValue},
    parser::{DescriptorParser, IntType},
};
use btcser_mutator::{sampler::ChaoSampler, ByteArrayMutator, Mutator, StdSerializedValueMutator};

use clap::{Parser, Subcommand};
use libafl::mutators::MutatorsTuple;
use libafl::{
    corpus::NopCorpus,
    inputs::{BytesInput, HasMutatorBytes},
    mutators::{havoc_mutations_no_crossover, HavocMutationsNoCrossoverType},
    state::{HasRand, StdState},
};
use libafl_bolts::{
    rands::{Rand, StdRand},
    HasLen,
};

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
    state: StdState<BytesInput, NopCorpus<BytesInput>, StdRand, NopCorpus<BytesInput>>,
    mutator: HavocMutationsNoCrossoverType,
}

impl ByteArrayMutator for SimpleMutator {
    fn new(seed: u64) -> Self {
        let state = StdState::new(
            StdRand::with_seed(seed),
            NopCorpus::new(),
            NopCorpus::new(),
            &mut (),
            &mut (),
        )
        .unwrap();

        let mutator = havoc_mutations_no_crossover();
        Self { state, mutator }
    }

    fn mutate(&mut self, bytes: &mut Vec<u8>) {
        let mut input = BytesInput::from(bytes.clone());
        let idx = self.state.rand_mut().next() % self.mutator.len() as u64;
        let _ = self
            .mutator
            .get_and_mutate(idx.into(), &mut self.state, &mut input);
        bytes.clear();
        bytes.extend(input.bytes());
    }

    fn mutate_in_place(&mut self, bytes: &mut [u8]) {
        let mut input = BytesInput::from(bytes.to_vec());
        let idx = self.state.rand_mut().next() % self.mutator.len() as u64;
        let _ = self
            .mutator
            .get_and_mutate(idx.into(), &mut self.state, &mut input);
        bytes.copy_from_slice(&input.bytes()[..bytes.len()]);
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
