use btcser::object::{ObjectParser, SerializedValue};
use btcser::parser::DescriptorParser;
use std::env;
use std::fs;
use std::io::{self, Read};

fn print_serialized_value(value: &SerializedValue, indent: usize, path: &str) {
    let indent_str = "  ".repeat(indent);

    match &value.field_type {
        btcser::parser::FieldType::Struct(name) => {
            println!("{}[{}] {}", indent_str, path, name);
        }
        btcser::parser::FieldType::Vec(inner_type) => {
            println!(
                "{}[{}] Vec<{:?}> (length: {})",
                indent_str,
                path,
                inner_type,
                value.nested_values.len()
            );
        }
        btcser::parser::FieldType::Slice(inner_type, _) => {
            println!(
                "{}[{}] Slice<{:?}> (length: {})",
                indent_str,
                path,
                inner_type,
                value.nested_values.len()
            );
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

fn main() -> Result<(), String> {
    // Get descriptor file path and name from command line args
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        return Err("Usage: obj-parser <descriptor_file> <descriptor_name>".to_string());
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

    // Create mutator
    let object_parser = ObjectParser::new(descriptor, &parser);

    // Read hex-encoded input from stdin
    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .map_err(|e| format!("Failed to read from stdin: {}", e))?;

    // Trim whitespace and decode hex
    let input = input.trim();
    let bytes = hex::decode(input).map_err(|e| format!("Failed to decode hex input: {}", e))?;

    // Parse the input
    let values = object_parser.parse(&bytes)?;

    // Pretty print the result
    for (i, value) in values.iter().enumerate() {
        print_serialized_value(value, 0, &i.to_string());
    }

    Ok(())
}
