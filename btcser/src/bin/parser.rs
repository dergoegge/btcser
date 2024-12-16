use btcser::parser::DescriptorParser;
use std::io::Read;

fn main() {
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
