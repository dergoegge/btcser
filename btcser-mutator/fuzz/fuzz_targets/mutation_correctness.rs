#![no_main]

use btcser::{object::ObjectParser, parser::DescriptorParser};
use btcser_mutator::{sampler::ChaoSampler, ByteArrayMutator, Mutator, StdSerializedValueMutator};

use libfuzzer_sys::fuzz_target;

struct TestByteArrayMutator;

impl ByteArrayMutator for TestByteArrayMutator {
    fn new(_seed: u64) -> Self {
        Self {}
    }
    fn mutate(&mut self, bytes: &mut Vec<u8>) {
        bytes.fill(0xFF);
    }

    fn mutate_in_place(&mut self, bytes: &mut [u8]) {
        bytes.fill(0xFF);
    }
}

// Test that mutations are always valid serializations of the provided descriptor
fuzz_target!(|data: &[u8]| {
    let descriptor_str = r#"
            TinyStruct {
                bool,                       # Test empty struct edge cases
            }

            SmallStruct {
                u8,                         # Test length field
                slice<u16, '0'>             # Slice referring to previous field
            }

            NestedVec {
                vec<vec<vec<u8>>>,          # Deeply nested vectors
                u8                          # Field after nested structure
            }

            ComplexInner {
                bool,                       # Boolean field
                vec<SmallStruct>,           # Vector of structs with slices
                slice<u256, '0'>,           # Large integers in slice
                vec<u8>,                    # Byte vector
                bytes<32>                   # Fixed-size bytes
            }

            Inner1 {
                u8,                         # Length field
                slice<ComplexInner, '0'>,   # Slice of complex structs
                vec<U16>,                   # Big-endian integers
                bytes<4>(0xdeadbeef),       # Fixed bytes
                TinyStruct                  # Empty struct edge case
            }

            Inner2 {
                bool,                       # Length field
                slice<U64, '0'>,            # Big-endian slice
                vec<Inner1>,                # Nested structs
                NestedVec                   # Deeply nested structure
            }

            Test {
                vec<u8>,                    # Byte vector
                Inner1,                     # Nested struct
                vec<Inner2>,                # Vector of complex structs
                slice<Inner2, '0'>,         # Slice of complex structs
                U256,                       # Big-endian large integer
                vec<vec<u16>>,              # Nested vectors
                SmallStruct,                # Struct with internal references
                bool,                       # Boolean after complex fields
                slice<TinyStruct, '7'>,     # Slice referring to later field
                u8                          # Final length field
            }
        "#;

    let mut parser = DescriptorParser::new();
    parser.parse_file(descriptor_str).unwrap();
    let descriptor = parser.get_descriptor("Test").unwrap().clone();

    let mutator = Mutator::new(descriptor.clone(), &parser);

    // Attempt to parse and perform a mutation
    if let Ok(mutated_bytes) =
        mutator.mutate::<ChaoSampler<_>, StdSerializedValueMutator<TestByteArrayMutator>>(&data, 0)
    {
        // If mutating succeeded, assert that reparsing the mutation result succeeds.
        let obj_parser = ObjectParser::new(descriptor, &parser);
        let _ = obj_parser
            .parse(&mutated_bytes)
            .expect("Mutated value should be parseable");

        // Test crossover with the original and mutated data
        if let Ok(crossover_bytes) = mutator.cross_over::<ChaoSampler<_>, ChaoSampler<_>, StdSerializedValueMutator<TestByteArrayMutator>>(&data, &mutated_bytes, 0) {
            // Verify that the crossover result is also parseable
            let _ = obj_parser
                .parse(&crossover_bytes)
                .expect("Crossover value should be parseable");
        }
    }
});
