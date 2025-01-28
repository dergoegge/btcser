use std::collections::HashSet;

use btcser::{
    object::{encode_compact_size, Object, ObjectParser},
    parser::{Descriptor, DescriptorParser, FieldType},
};
use btcser_mutator::{sampler::ChaoSampler, ByteArrayMutator, StdObjectMutator};
use custom_mutator::{export_mutator, CustomMutator};

use libafl::{
    corpus::NopCorpus,
    inputs::{BytesInput, HasMutatorBytes},
    mutators::{
        havoc_mutations, scheduled::StdScheduledMutator, tokens_mutations, Mutator, Tokens,
    },
    state::StdState,
    HasMetadata,
};
use libafl_bolts::{rands::StdRand, tuples::Merge};

// Dummy needed due to btcser-mutator
#[allow(non_snake_case)]
#[no_mangle]
fn LLVMFuzzerMutate(_data: *mut u8, _size: usize, _max_size: usize) -> usize {
    return 0;
}

struct LibAflMutator {
    state: StdState<BytesInput, NopCorpus<BytesInput>, StdRand, NopCorpus<BytesInput>>,
}

impl ByteArrayMutator for LibAflMutator {
    fn new(seed: u64) -> Self {
        let mut state = StdState::new(
            StdRand::with_seed(seed),
            NopCorpus::new(),
            NopCorpus::new(),
            &mut (),
            &mut (),
        )
        .unwrap();

        let mut tokens = Tokens::new();
        if let Ok(tokens_file) = std::env::var("BTCSER_AFLPP_TOKENS") {
            tokens.add_from_file(&tokens_file).unwrap();
        }
        state.add_metadata(tokens);

        Self { state }
    }

    fn mutate(&mut self, bytes: &mut Vec<u8>) {
        let mut input = BytesInput::from(bytes.clone());

        let mut mutator = StdScheduledMutator::new(havoc_mutations().merge(tokens_mutations()));
        let _ = mutator.mutate(&mut self.state, &mut input);

        bytes.clear();
        bytes.extend(input.bytes());
    }

    fn mutate_in_place(&mut self, bytes: &mut [u8]) {
        let mut input = BytesInput::from(bytes.to_vec());

        let mut mutator = StdScheduledMutator::new(havoc_mutations().merge(tokens_mutations()));
        let _ = mutator.mutate(&mut self.state, &mut input);

        let input_bytes = input.bytes();
        let len_to_copy = std::cmp::min(bytes.len(), input_bytes.len());
        bytes[..len_to_copy].copy_from_slice(&input_bytes[..len_to_copy]);
    }
}

struct BtcserMutator {
    parser: DescriptorParser,
    descriptor: Descriptor,
    buffer: Vec<u8>,
    post_process_buffer: Vec<u8>,
    seed: u64,
    debug: bool,
    as_bytes: HashSet<String>,
}

impl BtcserMutator {
    fn post_process_object_step(
        &mut self,
        objects: &[Object],
        current_position: &mut usize,
        original_data: &[u8],
    ) -> Result<(), String> {
        for object in objects {
            match object.field_type() {
                FieldType::Struct(name) if self.as_bytes.contains(name.as_str()) => {
                    let object_start =
                        object.bytes().as_ptr() as usize - original_data.as_ptr() as usize;
                    // Copy upto the object start
                    self.post_process_buffer
                        .extend_from_slice(&original_data[*current_position..object_start]);
                    // Encode the object as a vec<u8>
                    self.post_process_buffer
                        .extend_from_slice(&encode_compact_size(object.bytes().len() as u64));
                    self.post_process_buffer.extend(object.bytes());
                    // Advance current_position
                    *current_position = object_start + object.bytes().len();
                }
                FieldType::Struct(_) | FieldType::Vec(_) | FieldType::Slice(_, _) => {
                    // Recursively process the nested objects
                    self.post_process_object_step(
                        object.nested_values(),
                        current_position,
                        original_data,
                    )?;
                }
                _ => {}
            }
        }

        // Copy the rest of the original data
        self.post_process_buffer
            .extend_from_slice(&original_data[*current_position..]);

        Ok(())
    }

    fn post_process_object(&mut self, bytes: &[u8]) -> Result<(), String> {
        let parser = ObjectParser::new(self.descriptor.clone(), &self.parser);

        let object = parser.parse(bytes)?;

        self.post_process_buffer.clear();
        let mut current_position = 0;

        self.post_process_object_step(&object, &mut current_position, bytes)?;

        Ok(())
    }
}

impl CustomMutator for BtcserMutator {
    type Error = String;

    fn init(seed: u32) -> Result<Self, Self::Error>
    where
        Self: Sized,
    {
        let debug = std::env::var("BTCSER_AFLPP_DEBUG").is_ok();

        let (descriptor_str, name_str) = {
            match (
                std::env::var("BTCSER_DESCRIPTOR_FILE"),
                std::env::var("BTCSER_DESCRIPTOR"),
            ) {
                (Ok(d), Ok(n)) => (d, n),
                _ => return Err("Descriptor file or descriptor not set (use BTCSER_DESCRIPTOR_FILE and BTCSER_DESCRIPTOR)".to_string()),
            }
        };

        let mut parser = btcser::parser::DescriptorParser::new();
        let descriptor = {
            match parser.parse_file(&descriptor_str) {
                Ok(_) => match parser.get_descriptor(&name_str) {
                    Some(d) => d.clone(),
                    None => {
                        return Err(format!("Descriptor '{}' not found", name_str));
                    }
                },
                Err(err) => {
                    return Err(format!("Descriptor didn't parse: {}", err));
                }
            }
        };

        let mut as_bytes = HashSet::new();

        if let Ok(types) = std::env::var("BTCSER_AFLPP_AS_BYTES") {
            for type_name in types.split(',') {
                let type_name = type_name.trim();
                as_bytes.insert(type_name.to_string());

                if parser.get_descriptor(type_name).is_none() {
                    return Err(format!("Descriptor '{}' not found", type_name));
                }
            }
        }

        Ok(BtcserMutator {
            parser,
            descriptor,
            buffer: Vec::with_capacity(1024 * 1024),
            post_process_buffer: Vec::with_capacity(1024 * 1024),
            seed: seed as u64,
            debug,
            as_bytes,
        })
    }

    fn fuzz<'b, 's: 'b>(
        &'s mut self,
        buffer: &'b mut [u8],
        add_buff: Option<&[u8]>,
        _max_size: usize,
    ) -> Result<Option<&'b [u8]>, Self::Error> {
        let mutator = btcser_mutator::Mutator::new(self.descriptor.clone(), &self.parser);

        let result = match add_buff {
            Some(add_buff) => {
                match mutator
                    .cross_over::<ChaoSampler<_>, ChaoSampler<_>, StdObjectMutator<LibAflMutator>>(
                        buffer, add_buff, self.seed,
                    ) {
                    Ok(mutated) => Ok(mutated),
                    Err(_) => {
                        // Fallback to regular mutate if cross_over fails
                        mutator.mutate::<ChaoSampler<_>, StdObjectMutator<LibAflMutator>>(
                            buffer, self.seed,
                        )
                    }
                }
            }
            None => {
                mutator.mutate::<ChaoSampler<_>, StdObjectMutator<LibAflMutator>>(buffer, self.seed)
            }
        };

        self.seed += 1;

        match result {
            Ok(mutated) => {
                self.buffer.clear();
                self.buffer.extend_from_slice(&mutated);
                Ok(Some(&self.buffer))
            }
            Err(err) => {
                if self.debug {
                    eprintln!("Error mutating: {}", err);
                }
                Ok(None)
            }
        }
    }

    fn post_process<'b, 's: 'b>(
        &'s mut self,
        buffer: &'b mut [u8],
    ) -> Result<Option<&'b [u8]>, Self::Error> {
        if self.as_bytes.is_empty() {
            return Ok(Some(buffer));
        }

        self.post_process_object(buffer)?;
        Ok(Some(&self.post_process_buffer))
    }

    fn fuzz_count(&mut self, _buffer: &[u8]) -> Result<u32, Self::Error> {
        Ok(8192)
    }
}

export_mutator!(BtcserMutator);
