use btcser_mutator::{sampler::ChaoSampler, ByteArrayMutator, StdObjectMutator};
use custom_mutator::{export_mutator, CustomMutator};

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

// Dummy needed due to btcser-mutator
#[allow(non_snake_case)]
#[no_mangle]
fn LLVMFuzzerMutate(_data: *mut u8, _size: usize, _max_size: usize) -> usize {
    return 0;
}

struct LibAflMutator {
    state: StdState<BytesInput, NopCorpus<BytesInput>, StdRand, NopCorpus<BytesInput>>,
    mutator: HavocMutationsNoCrossoverType,
}

impl ByteArrayMutator for LibAflMutator {
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
        let input_bytes = input.bytes();
        let len_to_copy = std::cmp::min(bytes.len(), input_bytes.len());
        bytes[..len_to_copy].copy_from_slice(&input_bytes[..len_to_copy]);
    }
}

struct BtcserMutator<'a> {
    mutator: btcser_mutator::Mutator<'a>,
    buffer: Vec<u8>,
    seed: u64,
    debug: bool,
}

impl<'a> CustomMutator for BtcserMutator<'a> {
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

        let mut parser = Box::new(btcser::parser::DescriptorParser::new());
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

        let mutator = btcser_mutator::Mutator::new(descriptor, Box::leak(parser));
        Ok(BtcserMutator {
            mutator,
            buffer: Vec::with_capacity(1024 * 1024),
            seed: seed as u64,
            debug,
        })
    }

    fn fuzz<'b, 's: 'b>(
        &'s mut self,
        buffer: &'b mut [u8],
        add_buff: Option<&[u8]>,
        _max_size: usize,
    ) -> Result<Option<&'b [u8]>, Self::Error> {
        let result = match add_buff {
            Some(add_buff) => {
                match self
                    .mutator
                    .cross_over::<ChaoSampler<_>, ChaoSampler<_>, StdObjectMutator<LibAflMutator>>(
                        buffer, add_buff, self.seed,
                    ) {
                    Ok(mutated) => Ok(mutated),
                    Err(_) => {
                        // Fallback to regular mutate if cross_over fails
                        self.mutator
                            .mutate::<ChaoSampler<_>, StdObjectMutator<LibAflMutator>>(
                                buffer, self.seed,
                            )
                    }
                }
            }
            None => self
                .mutator
                .mutate::<ChaoSampler<_>, StdObjectMutator<LibAflMutator>>(buffer, self.seed),
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

    fn fuzz_count(&mut self, _buffer: &[u8]) -> Result<u32, Self::Error> {
        Ok(16)
    }
}

export_mutator!(BtcserMutator);
