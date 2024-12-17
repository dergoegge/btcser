use btcser::{
    object::{
        decode_compact_size, decode_varint, decode_varint_non_negative, encode_compact_size,
        encode_varint, encode_varint_non_negative, find_value_in_object, ObjectParser,
        SerializedValue,
    },
    parser::{Descriptor, DescriptorParser, FieldPath, FieldType, IntType},
};

use crate::sampler::WeightedReservoirSampler;

#[derive(Debug, Clone, PartialEq)]
pub enum Mutation {
    Add,
    Delete,
    Mutate,
    Copy(Option<Vec<u8>>), // Overwrite the value of one field with that of another
    Clone(Option<Vec<u8>>), // Create a new field by copying the value of another
}

#[derive(Debug, Clone)]
pub struct SampledMutation {
    mutation: Mutation,
    path: FieldPath,

    // Certain mutations require an additional value to be modified.
    //
    // E.g.: mutating a slice by adding an element will require adjusting the preceding length
    // specifying field, i.e mutating the slice in `Test { u8,slice<u64,'0'> }` wil require
    // adjusting the first field of type u8.
    additional_path: Option<FieldPath>,
}

pub trait ByteArrayMutator {
    fn new(seed: u64) -> Self;
    fn mutate(&self, bytes: &mut Vec<u8>);
    fn mutate_in_place(&self, bytes: &mut [u8]);
}

pub struct StdSerializedValueMutator<B: ByteArrayMutator> {
    pub byte_array_mutator: B,
}

impl<'a, B: ByteArrayMutator> SerializedValueMutator<'a> for StdSerializedValueMutator<B> {
    fn mutate(&self, value: &SerializedValue<'a>) -> Result<Vec<u8>, String> {
        match &value.field_type {
            // For booleans, just flip the value
            FieldType::Bool => {
                let mut bytes = vec![value.bytes[0]];
                bytes[0] ^= 1; // Flip the least significant bit
                Ok(bytes)
            }

            // For integers, mutate in place
            FieldType::Int(int_type) => {
                let mut bytes = value.bytes.to_vec();
                match int_type {
                    // For variable-length integers, decode-mutate-encode
                    IntType::CompactSize(_) | IntType::VarInt | IntType::VarIntNonNegative => {
                        let mut decoded_bytes = match int_type {
                            IntType::CompactSize(_) => {
                                let (value, _, _) = decode_compact_size(&bytes)?;
                                value.to_le_bytes().to_vec()
                            }
                            IntType::VarInt => {
                                let (value, _) = decode_varint(&bytes)?;
                                value.to_le_bytes().to_vec()
                            }
                            IntType::VarIntNonNegative => {
                                let (value, _) = decode_varint_non_negative(&bytes)?;
                                value.to_le_bytes().to_vec()
                            }
                            _ => unreachable!(),
                        };

                        // Mutate the decoded value
                        self.byte_array_mutator.mutate_in_place(&mut decoded_bytes);

                        // Re-encode the mutated value
                        let value = u64::from_le_bytes(decoded_bytes[..8].try_into().unwrap());
                        return match int_type {
                            IntType::CompactSize(_) => Ok(encode_compact_size(value)),
                            IntType::VarInt => Ok(encode_varint(value)),
                            IntType::VarIntNonNegative => Ok(encode_varint_non_negative(value)),
                            _ => unreachable!(),
                        };
                    }
                    // For fixed-size integers, mutate the entire value
                    _ => {
                        self.byte_array_mutator.mutate_in_place(&mut bytes);
                    }
                };
                Ok(bytes)
            }

            // For byte arrays (fixed size), mutate in place
            FieldType::Bytes(_) => {
                let mut bytes = value.bytes.to_vec();
                self.byte_array_mutator.mutate_in_place(&mut bytes);
                Ok(bytes)
            }

            // For vectors, handle differently based on element type
            FieldType::Vec(inner_type) => {
                match &**inner_type {
                    // For Vec<u8>, treat as a single byte array but preserve the length
                    FieldType::Int(IntType::U8) => {
                        let length_size = match value.bytes[0] {
                            0..=252 => 1,
                            253 => 3,
                            254 => 5,
                            255 => 9,
                        };
                        let mut bytes = value.bytes[length_size..].to_vec();
                        // Only mutate the content, preserving the length prefix
                        self.byte_array_mutator.mutate(&mut bytes);
                        let mut len_prefixed = encode_compact_size(bytes.len() as u64);
                        len_prefixed.extend_from_slice(&bytes);
                        Ok(len_prefixed)
                    }
                    // For other vector types, return as is (handled by Add/Delete mutations)
                    _ => Ok(value.bytes.to_vec()),
                }
            }

            // For slices, handle differently based on element type
            FieldType::Slice(inner_type, _) => {
                match &**inner_type {
                    // For Slice<u8>, treat as a single byte array
                    FieldType::Int(IntType::U8) => {
                        let mut bytes = value.bytes.to_vec();
                        self.byte_array_mutator.mutate(&mut bytes);
                        Ok(bytes)
                    }
                    // For other slice types, return as is (handled by Add/Delete mutations)
                    _ => Ok(value.bytes.to_vec()),
                }
            }

            // Disallow mutations for structs
            FieldType::Struct(name) => {
                Err(format!("Mutation not supported for struct type: {}", name))
            }
        }
    }

    fn new(seed: u64) -> Self {
        Self {
            byte_array_mutator: ByteArrayMutator::new(seed),
        }
    }
}

pub trait SerializedValueMutator<'a> {
    fn new(seed: u64) -> Self;

    fn mutate(&self, value: &SerializedValue<'a>) -> Result<Vec<u8>, String>;

    fn generate(
        &self,
        field_type: &FieldType,
        parser: &DescriptorParser,
    ) -> Result<Vec<u8>, String> {
        match field_type {
            FieldType::Bool => Ok(vec![0]), // false
            FieldType::Int(int_type) => match int_type {
                IntType::U8 | IntType::I8 => Ok(vec![0]),
                IntType::U16 | IntType::U16BE | IntType::I16 | IntType::I16BE => Ok(vec![0, 0]),
                IntType::U32 | IntType::U32BE | IntType::I32 | IntType::I32BE => {
                    Ok(vec![0, 0, 0, 0])
                }
                IntType::U64 | IntType::U64BE | IntType::I64 | IntType::I64BE => {
                    Ok(vec![0, 0, 0, 0, 0, 0, 0, 0])
                }
                IntType::U256 | IntType::U256BE | IntType::I256 | IntType::I256BE => {
                    Ok(vec![0; 32])
                }
                IntType::CompactSize(_) | IntType::VarInt | IntType::VarIntNonNegative => {
                    Ok(vec![0])
                } // Minimal encoding for 0
            },
            FieldType::Bytes(size) => Ok(vec![0; *size]),
            FieldType::Vec(_) => {
                // Generate an empty vector
                Ok(vec![0]) // CompactSize(0)
            }
            FieldType::Slice(_, _) => Ok(vec![]), // Empty slice
            FieldType::Struct(name) => {
                // For structs, concatenate default values of all fields
                let mut bytes = Vec::new();
                if let Some(descriptor) = parser.get_descriptor(name) {
                    for field in &descriptor.fields {
                        if let Some(constant_value) = &field.constant_value {
                            bytes.extend(constant_value.clone());
                        } else {
                            bytes.extend(self.generate(&field.field_type, parser)?);
                        }
                    }
                    Ok(bytes)
                } else {
                    Err(format!("Unknown struct type: {}", name))
                }
            }
        }
    }

    fn fixup_length_field(
        &self,
        length_field: &SerializedValue<'a>,
        new_length: u64,
        parser: &DescriptorParser,
    ) -> Result<Vec<u8>, String> {
        match &length_field.field_type {
            FieldType::Int(int_type) => match int_type {
                IntType::U8 => {
                    if new_length > u8::MAX as u64 {
                        return Err(format!("Length {} too large for u8", new_length));
                    }
                    Ok(vec![new_length as u8])
                }
                IntType::U16 => {
                    if new_length > u16::MAX as u64 {
                        return Err(format!("Length {} too large for u16", new_length));
                    }
                    Ok(u16::to_le_bytes(new_length as u16).to_vec())
                }
                IntType::U16BE => {
                    if new_length > u16::MAX as u64 {
                        return Err(format!("Length {} too large for u16", new_length));
                    }
                    Ok(u16::to_be_bytes(new_length as u16).to_vec())
                }
                IntType::U32 => {
                    if new_length > u32::MAX as u64 {
                        return Err(format!("Length {} too large for u32", new_length));
                    }
                    Ok(u32::to_le_bytes(new_length as u32).to_vec())
                }
                IntType::U32BE => {
                    if new_length > u32::MAX as u64 {
                        return Err(format!("Length {} too large for u32", new_length));
                    }
                    Ok(u32::to_be_bytes(new_length as u32).to_vec())
                }
                IntType::U64 => Ok(u64::to_le_bytes(new_length).to_vec()),
                IntType::U64BE => Ok(u64::to_be_bytes(new_length).to_vec()),
                IntType::CompactSize(_) => Ok(encode_compact_size(new_length)),
                IntType::VarInt => Ok(encode_varint(new_length)),
                IntType::VarIntNonNegative => Ok(encode_varint_non_negative(new_length)),
                _ => Err(format!(
                    "Unsupported integer type for length field: {:?}",
                    int_type
                )),
            },
            FieldType::Vec(_) => {
                // Get the size of the old length prefix
                let old_length_size = match length_field.bytes[0] {
                    0..=252 => 1,
                    253 => 3,
                    254 => 5,
                    255 => 9,
                };

                let old_length = length_field.nested_values.len() as u64;
                assert!(old_length != new_length);
                let mut new_bytes = encode_compact_size(new_length);

                if new_length > old_length {
                    // Growing: keep existing contents
                    new_bytes.extend_from_slice(&length_field.bytes[old_length_size..]);

                    let vec_type = match &length_field.field_type {
                        FieldType::Vec(inner_type) => &**inner_type,
                        _ => unreachable!(),
                    };

                    // Generate a new element for the nested type
                    let new_element_bytes = self.generate(&vec_type, parser)?;
                    let new_elements = new_length - old_length;

                    // Append the new elements
                    for _ in 0..new_elements {
                        new_bytes.extend_from_slice(&new_element_bytes);
                    }
                } else if new_length < old_length && new_length != 0 {
                    // Shrinking: keep only the first new_length elements
                    let mut size_of_elements = 0;
                    for i in 0..new_length as usize {
                        size_of_elements += length_field.nested_values[i].bytes.len();
                    }
                    new_bytes.extend_from_slice(
                        &length_field.bytes[old_length_size..(old_length_size + size_of_elements)],
                    );
                }

                Ok(new_bytes)
            }
            _ => Err(format!(
                "Invalid field type for length field: {:?}",
                length_field.field_type
            )),
        }
    }
}

#[derive(Debug)]
struct PerformedMutation {
    mutation: Mutation,
    path: FieldPath,
    mutated_bytes: Vec<u8>,
}

fn compare_mutations(a: &PerformedMutation, b: &PerformedMutation) -> std::cmp::Ordering {
    // First compare by path indices
    let path_cmp = a.path.indices.cmp(&b.path.indices);
    if path_cmp != std::cmp::Ordering::Equal {
        return path_cmp;
    }

    // If paths are equal, use mutation type as tiebreaker
    // Mutate should come before Add/Delete
    match (&a.mutation, &b.mutation) {
        (Mutation::Mutate, Mutation::Mutate) => std::cmp::Ordering::Equal,
        (Mutation::Mutate, _) => std::cmp::Ordering::Less,
        (_, Mutation::Mutate) => std::cmp::Ordering::Greater,
        _ => std::cmp::Ordering::Equal,
    }
}

// Helper function to handle length field updates for collections
fn handle_length_update<'a, M>(
    field_type: &FieldType,
    new_length: u64,
    value: &SerializedValue,
    mutation_path: &FieldPath,
    additional_value: Option<&SerializedValue<'a>>,
    additional_path: Option<FieldPath>,
    mutator: &M,
    parser: &DescriptorParser,
    values: &[SerializedValue<'a>],
) -> Result<Vec<PerformedMutation>, String>
where
    M: SerializedValueMutator<'a>,
{
    match field_type {
        FieldType::Vec(_) => {
            // For vectors, update the length encoding at the start
            let length_bytes = encode_compact_size(new_length);
            let old_length_size = match value.bytes[0] {
                0..=252 => 1,
                253 => 3,
                254 => 5,
                255 => 9,
            };

            // Combine new length encoding with existing vector contents
            let mut combined_bytes = length_bytes;
            combined_bytes.extend_from_slice(&value.bytes[old_length_size..]);

            let mut mutations = vec![PerformedMutation {
                mutation: Mutation::Mutate,
                path: mutation_path.clone(),
                mutated_bytes: combined_bytes,
            }];

            // If this vector is a length field for a slice, add an Add/Delete mutation for the slice
            if let Some(slice_idx) = value.length_field_for {
                let mut slice_path = mutation_path.clone();
                slice_path.indices.pop();
                slice_path.indices.push(slice_idx as usize);

                let old_length = value.nested_values.len();
                let mut new_element = vec![];
                let slice_mutation = if new_length > old_length as u64 {
                    let slice_value = find_value_in_object(values, &slice_path)
                        .ok_or_else(|| "Could not find slice for vec length field!".to_string())?;

                    let slice_type = match &slice_value.field_type {
                        FieldType::Slice(inner_type, _) => &**inner_type,
                        _ => unreachable!(),
                    };

                    new_element = mutator.generate(slice_type, parser)?;

                    Mutation::Add
                } else {
                    Mutation::Delete
                };

                mutations.push(PerformedMutation {
                    mutation: slice_mutation,
                    path: slice_path,
                    mutated_bytes: new_element, // Include the generated element bytes
                });
            }

            Ok(mutations)
        }
        FieldType::Slice(_, _) => {
            // For slices, update the separate length field if present
            if let Some(length_field) = additional_value {
                let new_length_bytes =
                    mutator.fixup_length_field(length_field, new_length, parser)?;
                Ok(vec![PerformedMutation {
                    mutation: Mutation::Mutate,
                    path: additional_path.unwrap(),
                    mutated_bytes: new_length_bytes,
                }])
            } else {
                Ok(vec![])
            }
        }
        _ => unreachable!(),
    }
}

// Take a parsed binary object (i.e. list of `SerializedValue`s) and apply a mutation. Returns
// locations and mutated bytes for those locations.
fn mutate<'a, 'b: 'a, M: SerializedValueMutator<'a>>(
    values: &'b [SerializedValue<'b>],
    mutation: SampledMutation,
    mutator: &M,
    parser: &DescriptorParser,
) -> Result<Vec<PerformedMutation>, String> {
    let Some(value) = find_value_in_object(values, &mutation.path) else {
        return Err(format!(
            "Could not find value at path {:?} to mutate!",
            mutation.path.indices
        ));
    };

    let additional_value = if let Some(path) = &mutation.additional_path {
        Some(
            find_value_in_object(values, &path)
                .ok_or_else(|| "Could not find additional value for mutation!".to_string())?,
        )
    } else {
        None
    };

    match mutation.mutation {
        Mutation::Mutate | Mutation::Copy(Some(_)) => {
            let mutated_bytes = match mutation.mutation {
                Mutation::Mutate => mutator.mutate(value)?,
                Mutation::Copy(Some(other_bytes)) => other_bytes,
                _ => unreachable!(),
            };
            let new_length = mutated_bytes.len();

            let mut mutations = vec![PerformedMutation {
                mutation: Mutation::Mutate,
                path: mutation.path.clone(),
                mutated_bytes,
            }];

            if new_length != value.bytes.len() {
                // If the length changed and there is and additional path value present, then we
                // adjust the length field (i.e. the additional value).
                if let Some(length_field) = additional_value {
                    let new_length_bytes =
                        mutator.fixup_length_field(length_field, new_length as u64, parser)?;
                    mutations.push(PerformedMutation {
                        mutation: Mutation::Mutate,
                        path: mutation.additional_path.clone().unwrap(),
                        mutated_bytes: new_length_bytes,
                    });
                }
            }

            Ok(mutations)
        }
        Mutation::Add | Mutation::Clone(Some(_)) => {
            let element_type = match &value.field_type {
                FieldType::Vec(inner_type) => inner_type,
                FieldType::Slice(inner_type, _) => inner_type,
                _ => {
                    return Err("Can only add elements to Vec or Slice types".to_string());
                }
            };
            let new_element = match mutation.mutation {
                Mutation::Add => mutator.generate(&**element_type, parser)?,
                Mutation::Clone(Some(cloned_bytes)) => cloned_bytes,
                _ => unreachable!(),
            };
            let new_length = value.nested_values.len() + 1;

            let mut mutations = vec![PerformedMutation {
                mutation: Mutation::Add,
                path: mutation.path.clone(),
                mutated_bytes: new_element,
            }];

            // Handle length field updates
            mutations.extend(handle_length_update(
                &value.field_type,
                new_length as u64,
                value,
                &mutation.path,
                additional_value,
                mutation.additional_path.clone(),
                mutator,
                parser,
                values,
            )?);

            Ok(mutations)
        }
        Mutation::Delete => {
            if value.nested_values.is_empty() {
                return Err("Cannot delete from empty collection".to_string());
            }

            let new_length = value.nested_values.len() - 1;
            let mut mutations = Vec::new();

            match &value.field_type {
                FieldType::Vec(_) => {
                    let element_size = value
                        .nested_values
                        .last()
                        .expect("vector can't be empty here")
                        .bytes
                        .len(); // size of last element we're gonna cut off
                    let content_end = value.bytes.len() - element_size;

                    // Handle length field updates with truncated content
                    let mut length_mutations = handle_length_update(
                        &value.field_type,
                        new_length as u64,
                        value,
                        &mutation.path,
                        additional_value,
                        mutation.additional_path.clone(),
                        mutator,
                        parser,
                        values,
                    )?;

                    // For vectors, modify the combined bytes to exclude the last element
                    if let Some(length_mutation) = length_mutations.get_mut(0) {
                        length_mutation.mutated_bytes.truncate(content_end);
                    }
                    mutations.extend(length_mutations);
                }
                FieldType::Slice(_, _) => {
                    mutations.push(PerformedMutation {
                        mutation: Mutation::Delete,
                        path: mutation.path.clone(),
                        mutated_bytes: Vec::new(),
                    });

                    // Handle length field updates
                    mutations.extend(handle_length_update(
                        &value.field_type,
                        new_length as u64,
                        value,
                        &mutation.path,
                        additional_value,
                        mutation.additional_path.clone(),
                        mutator,
                        parser,
                        values,
                    )?);
                }
                _ => unreachable!(),
            }

            Ok(mutations)
        }
        _ => Err("unsupported mutation".to_string()),
    }
}

// Given a set of parsed values and a set of mutations, apply the mutations to the values and return
// the final serialized continuous bytes.
fn finalize_mutations<'a>(
    original_data: &[u8],
    parsed_values: &[SerializedValue<'a>],
    mutations: Vec<PerformedMutation>,
) -> Result<Vec<u8>, String> {
    // Sort mutations by path to ensure we process them in order
    let mut sorted_mutations = mutations;
    sorted_mutations.sort_by(compare_mutations);

    let mut serialized = Vec::new();
    let mut current_position = 0;

    // Process each mutation in order
    for mutation in sorted_mutations {
        // Find the value being mutated
        let value = find_value_in_object(parsed_values, &mutation.path)
            .ok_or_else(|| "Invalid mutation path".to_string())?;

        // Calculate the start position of this value in the original data
        let value_start = value.bytes.as_ptr() as usize - original_data.as_ptr() as usize;

        // Copy any bytes between the current position and the start of this value
        if value_start > current_position {
            serialized.extend(&original_data[current_position..value_start]);
        }

        // Apply the mutation
        match mutation.mutation {
            Mutation::Mutate => {
                serialized.extend(&mutation.mutated_bytes);
                current_position = value_start + value.bytes.len();
            }
            Mutation::Add => {
                match &value.field_type {
                    FieldType::Vec(_) => {
                        // For vectors, only add the new element bytes
                        // (the Mutate mutation will handle the length encoding)
                        serialized.extend(&mutation.mutated_bytes);
                    }
                    _ => {
                        // For slices, keep existing elements and append new one
                        serialized.extend(value.bytes);
                        serialized.extend(&mutation.mutated_bytes);
                    }
                }
                current_position = value_start + value.bytes.len();
            }
            Mutation::Delete => {
                // For deletion, we need to calculate the size of one element
                let element_size = match &value.field_type {
                    FieldType::Slice(_, _) => value
                        .nested_values
                        .last()
                        .expect("should not sample delete mutation on empty slice")
                        .bytes
                        .len(),
                    _ => return Err("Delete mutation only supported for Slice types".to_string()),
                };

                // Copy all bytes except the last element
                serialized.extend(&value.bytes[..value.bytes.len() - element_size]);
                current_position = value_start + value.bytes.len();
            }
            _ => return Err("unsupported mutation".to_string()),
        }
    }

    // Copy any remaining bytes after the last mutation
    if current_position < original_data.len() {
        serialized.extend(&original_data[current_position..]);
    }

    Ok(serialized)
}

pub struct Mutator<'p> {
    descriptor: Descriptor,
    parser: &'p DescriptorParser,
}

impl<'p> Mutator<'p> {
    fn sample_mutations<'a, S>(
        &self,
        values: &'a [SerializedValue<'a>],
        sampler: &mut S,
        current_path: Vec<usize>,
        cross_over: bool,
    ) where
        S: WeightedReservoirSampler<SampledMutation>,
    {
        let (cross_over_weight, mutate_weight) = if cross_over { (1.0, 0.0) } else { (0.0, 1.0) };

        // For each value at the current level
        for (idx, value) in values.iter().enumerate() {
            let mut path = current_path.clone();
            path.push(idx);
            let field_path = FieldPath::new(path.clone());

            // Sample potential mutations for this value
            match &value.field_type {
                FieldType::Bool => {
                    if value.length_field_for.is_none() {
                        sampler.add(
                            SampledMutation {
                                mutation: Mutation::Mutate,
                                path: field_path.clone(),
                                additional_path: None,
                            },
                            mutate_weight,
                        );

                        sampler.add(
                            SampledMutation {
                                mutation: Mutation::Copy(None),
                                path: field_path,
                                additional_path: None,
                            },
                            cross_over_weight,
                        );
                    }
                }
                FieldType::Int(_) => {
                    // Skip mutation if this is a length field
                    if value.length_field_for.is_none() {
                        sampler.add(
                            SampledMutation {
                                mutation: Mutation::Copy(None),
                                path: field_path.clone(),
                                additional_path: None,
                            },
                            cross_over_weight,
                        );

                        sampler.add(
                            SampledMutation {
                                mutation: Mutation::Mutate,
                                path: field_path,
                                additional_path: None,
                            },
                            mutate_weight,
                        );
                    }
                }
                FieldType::Bytes(_) => {
                    sampler.add(
                        SampledMutation {
                            mutation: Mutation::Copy(None),
                            path: field_path.clone(),
                            additional_path: None,
                        },
                        cross_over_weight,
                    );

                    sampler.add(
                        SampledMutation {
                            mutation: Mutation::Mutate,
                            path: field_path,
                            additional_path: None,
                        },
                        mutate_weight,
                    );
                }
                FieldType::Vec(inner_type) => {
                    if value.length_field_for.is_none() {
                        sampler.add(
                            SampledMutation {
                                mutation: Mutation::Copy(None),
                                path: field_path.clone(),
                                additional_path: None,
                            },
                            cross_over_weight,
                        );
                    }

                    if matches!(**inner_type, FieldType::Int(IntType::U8)) {
                        if value.length_field_for.is_none() {
                            sampler.add(
                                SampledMutation {
                                    mutation: Mutation::Mutate,
                                    path: field_path,
                                    additional_path: None,
                                },
                                mutate_weight,
                            );
                        }
                    } else {
                        sampler.add(
                            SampledMutation {
                                mutation: Mutation::Clone(None),
                                path: field_path.clone(),
                                additional_path: None,
                            },
                            cross_over_weight,
                        );
                        sampler.add(
                            SampledMutation {
                                mutation: Mutation::Add,
                                path: field_path.clone(),
                                additional_path: None,
                            },
                            mutate_weight,
                        );

                        if !value.nested_values.is_empty() {
                            sampler.add(
                                SampledMutation {
                                    mutation: Mutation::Delete,
                                    path: field_path,
                                    additional_path: None,
                                },
                                mutate_weight,
                            );
                        }

                        self.sample_mutations(&value.nested_values, sampler, path, cross_over);
                    }
                }
                FieldType::Slice(inner_type, idx) => {
                    let mut additional_path = field_path.clone();
                    additional_path.indices.pop();
                    additional_path.indices.push(*idx as usize);

                    // Get the length field value directly
                    let length_field = &values[*idx as usize];

                    if matches!(**inner_type, FieldType::Int(IntType::U8)) {
                        // TODO allow copying for all slices not just byte slices. For that, we'll
                        // need to store the length of the slice to be copied on the copy mutation
                        // somehow...
                        sampler.add(
                            SampledMutation {
                                mutation: Mutation::Copy(None),
                                path: field_path.clone(),
                                additional_path: Some(additional_path.clone()),
                            },
                            cross_over_weight,
                        );

                        sampler.add(
                            SampledMutation {
                                mutation: Mutation::Mutate,
                                path: field_path,
                                additional_path: Some(additional_path),
                            },
                            mutate_weight,
                        );
                    } else {
                        // For boolean length fields, only allow Add if the slice is empty
                        let should_allow_add = match &length_field.field_type {
                            FieldType::Bool => value.nested_values.is_empty(),
                            _ => true,
                        };

                        if should_allow_add {
                            sampler.add(
                                SampledMutation {
                                    mutation: Mutation::Clone(None),
                                    path: field_path.clone(),
                                    additional_path: Some(additional_path.clone()),
                                },
                                cross_over_weight,
                            );

                            sampler.add(
                                SampledMutation {
                                    mutation: Mutation::Add,
                                    path: field_path.clone(),
                                    additional_path: Some(additional_path.clone()),
                                },
                                mutate_weight,
                            );
                        }

                        if !value.nested_values.is_empty() {
                            sampler.add(
                                SampledMutation {
                                    mutation: Mutation::Delete,
                                    path: field_path,
                                    additional_path: Some(additional_path),
                                },
                                mutate_weight,
                            );
                        }

                        self.sample_mutations(&value.nested_values, sampler, path, cross_over);
                    }
                }
                FieldType::Struct(_) => {
                    sampler.add(
                        SampledMutation {
                            mutation: Mutation::Copy(None),
                            path: field_path.clone(),
                            additional_path: None,
                        },
                        cross_over_weight,
                    );

                    self.sample_mutations(&value.nested_values, sampler, path, cross_over);
                }
            }
        }
    }

    fn sample_data_sources<'a, S>(
        &self,
        values: &'a [SerializedValue<'a>],
        field_type: &FieldType,
        sampler: &mut S,
        current_path: Vec<usize>,
    ) where
        S: WeightedReservoirSampler<FieldPath>,
    {
        // For each value at the current level
        for (idx, value) in values.iter().enumerate() {
            let mut path = current_path.clone();
            path.push(idx);

            // Check if this value's type matches the target field_type
            if value.field_type == *field_type {
                sampler.add(FieldPath::new(path.clone()), 1.0);
            }

            // Recursively check nested values
            match &value.field_type {
                FieldType::Vec(_) | FieldType::Slice(_, _) | FieldType::Struct(_) => {
                    self.sample_data_sources(&value.nested_values, field_type, sampler, path);
                }
                _ => {} // Other types don't have nested values
            }
        }
    }

    /// Mutates the given binary data according to the descriptor's format specification.
    ///
    /// # Arguments
    /// * `data` - A slice of bytes to mutate
    /// * `seed` - A seed value for deterministic mutation sampling
    ///
    /// # Type Parameters
    /// * `S` - A type implementing `WeightedReservoirSampler<SampledMutation>`
    /// * `M` - A type implementing `SerializedValueMutator`
    ///
    /// # Returns
    /// * `Ok(Vec<u8>)` - The mutated bytes
    /// * `Err(String)` - An error message if mutation fails
    pub fn mutate<'b, S, M>(&self, data: &'b [u8], seed: u64) -> Result<Vec<u8>, String>
    where
        S: WeightedReservoirSampler<SampledMutation>,
        M: for<'a> SerializedValueMutator<'a>,
        'b: 'p,
    {
        let mut sampler = S::new(seed);
        let mutator = M::new(seed);
        let obj_parser = ObjectParser::new(self.descriptor.clone(), self.parser);

        // Try parsing the blob according to `self.descriptor`
        let values = match obj_parser.parse(data) {
            Ok(v) => v,
            Err(_) => {
                // If parsing fails, generate a dummy value using the descriptor
                return mutator.generate(
                    &FieldType::Struct(self.descriptor.name.clone()),
                    self.parser,
                );
            }
        };

        // Given the parsed object, sample all possible mutations and pick one!
        self.sample_mutations(&values, &mut sampler, vec![], false); // no cross-over
        let mutation = sampler
            .get_sample()
            .ok_or_else(|| "No mutation sample available".to_string())?;

        // Perform the mutation
        let performed = mutate(&values, mutation, &mutator, self.parser)?;
        // Reserialize the object with the mutation applied
        finalize_mutations(data, &values, performed)
    }

    pub fn cross_over<'b, S, D, M>(
        &self,
        to_mutate: &'b [u8],
        sample_from: &'b [u8],
        seed: u64,
    ) -> Result<Vec<u8>, String>
    where
        S: WeightedReservoirSampler<SampledMutation>,
        D: WeightedReservoirSampler<FieldPath>,
        M: for<'a> SerializedValueMutator<'a>,
        'b: 'p,
    {
        let mut sampler = S::new(seed);
        let mutator = M::new(seed);
        let obj_parser = ObjectParser::new(self.descriptor.clone(), self.parser);

        let values_to_mutate = obj_parser.parse(to_mutate)?;
        let values_to_sample_from = obj_parser.parse(sample_from)?;

        self.sample_mutations(&values_to_mutate, &mut sampler, vec![], true);
        let mut mutation = sampler
            .get_sample()
            .ok_or_else(|| "No mutation sample available".to_string())?;

        let data_source = match mutation.mutation {
            Mutation::Clone(None) | Mutation::Copy(None) => {
                let mut source_sampler = D::new(seed);
                let value = find_value_in_object(&values_to_mutate, &mutation.path)
                    .expect("path to sampled mutation has to exist");

                // Get the target type based on mutation type
                let target_type = match &mutation.mutation {
                    Mutation::Clone(None) => match &value.field_type {
                        FieldType::Vec(inner_type) => &**inner_type,
                        FieldType::Slice(inner_type, _) => &**inner_type,
                        _ => &value.field_type,
                    },
                    Mutation::Copy(None) => &value.field_type,
                    _ => unreachable!(),
                };

                // Sample data sources using the correct type
                self.sample_data_sources(
                    &values_to_sample_from,
                    target_type,
                    &mut source_sampler,
                    vec![],
                );

                source_sampler
                    .get_sample()
                    .ok_or_else(|| "no data source found".to_string())
            }
            _ => {
                return Err(
                    "Only Clone and Copy are supported during cross-over mutations!".to_string(),
                )
            }
        }?;

        let source_value = find_value_in_object(&values_to_sample_from, &data_source)
            .expect("path to sampled data source has to exist");

        mutation.mutation = match mutation.mutation.clone() {
            Mutation::Copy(None) => Mutation::Copy(Some(source_value.bytes.to_vec())),
            Mutation::Clone(None) => Mutation::Clone(Some(source_value.bytes.to_vec())),
            _ => unreachable!(),
        };

        // Perform the mutation
        let performed = mutate(&values_to_mutate, mutation, &mutator, self.parser)?;
        // Reserialize the object with the mutation applied
        finalize_mutations(to_mutate, &values_to_mutate, performed)
    }

    /// Creates a new Mutator instance with the given descriptor and parser.
    ///
    /// # Arguments
    /// * `descriptor` - The descriptor defining the data format
    /// * `parser` - Reference to the descriptor parser
    ///
    /// # Returns
    /// A new Mutator instance
    pub fn new(descriptor: Descriptor, parser: &'p DescriptorParser) -> Self {
        Self { descriptor, parser }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::tests::TestSampler;
    use crate::sampler::ChaoSampler;

    // Helper function to parse descriptor with a specific parser instance
    fn parse_descriptor_with_parser(
        content: &str,
        parser: &mut DescriptorParser,
    ) -> Result<Descriptor, String> {
        parser.parse_file(content).unwrap();
        parser
            .get_descriptor("Test")
            .cloned()
            .ok_or_else(|| "Descriptor 'Test' not found".to_string())
    }

    // Add this new struct for testing
    struct TestByteArrayMutator;

    impl ByteArrayMutator for TestByteArrayMutator {
        fn new(_seed: u64) -> Self {
            Self {}
        }
        fn mutate(&self, bytes: &mut Vec<u8>) {
            bytes.fill(0xFF);
        }

        fn mutate_in_place(&self, bytes: &mut [u8]) {
            bytes.fill(0xFF);
        }
    }

    #[test]
    fn test_sample_mutations_basic_types() -> Result<(), String> {
        let mut parser = DescriptorParser::new();
        let descriptor =
            parse_descriptor_with_parser("Test { bool, u8, bytes<4> }", &mut parser).unwrap();
        let obj_parser = ObjectParser::new(descriptor.clone(), &parser);

        let mutator = Mutator {
            descriptor,
            parser: &parser,
        };

        let data = [
            0x01, // bool
            0x42, // u8
            0x01, 0x02, 0x03, 0x04, // bytes<4>
        ];

        let values = obj_parser.parse(&data).unwrap();
        let mut sampler = TestSampler::new(0);
        mutator.sample_mutations(&values, &mut sampler, Vec::new(), false);

        // We expect 3 Mutate mutations (one for each field)
        let samples = sampler.get_samples();
        assert_eq!(samples.len(), 3);

        // Check each mutation matches exactly what we expect
        assert!(matches!(
            samples[0],
            SampledMutation {
                mutation: Mutation::Mutate,
                path: FieldPath { indices: ref idx },
                additional_path: None
            } if idx == &vec![0] && matches!(values[0].field_type, FieldType::Bool)
        ));

        assert!(matches!(
            samples[1],
            SampledMutation {
                mutation: Mutation::Mutate,
                path: FieldPath { indices: ref idx },
                additional_path: None
            } if idx == &vec![1] && matches!(values[1].field_type, FieldType::Int(IntType::U8))
        ));

        assert!(matches!(
            samples[2],
            SampledMutation {
                mutation: Mutation::Mutate,
                path: FieldPath { indices: ref idx },
                additional_path: None
            } if idx == &vec![2] && matches!(values[2].field_type, FieldType::Bytes(4))
        ));

        Ok(())
    }

    #[test]
    fn test_sample_mutations_byte_arrays() -> Result<(), String> {
        let mut parser = DescriptorParser::new();
        let descriptor =
            parse_descriptor_with_parser("Test { vec<u8>, slice<u8, '0'> }", &mut parser).unwrap();
        let obj_parser = ObjectParser::new(descriptor.clone(), &parser);

        let mutator = Mutator {
            descriptor,
            parser: &parser,
        };

        let data = [
            0x02, // vec length
            0x42, 0x43, // vec contents
            0x44, 0x45, // slice contents
        ];

        let values = obj_parser.parse(&data).unwrap();
        let mut sampler = TestSampler::new(0);
        mutator.sample_mutations(&values, &mut sampler, Vec::new(), false);

        // We expect exactly 1 Mutate mutation (for slice)
        let samples = sampler.get_samples();
        assert_eq!(samples.len(), 1);
        assert!(matches!(samples[0].mutation, Mutation::Mutate));

        Ok(())
    }

    #[test]
    fn test_sample_mutations_slice_with_custom_type() -> Result<(), String> {
        // Define a struct with a u8 field that will be used in the slice
        let mut parser = DescriptorParser::new();
        parser
            .parse_file(
                "Inner { u8 }
             Outer { u8, slice<Inner, '0'> }",
            )
            .unwrap();

        let descriptor = parser.descriptors.get("Outer").unwrap().clone();
        let obj_parser = ObjectParser::new(descriptor.clone(), &parser);

        let mutator = Mutator {
            descriptor,
            parser: &parser,
        };

        let data = [
            0x02, // length field (u8 = 2)
            0x42, 0x43, // two Inner structs with u8 values
        ];

        let values = obj_parser.parse(&data).unwrap();
        let mut sampler = TestSampler::new(0);
        mutator.sample_mutations(&values, &mut sampler, Vec::new(), false);

        // We expect:
        // 1. Mutations for the slice itself (add, delete), additional_path required
        // 2. Two mutations for each Inner struct's u8 field
        let samples = sampler.get_samples();
        assert_eq!(samples.len(), 4);

        println!("{:#?}", samples);
        // Check slice mutations
        assert!(matches!(
            &samples[0],
            SampledMutation {
                mutation: Mutation::Add,
                path: FieldPath { indices: ref idx },
                additional_path: Some(FieldPath { indices: ref add_idx })
            } if idx == &vec![1] && add_idx == &vec![0]
        ));
        assert!(matches!(
            &samples[1],
            SampledMutation {
                mutation: Mutation::Delete,
                path: FieldPath { indices: ref idx },
                additional_path: Some(FieldPath { indices: ref add_idx })
            } if idx == &vec![1] && add_idx == &vec![0]
        ));

        // Check first Inner struct's u8 field
        assert!(matches!(
            &samples[2],
            SampledMutation {
                mutation: Mutation::Mutate,
                path: FieldPath { indices: ref idx },
                additional_path: None,
            } if idx == &vec![1, 0, 0]
        ));

        // Check second Inner struct's u8 field
        assert!(matches!(
            &samples[3],
            SampledMutation {
                mutation: Mutation::Mutate,
                path: FieldPath { indices: ref idx },
                additional_path: None
            } if idx == &vec![1, 1, 0]
        ));

        Ok(())
    }

    #[test]
    fn test_basic_mutations() -> Result<(), String> {
        let mut parser = DescriptorParser::new();
        let descriptor =
            parse_descriptor_with_parser("Test { bool, u8, u16, u256, bytes<4> }", &mut parser)
                .unwrap();
        let obj_parser = ObjectParser::new(descriptor, &parser);

        let data = [
            0x01, // bool (true)
            0x42, // u8 (66)
            0x34, 0x12, // u16 (0x1234 = 4660)
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
            0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b,
            0x1c, 0x1d, 0x1e, 0x1f, // u256 (32 bytes)
            0xaa, 0xbb, 0xcc, 0xdd, // bytes<4>
        ];

        let values = obj_parser.parse(&data).unwrap();
        let mutator = StdSerializedValueMutator {
            byte_array_mutator: TestByteArrayMutator,
        };

        // Test mutating bool
        let bool_mutation = SampledMutation {
            mutation: Mutation::Mutate,
            path: FieldPath::new(vec![0]),
            additional_path: None,
        };
        let result = mutate(&values, bool_mutation, &mutator, &parser).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].path.indices, vec![0]);
        assert_eq!(result[0].mutated_bytes, vec![0x0]);

        // Test mutating u8
        let u8_mutation = SampledMutation {
            mutation: Mutation::Mutate,
            path: FieldPath::new(vec![1]),
            additional_path: None,
        };
        let result = mutate(&values, u8_mutation, &mutator, &parser).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].path.indices, vec![1]);
        assert_eq!(result[0].mutated_bytes, vec![0xFF]);

        // Test mutating u16
        let u16_mutation = SampledMutation {
            mutation: Mutation::Mutate,
            path: FieldPath::new(vec![2]),
            additional_path: None,
        };
        let result = mutate(&values, u16_mutation, &mutator, &parser).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].path.indices, vec![2]);
        assert_eq!(result[0].mutated_bytes, vec![0xFF, 0xFF]);

        // Test mutating u256
        let u256_mutation = SampledMutation {
            mutation: Mutation::Mutate,
            path: FieldPath::new(vec![3]),
            additional_path: None,
        };
        let result = mutate(&values, u256_mutation, &mutator, &parser).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].path.indices, vec![3]);
        assert_eq!(result[0].mutated_bytes, vec![0xFF; 32]);

        // Test mutating bytes
        let bytes_mutation = SampledMutation {
            mutation: Mutation::Mutate,
            path: FieldPath::new(vec![4]),
            additional_path: None,
        };
        let result = mutate(&values, bytes_mutation, &mutator, &parser).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].path.indices, vec![4]);
        assert_eq!(result[0].mutated_bytes, vec![0xFF; 4]);

        // Test invalid path
        let invalid_mutation = SampledMutation {
            mutation: Mutation::Mutate,
            path: FieldPath::new(vec![99]), // Invalid index
            additional_path: None,
        };
        assert!(mutate(&values, invalid_mutation, &mutator, &parser).is_err());

        Ok(())
    }

    #[test]
    fn test_struct_generation() -> Result<(), String> {
        let mut parser = DescriptorParser::new();
        parser
            .parse_file(
                "Inner { u8, u16 }
             Outer { Inner, vec<Inner> }",
            )
            .unwrap();

        let mutator = StdSerializedValueMutator {
            byte_array_mutator: TestByteArrayMutator,
        };

        // Test generating an Inner struct
        let inner_type = FieldType::Struct("Inner".to_string());
        let inner_bytes = mutator.generate(&inner_type, &parser).unwrap();
        assert_eq!(inner_bytes, vec![0x0, 0x0, 0x0]); // u8 + u16

        // Test generating an Outer struct
        let outer_type = FieldType::Struct("Outer".to_string());
        let outer_bytes = mutator.generate(&outer_type, &parser).unwrap();
        // Should contain: Inner struct (3 bytes) + Vec<Inner> (1 byte length + 3 bytes data)
        assert_eq!(
            outer_bytes,
            vec![
                0x0, 0x0, 0x0, // First Inner
                0x0, // Vec length
            ]
        );

        Ok(())
    }

    #[test]
    fn test_mutation_cycle_end_to_end() -> Result<(), String> {
        // Define test cases as a vector of tuples:
        // (descriptor, initial_data, mutation, expected_result, test_name)
        let test_cases = vec![
            (
                // Basic u16 mutation case
                r#"Test {
                    u8,                 # length field for the following slice
                    slice<u16, '0'>,    # slice of u16s, length determined by field 0
                    bytes<2>            # trailing bytes
                }"#,
                vec![
                    0x02, // length = 2
                    0x34, 0x12, // first u16 (LE)
                    0x78, 0x56, // second u16 (LE)
                    0xAA, 0xBB, // trailing bytes
                ],
                SampledMutation {
                    mutation: Mutation::Mutate,
                    path: FieldPath::new(vec![1, 0]), // First u16 in the slice
                    additional_path: None,
                },
                vec![
                    0x02, // length still 2
                    0xFF, 0xFF, // mutated first u16
                    0x78, 0x56, // original second u16
                    0xAA, 0xBB, // original trailing bytes
                ],
                "basic u16 mutation",
            ),
            (
                // Add element case
                r#"Test {
                    u8,                 # length field for the following slice
                    slice<u16, '0'>,    # slice of u16s, length determined by field 0
                    bytes<2>            # trailing bytes
                }"#,
                vec![
                    0x02, // length = 2
                    0x34, 0x12, // first u16 (LE)
                    0x78, 0x56, // second u16 (LE)
                    0xAA, 0xBB, // trailing bytes
                ],
                SampledMutation {
                    mutation: Mutation::Add,
                    path: FieldPath::new(vec![1]),
                    additional_path: Some(FieldPath::new(vec![0])),
                },
                vec![
                    0x03, // length now 3
                    0x34, 0x12, // original first u16
                    0x78, 0x56, // original second u16
                    0x00, 0x00, // new u16
                    0xAA, 0xBB, // original trailing bytes
                ],
                "add element to slice",
            ),
            (
                // Delete element case
                r#"Test {
                    u8,                 # length field for the following slice
                    slice<u16, '0'>,    # slice of u16s, length determined by field 0
                    bytes<2>            # trailing bytes
                }"#,
                vec![
                    0x02, // length = 2
                    0x34, 0x12, // first u16 (LE)
                    0x78, 0x56, // second u16 (LE)
                    0xAA, 0xBB, // trailing bytes
                ],
                SampledMutation {
                    mutation: Mutation::Delete,
                    path: FieldPath::new(vec![1]),
                    additional_path: Some(FieldPath::new(vec![0])),
                },
                vec![
                    0x01, // length now 1
                    0x34, 0x12, // only first u16 remains
                    0xAA, 0xBB, // original trailing bytes
                ],
                "delete element from slice",
            ),
            (
                // Nested struct mutation
                r#"Inner { u16 }
                   Test {
                       u8,                 # length field
                       slice<Inner, '0'>,  # slice of Inner structs
                       bytes<2>            # trailing bytes
                   }"#,
                vec![
                    0x02, // length = 2
                    0x34, 0x12, // first Inner.u16 (LE)
                    0x78, 0x56, // second Inner.u16 (LE)
                    0xAA, 0xBB, // trailing bytes
                ],
                SampledMutation {
                    mutation: Mutation::Mutate,
                    path: FieldPath::new(vec![1, 0, 0]), // First Inner struct's u16
                    additional_path: None,
                },
                vec![
                    0x02, // length still 2
                    0xFF, 0xFF, // mutated first Inner.u16
                    0x78, 0x56, // original second Inner.u16
                    0xAA, 0xBB, // original trailing bytes
                ],
                "nested struct mutation",
            ),
            (
                // Empty slice case
                r#"Test {
                    u8,                 # length field for the following slice
                    slice<u16, '0'>,    # slice of u16s, length determined by field 0
                    bytes<2>            # trailing bytes
                }"#,
                vec![
                    0x00, // length = 0
                    0xAA, 0xBB, // trailing bytes
                ],
                SampledMutation {
                    mutation: Mutation::Add,
                    path: FieldPath::new(vec![1]),
                    additional_path: Some(FieldPath::new(vec![0])),
                },
                vec![
                    0x01, // length now 1
                    0x00, 0x00, // new u16
                    0xAA, 0xBB, // original trailing bytes
                ],
                "add to empty slice",
            ),
            (
                r#"Test {
                    vec<u16>,           # length field for the following slice
                    slice<u16, '0'>,    # slice of u16s, length determined by field 0
                    bytes<2>            # trailing bytes
                }"#,
                vec![
                    0x00, // length = 0
                    0xAA, 0xBB, // trailing bytes
                ],
                SampledMutation {
                    mutation: Mutation::Add,
                    path: FieldPath::new(vec![0]),
                    additional_path: None,
                },
                vec![
                    0x01, // length now 1
                    0x00, 0x00, // new u16
                    0x00, 0x00, // new slice contents (mirroring the length of the vec)
                    0xAA, 0xBB, // trailing bytes
                ],
                "change slice length field (vec<u8>)",
            ),
            (
                r#"Test {
                    vec<u16>,           # length field for the following slice
                    slice<u16, '0'>,    # slice of u16s, length determined by field 0
                    bytes<2>            # trailing bytes
                }"#,
                vec![
                    0x00, // length = 0
                    0xAA, 0xBB, // trailing bytes
                ],
                SampledMutation {
                    mutation: Mutation::Add,
                    path: FieldPath::new(vec![1]),
                    additional_path: Some(FieldPath::new(vec![0])),
                },
                vec![
                    0x01, // length now 1
                    0x00, 0x00, // new u16
                    0x00, 0x00, // new slice contents (mirroring the length of the vec)
                    0xAA, 0xBB, // trailing bytes
                ],
                "change slice length (length field vec<u8> should also be updated)",
            ),
            (
                // Vector of u8s (byte array) mutation
                r#"Test {
                    vec<u8>,           # vector of bytes
                    bytes<2>           # trailing bytes
                }"#,
                vec![
                    0x03, // length = 3
                    0x11, 0x22, 0x33, // three bytes
                    0xAA, 0xBB, // trailing bytes
                ],
                SampledMutation {
                    mutation: Mutation::Mutate,
                    path: FieldPath::new(vec![0]),
                    additional_path: None,
                },
                vec![
                    0x03, // length still 3
                    0xFF, 0xFF, 0xFF, // mutated bytes
                    0xAA, 0xBB, // original trailing bytes
                ],
                "vector of bytes mutation",
            ),
            (
                // Multiple nested vectors
                r#"Test {
                    vec<vec<u16>>,     # vector of vectors of u16
                    bytes<2>           # trailing bytes
                }"#,
                vec![
                    0x02, // outer length = 2
                    0x01, // first inner length = 1
                    0x34, 0x12, // first inner vector's u16
                    0x02, // second inner length = 2
                    0x56, 0x34, // second inner vector's first u16
                    0x78, 0x56, // second inner vector's second u16
                    0xAA, 0xBB, // trailing bytes
                ],
                SampledMutation {
                    mutation: Mutation::Mutate,
                    path: FieldPath::new(vec![0, 1, 0]), // First u16 in second inner vector
                    additional_path: None,
                },
                vec![
                    0x02, // outer length = 2
                    0x01, // first inner length = 1
                    0x34, 0x12, // original first inner vector's u16
                    0x02, // second inner length = 2
                    0xFF, 0xFF, // mutated u16
                    0x78, 0x56, // original second u16
                    0xAA, 0xBB, // original trailing bytes
                ],
                "nested vector mutation",
            ),
            (
                // Add element to vec<u16>
                r#"Test {
                    vec<u16>,        # vector of u16s
                    bytes<2>         # trailing bytes
                }"#,
                vec![
                    0x02, // length = 2
                    0x34, 0x12, // first u16 (LE)
                    0x78, 0x56, // second u16 (LE)
                    0xAA, 0xBB, // trailing bytes
                ],
                SampledMutation {
                    mutation: Mutation::Add,
                    path: FieldPath::new(vec![0]),
                    additional_path: None,
                },
                vec![
                    0x03, // length now 3
                    0x34, 0x12, // original first u16
                    0x78, 0x56, // original second u16
                    0x00, 0x00, // new u16
                    0xAA, 0xBB, // original trailing bytes
                ],
                "add element to vec<u16>",
            ),
            (
                // Add element to empty vec<u16>
                r#"Test {
                    vec<u16>,        # vector of u16s
                    bytes<2>         # trailing bytes
                }"#,
                vec![
                    0x00, // length = 0
                    0xAA, 0xBB, // trailing bytes
                ],
                SampledMutation {
                    mutation: Mutation::Add,
                    path: FieldPath::new(vec![0]),
                    additional_path: None,
                },
                vec![
                    0x01, // length now 1
                    0x00, 0x00, // new u16
                    0xAA, 0xBB, // original trailing bytes
                ],
                "add element to empty vec<u16>",
            ),
            (
                // Delete element making vec<u16> empty
                r#"Test {
                    vec<u16>,        # vector of u16s
                    bytes<2>         # trailing bytes
                }"#,
                vec![
                    0x01, // length = 1
                    0x42, 0x42, // first u16
                    0xAA, 0xBB, // trailing bytes
                ],
                SampledMutation {
                    mutation: Mutation::Delete,
                    path: FieldPath::new(vec![0]),
                    additional_path: None,
                },
                vec![
                    0x00, // length now 0
                    0xAA, 0xBB, // original trailing bytes
                ],
                "delete element making vec<u16> empty",
            ),
            (
                // Delete element from vec<u16>
                r#"Test {
                    vec<u16>,        # vector of u16s
                    bytes<2>         # trailing bytes
                }"#,
                vec![
                    0x02, // length = 2
                    0x34, 0x12, // first u16 (LE)
                    0x78, 0x56, // second u16 (LE)
                    0xAA, 0xBB, // trailing bytes
                ],
                SampledMutation {
                    mutation: Mutation::Delete,
                    path: FieldPath::new(vec![0]),
                    additional_path: None,
                },
                vec![
                    0x01, // length now 1
                    0x34, 0x12, // only first u16 remains
                    0xAA, 0xBB, // original trailing bytes
                ],
                "delete element from vec<u16>",
            ),
            (
                // Add element to nested vec<vec<u16>>
                r#"Test {
                    vec<vec<u16>>,   # vector of vectors of u16s
                    bytes<2>         # trailing bytes
                }"#,
                vec![
                    0x02, // outer length = 2
                    0x01, // first inner length = 1
                    0x34, 0x12, // first inner vector's u16
                    0x02, // second inner length = 2
                    0x56, 0x34, // second inner vector's first u16
                    0x78, 0x56, // second inner vector's second u16
                    0xAA, 0xBB, // trailing bytes
                ],
                SampledMutation {
                    mutation: Mutation::Add,
                    path: FieldPath::new(vec![0, 1]), // Add to second inner vector
                    additional_path: None,
                },
                vec![
                    0x02, // outer length still 2
                    0x01, // first inner length still 1
                    0x34, 0x12, // first inner vector's u16
                    0x03, // second inner length now 3
                    0x56, 0x34, // second inner vector's first u16
                    0x78, 0x56, // second inner vector's second u16
                    0x00, 0x00, // new u16
                    0xAA, 0xBB, // original trailing bytes
                ],
                "add element to nested vec",
            ),
            (
                // Delete element from nested vec<vec<u16>>
                r#"Test {
                    vec<vec<u16>>,   # vector of vectors of u16s
                    bytes<2>         # trailing bytes
                }"#,
                vec![
                    0x02, // outer length = 2
                    0x01, // first inner length = 1
                    0x34, 0x12, // first inner vector's u16
                    0x02, // second inner length = 2
                    0x56, 0x34, // second inner vector's first u16
                    0x78, 0x56, // second inner vector's second u16
                    0xAA, 0xBB, // trailing bytes
                ],
                SampledMutation {
                    mutation: Mutation::Delete,
                    path: FieldPath::new(vec![0, 1]), // Delete from second inner vector
                    additional_path: None,
                },
                vec![
                    0x02, // outer length still 2
                    0x01, // first inner length still 1
                    0x34, 0x12, // first inner vector's u16
                    0x01, // second inner length now 1
                    0x56, 0x34, // only first u16 remains
                    0xAA, 0xBB, // original trailing bytes
                ],
                "delete element from nested vec",
            ),
            (
                // Delete element from nested vec<vec<u16>>
                r#"Test {
                    vec<vec<u16>>,   # vector of vectors of u16s
                    bytes<2>         # trailing bytes
                }"#,
                vec![
                    0x02, // outer length = 2
                    0x01, // first inner length = 1
                    0x34, 0x12, // content
                    0x01, // second inner length = 1
                    0x56, 0x34, // content
                    0xFF, 0xFF, // trailing bytes
                ],
                SampledMutation {
                    mutation: Mutation::Delete,
                    path: FieldPath::new(vec![0]), // Delete from second inner vector
                    additional_path: None,
                },
                vec![
                    0x01, // outer length = 2
                    0x01, // first inner length = 1
                    0x34, 0x12, // content
                    0xFF, 0xFF, // trailing bytes
                ],
                "delete element from nested vec (2)",
            ),
            (
                r#"Test {
                    vec<u16>,         # byte vector
                    slice<u16, '0'>,  # slice of structs, length in field 0
                }"#,
                vec![2, 255, 255, 0, 0, 255, 255, 0, 0],
                SampledMutation {
                    mutation: Mutation::Delete,
                    path: FieldPath { indices: vec![1] },
                    additional_path: Some(FieldPath { indices: vec![0] }),
                },
                vec![1, 255, 255, 255, 255],
                "stress test edge case #1",
            ),
            (
                r#"Test {
                    vec<u8>,
                    slice<u16, '0'>,
                    vec<vec<u16>>
                }"#,
                vec![
                    2, // vec length
                    0, 0, // vec content
                    0, 0, 0, 0, // slice content
                    3, // outer vec length
                    1, 0, 0, // first vec 1 elem
                    1, 0, 0, // second vec 1 elem
                    0, // third vec 0 elems
                ],
                SampledMutation {
                    mutation: Mutation::Delete,
                    path: FieldPath { indices: vec![2] },
                    additional_path: None,
                },
                vec![
                    2, // vec length
                    0, 0, // vec content
                    0, 0, 0, 0, // slice content
                    2, // outer vec length (one deleted)
                    1, 0, 0, // first vec 1 elem
                    1, 0, 0, // second vec 1 elem
                ],
                "stress test edge case #2",
            ),
            (
                // Copy u16 value
                r#"Test {
                    u16,            # first u16
                    u16,            # second u16 (target for copy)
                    bytes<2>        # trailing bytes
                }"#,
                vec![
                    0x34, 0x12, // first u16 (LE)
                    0x78, 0x56, // second u16 (LE)
                    0xAA, 0xBB, // trailing bytes
                ],
                SampledMutation {
                    mutation: Mutation::Copy(Some(vec![0x34, 0x12])), // Copy first u16's value
                    path: FieldPath::new(vec![1]),                    // Target second u16
                    additional_path: None,
                },
                vec![
                    0x34, 0x12, // original first u16
                    0x34, 0x12, // copied value
                    0xAA, 0xBB, // original trailing bytes
                ],
                "copy u16 value",
            ),
            (
                // Clone element into vec<u16>
                r#"Test {
                    vec<u16>,        # vector of u16s
                    u16,             # source u16 to clone
                    bytes<2>         # trailing bytes
                }"#,
                vec![
                    0x02, // length = 2
                    0x34, 0x12, // first u16 (LE)
                    0x78, 0x56, // second u16 (LE)
                    0x99, 0x88, // source u16
                    0xAA, 0xBB, // trailing bytes
                ],
                SampledMutation {
                    mutation: Mutation::Clone(Some(vec![0x99, 0x88])), // Clone source u16's value
                    path: FieldPath::new(vec![0]),                     // Target vector
                    additional_path: None,
                },
                vec![
                    0x03, // length now 3
                    0x34, 0x12, // original first u16
                    0x78, 0x56, // original second u16
                    0x99, 0x88, // cloned value
                    0x99, 0x88, // original source u16
                    0xAA, 0xBB, // original trailing bytes
                ],
                "clone u16 into vector",
            ),
            (
                // Copy slice element
                r#"Test {
                    u8,                 # length field
                    slice<u16, '0'>,    # slice of u16s
                    bytes<2>            # trailing bytes
                }"#,
                vec![
                    0x02, // length = 2
                    0x34, 0x12, // first u16 (LE)
                    0x78, 0x56, // second u16 (LE)
                    0xAA, 0xBB, // trailing bytes
                ],
                SampledMutation {
                    mutation: Mutation::Copy(Some(vec![0x34, 0x12])), // Copy first u16's value
                    path: FieldPath::new(vec![1, 1]),                 // Target second slice element
                    additional_path: None,
                },
                vec![
                    0x02, // length still 2
                    0x34, 0x12, // original first u16
                    0x34, 0x12, // copied value
                    0xAA, 0xBB, // original trailing bytes
                ],
                "copy slice element",
            ),
            (
                // Clone element into nested vec
                r#"Test {
                    vec<vec<u16>>,   # vector of vectors of u16s
                    u16,             # source u16 to clone
                    bytes<2>         # trailing bytes
                }"#,
                vec![
                    0x02, // outer length = 2
                    0x01, // first inner length = 1
                    0x34, 0x12, // first inner vector's u16
                    0x02, // second inner length = 2
                    0x56, 0x34, // second inner vector's first u16
                    0x78, 0x56, // second inner vector's second u16
                    0x99, 0x88, // source u16
                    0xAA, 0xBB, // trailing bytes
                ],
                SampledMutation {
                    mutation: Mutation::Clone(Some(vec![0x99, 0x88])), // Clone source u16's value
                    path: FieldPath::new(vec![0, 1]),                  // Target second inner vector
                    additional_path: None,
                },
                vec![
                    0x02, // outer length still 2
                    0x01, // first inner length still 1
                    0x34, 0x12, // first inner vector's u16
                    0x03, // second inner length now 3
                    0x56, 0x34, // second inner vector's first u16
                    0x78, 0x56, // second inner vector's second u16
                    0x99, 0x88, // cloned value
                    0x99, 0x88, // original source u16
                    0xAA, 0xBB, // original trailing bytes
                ],
                "clone into nested vector",
            ),
            (
                // Copy struct value
                r#"Inner { u16, bool }
                   Test {
                       Inner,           # first struct
                       Inner,           # second struct (target for copy)
                       bytes<2>         # trailing bytes
                   }"#,
                vec![
                    0x34, 0x12, // first Inner.u16
                    0x01, // first Inner.bool
                    0x78, 0x56, // second Inner.u16
                    0x00, // second Inner.bool
                    0xAA, 0xBB, // trailing bytes
                ],
                SampledMutation {
                    mutation: Mutation::Copy(Some(vec![0x34, 0x12, 0x01])), // Copy first Inner's value
                    path: FieldPath::new(vec![1]),                          // Target second Inner
                    additional_path: None,
                },
                vec![
                    0x34, 0x12, // original first Inner.u16
                    0x01, // original first Inner.bool
                    0x34, 0x12, // copied Inner.u16
                    0x01, // copied Inner.bool
                    0xAA, 0xBB, // original trailing bytes
                ],
                "copy struct value",
            ),
            (
                // Clone struct into vector
                r#"Inner { u16, bool }
                   Test {
                       vec<Inner>,      # vector of Inner structs
                       Inner,           # source Inner to clone
                       bytes<2>         # trailing bytes
                   }"#,
                vec![
                    0x02, // length = 2
                    0x34, 0x12, // first Inner.u16
                    0x01, // first Inner.bool
                    0x78, 0x56, // second Inner.u16
                    0x00, // second Inner.bool
                    0x99, 0x88, // source Inner.u16
                    0x01, // source Inner.bool
                    0xAA, 0xBB, // trailing bytes
                ],
                SampledMutation {
                    mutation: Mutation::Clone(Some(vec![0x99, 0x88, 0x01])), // Clone source Inner's value
                    path: FieldPath::new(vec![0]),                           // Target vector
                    additional_path: None,
                },
                vec![
                    0x03, // length now 3
                    0x34, 0x12, // original first Inner.u16
                    0x01, // original first Inner.bool
                    0x78, 0x56, // original second Inner.u16
                    0x00, // original second Inner.bool
                    0x99, 0x88, // cloned Inner.u16
                    0x01, // cloned Inner.bool
                    0x99, 0x88, // original source Inner.u16
                    0x01, // original source Inner.bool
                    0xAA, 0xBB, // original trailing bytes
                ],
                "clone struct into vector",
            ),
            (
                // Clone into slice
                r#"Inner { u16, bool }
                   Test {
                       u8,              # length field
                       slice<Inner, '0'>, # slice of Inner structs
                       Inner,           # source Inner
                       bytes<2>         # trailing bytes
                   }"#,
                vec![
                    0x02, // length = 2
                    0x34, 0x12, // first Inner.u16
                    0x01, // first Inner.bool
                    0x78, 0x56, // second Inner.u16
                    0x00, // second Inner.bool
                    0x99, 0x88, // source Inner.u16
                    0x01, // source Inner.bool
                    0xAA, 0xBB, // trailing bytes
                ],
                SampledMutation {
                    mutation: Mutation::Clone(Some(vec![0x99, 0x88, 0x01])), // Clone source Inner's value
                    path: FieldPath::new(vec![1]),                           // Target slice
                    additional_path: Some(FieldPath::new(vec![0])),          // Length field path
                },
                vec![
                    0x03, // length now 3
                    0x34, 0x12, // original first Inner.u16
                    0x01, // original first Inner.bool
                    0x78, 0x56, // original second Inner.u16
                    0x00, // original second Inner.bool
                    0x99, 0x88, // cloned Inner.u16
                    0x01, // cloned Inner.bool
                    0x99, 0x88, // original source Inner.u16
                    0x01, // original source Inner.bool
                    0xAA, 0xBB, // original trailing bytes
                ],
                "clone into slice",
            ),
            (
                // Copy vector of u16s
                r#"Test {
                    vec<u16>,        # first vector
                    vec<u16>,        # second vector (target for copy)
                    bytes<2>         # trailing bytes
                }"#,
                vec![
                    0x02, // first vec length = 2
                    0x34, 0x12, // first vec u16 #1
                    0x78, 0x56, // first vec u16 #2
                    0x01, // second vec length = 1
                    0x99, 0x88, // second vec u16 #1
                    0xAA, 0xBB, // trailing bytes
                ],
                SampledMutation {
                    mutation: Mutation::Copy(Some(vec![0x02, 0x34, 0x12, 0x78, 0x56])), // Copy first vector's value
                    path: FieldPath::new(vec![1]), // Target second vector
                    additional_path: None,
                },
                vec![
                    0x02, // first vec length = 2
                    0x34, 0x12, // first vec u16 #1
                    0x78, 0x56, // first vec u16 #2
                    0x02, // second vec length now 2
                    0x34, 0x12, // copied u16 #1
                    0x78, 0x56, // copied u16 #2
                    0xAA, 0xBB, // trailing bytes
                ],
                "copy vector of u16s",
            ),
            (
                // Clone vector into another vector
                r#"Test {
                    vec<vec<u16>>,   # outer vector
                    vec<u16>,        # source vector
                    bytes<2>         # trailing bytes
                }"#,
                vec![
                    0x02, // outer vec length = 2
                    0x01, // first inner vec length = 1
                    0x34, 0x12, // first inner vec u16
                    0x01, // second inner vec length = 1
                    0x56, 0x34, // second inner vec u16
                    0x02, // source vec length = 2
                    0x99, 0x88, // source vec u16 #1
                    0x77, 0x66, // source vec u16 #2
                    0xAA, 0xBB, // trailing bytes
                ],
                SampledMutation {
                    mutation: Mutation::Clone(Some(vec![0x02, 0x99, 0x88, 0x77, 0x66])), // Clone source vector
                    path: FieldPath::new(vec![0]), // Target outer vector
                    additional_path: None,
                },
                vec![
                    0x03, // outer vec length now 3
                    0x01, // first inner vec length = 1
                    0x34, 0x12, // first inner vec u16
                    0x01, // second inner vec length = 1
                    0x56, 0x34, // second inner vec u16
                    0x02, // new inner vec length = 2
                    0x99, 0x88, // cloned u16 #1
                    0x77, 0x66, // cloned u16 #2
                    0x02, // source vec length = 2
                    0x99, 0x88, // source vec u16 #1
                    0x77, 0x66, // source vec u16 #2
                    0xAA, 0xBB, // trailing bytes
                ],
                "clone vector into vector",
            ),
            (
                // Copy slice of u8s
                r#"Test {
                    u8,                # first length field
                    slice<u8, '0'>,    # first slice
                    u8,                # second length field
                    slice<u8, '2'>,    # second slice (target for copy)
                    bytes<2>           # trailing bytes
                }"#,
                vec![
                    0x02, // first slice length = 2
                    0x12, 0x34, // first slice bytes
                    0x01, // second slice length = 1
                    0x88, // second slice byte
                    0xAA, 0xBB, // trailing bytes
                ],
                SampledMutation {
                    mutation: Mutation::Copy(Some(vec![0x12, 0x34])), // Copy first slice's value
                    path: FieldPath::new(vec![3]),                    // Target second slice
                    additional_path: Some(FieldPath::new(vec![2])),   // Second length field path
                },
                vec![
                    0x02, // first slice length = 2
                    0x12, 0x34, // first slice bytes
                    0x02, // second slice length now 2
                    0x12, 0x34, // copied bytes
                    0xAA, 0xBB, // trailing bytes
                ],
                "copy slice of u8s",
            ),
        ];

        // Run all test cases
        for (descriptor_str, initial_data, mutation, expected_result, test_name) in test_cases {
            let mut parser = DescriptorParser::new();
            let descriptor = parse_descriptor_with_parser(descriptor_str, &mut parser).unwrap();
            let obj_parser = ObjectParser::new(descriptor, &parser);

            // Parse the initial data
            let values = obj_parser.parse(&initial_data).expect(&format!(
                "Failed to parse initial data for test '{}'",
                test_name
            ));

            // Create the mutator with our test byte array mutator
            let mutator = StdSerializedValueMutator {
                byte_array_mutator: TestByteArrayMutator,
            };

            // Apply the mutation
            let performed_mutations = mutate(&values, mutation, &mutator, &parser).unwrap();
            let final_bytes =
                finalize_mutations(&initial_data, &values, performed_mutations).unwrap();

            assert_eq!(
                final_bytes, expected_result,
                "Test '{}' failed:\nGot     {:02X?}\nExpected {:02X?}",
                test_name, final_bytes, expected_result
            );
        }

        Ok(())
    }

    #[test]
    fn test_mutation_comparison() {
        // Create test mutations with different paths and types
        let mut mutations = vec![
            PerformedMutation {
                mutation: Mutation::Add,
                path: FieldPath::new(vec![1, 2]),
                mutated_bytes: vec![],
            },
            PerformedMutation {
                mutation: Mutation::Mutate,
                path: FieldPath::new(vec![1, 2]),
                mutated_bytes: vec![],
            },
            PerformedMutation {
                mutation: Mutation::Delete,
                path: FieldPath::new(vec![1, 2]),
                mutated_bytes: vec![],
            },
            PerformedMutation {
                mutation: Mutation::Mutate,
                path: FieldPath::new(vec![0, 1]),
                mutated_bytes: vec![],
            },
            PerformedMutation {
                mutation: Mutation::Add,
                path: FieldPath::new(vec![2, 3]),
                mutated_bytes: vec![],
            },
        ];

        // Sort using our comparison function
        mutations.sort_by(compare_mutations);

        // Verify the order:
        // 1. Path [0,1] Mutate (earliest path)
        // 2. Path [1,2] Mutate (Mutate comes before Add/Delete for same path)
        // 3. Path [1,2] Add/Delete (same path, order doesn't matter between Add/Delete)
        // 4. Path [1,2] Add/Delete
        // 5. Path [2,3] Add (latest path)

        assert_eq!(mutations[0].path.indices, vec![0, 1]);
        assert!(matches!(mutations[0].mutation, Mutation::Mutate));

        assert_eq!(mutations[1].path.indices, vec![1, 2]);
        assert!(matches!(mutations[1].mutation, Mutation::Mutate));

        assert_eq!(mutations[2].path.indices, vec![1, 2]);
        assert!(!matches!(mutations[2].mutation, Mutation::Mutate));

        assert_eq!(mutations[3].path.indices, vec![1, 2]);
        assert!(!matches!(mutations[3].mutation, Mutation::Mutate));

        assert_eq!(mutations[4].path.indices, vec![2, 3]);
    }

    #[test]
    fn test_stress_mutator() -> Result<(), String> {
        // Define a complex descriptor with nested structures and various field types
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
                bytes<4>,                   # Fixed bytes
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

        // Create mutator instances
        let mutator = Mutator::new(descriptor.clone(), &parser);
        let value_mutator = StdSerializedValueMutator {
            byte_array_mutator: TestByteArrayMutator,
        };

        // Generate two initial blobs for cross-over testing
        let mut current_blob = value_mutator
            .generate(&FieldType::Struct("Test".to_string()), &parser)
            .unwrap();
        let mut donor_blob = value_mutator
            .generate(&FieldType::Struct("Test".to_string()), &parser)
            .unwrap();

        // Track mutations for verification
        let mut size_changes = Vec::new();
        let mut crossover_size_changes = Vec::new();

        // Perform multiple mutations, alternating between regular mutations and cross-over
        for seed in 0..1000 {
            if seed % 2 == 0 {
                // Regular mutation
                println!("Regular mutation on {:?}", current_blob);
                match mutator
                    .mutate::<ChaoSampler<_>, StdSerializedValueMutator<TestByteArrayMutator>>(
                        &current_blob,
                        seed,
                    ) {
                    Ok(mutated_blob) => {
                        size_changes.push(mutated_blob.len() as i64 - current_blob.len() as i64);

                        // Verify the mutated blob can still be parsed
                        let obj_parser = ObjectParser::new(descriptor.clone(), &parser);
                        if let Err(e) = obj_parser.parse(&mutated_blob) {
                            eprintln!("Mutation {} failed: {:#?}: {:?}", seed, e, mutated_blob);
                            assert!(false);
                        }

                        current_blob = mutated_blob;
                    }
                    Err(e) => {
                        println!("Mutation {} failed: {}", seed, e);
                    }
                }
            } else {
                // Cross-over mutation
                println!("Cross-over between {:?} and {:?}", current_blob, donor_blob);
                match mutator.cross_over::<ChaoSampler<_>, ChaoSampler<_>, StdSerializedValueMutator<TestByteArrayMutator>>(
                    &current_blob,
                    &donor_blob,
                    seed,
                ) {
                    Ok(crossed_blob) => {
                        crossover_size_changes.push(crossed_blob.len() as i64 - current_blob.len() as i64);

                        // Verify the crossed blob can still be parsed
                        let obj_parser = ObjectParser::new(descriptor.clone(), &parser);
                        if let Err(e) = obj_parser.parse(&crossed_blob) {
                            eprintln!("Cross-over {} failed: {:#?}: {:?}", seed, e, crossed_blob);
                            assert!(false);
                        }

                        // Alternate which blob gets updated
                        if seed % 4 == 1 {
                            current_blob = crossed_blob;
                        } else {
                            donor_blob = crossed_blob;
                        }
                    }
                    Err(e) => {
                        println!("Cross-over {} failed: {}", seed, e);
                    }
                }
            }
        }

        // Print and verify test results
        println!("Regular mutation size changes: {:?}", size_changes);
        println!("Cross-over size changes: {:?}", crossover_size_changes);

        // Calculate size change statistics for regular mutations
        let max_growth = size_changes.iter().max().unwrap_or(&0);
        let max_shrink = size_changes.iter().min().unwrap_or(&0);
        println!("Regular mutations - Maximum size growth: {}", max_growth);
        println!("Regular mutations - Maximum size shrink: {}", max_shrink);

        // Calculate size change statistics for cross-over mutations
        let crossover_max_growth = crossover_size_changes.iter().max().unwrap_or(&0);
        let crossover_max_shrink = crossover_size_changes.iter().min().unwrap_or(&0);
        println!(
            "Cross-over mutations - Maximum size growth: {}",
            crossover_max_growth
        );
        println!(
            "Cross-over mutations - Maximum size shrink: {}",
            crossover_max_shrink
        );

        // Verify both final blobs can still be parsed
        let obj_parser = ObjectParser::new(descriptor, &parser);
        assert!(
            obj_parser.parse(&current_blob).is_ok(),
            "Final current blob is not parseable"
        );
        assert!(
            obj_parser.parse(&donor_blob).is_ok(),
            "Final donor blob is not parseable"
        );

        Ok(())
    }

    #[test]
    fn test_sample_data_sources() -> Result<(), String> {
        let mut parser = DescriptorParser::new();
        parser
            .parse_file(
                r#"
                Inner { u16, bool }
                Test {
                    u16,
                    vec<Inner>,
                    slice<u16, '0'>,
                    bool
                }
                "#,
            )
            .unwrap();

        let descriptor = parser.get_descriptor("Test").unwrap().clone();
        let mutator = Mutator::new(descriptor.clone(), &parser);

        // Create some test data
        let data = vec![
            0x02, 0x00, // first u16
            0x01, // vec length (1)
            0x56, 0x78, // Inner.u16
            0x01, // Inner.bool
            0x90, 0x12, // slice u16
            0x34, 0x56, // slice u16
            0x01, // final bool
        ];

        let obj_parser = ObjectParser::new(descriptor, &parser);
        let values = obj_parser.parse(&data).unwrap();

        // Sample u16 fields
        let mut sampler = TestSampler::new(0);
        mutator.sample_data_sources(&values, &FieldType::Int(IntType::U16), &mut sampler, vec![]);

        let samples = sampler.get_samples();

        // Should find 4 u16 fields:
        // - The top-level u16
        // - The u16 in the Inner struct
        // - Two u16s in the slice
        assert_eq!(samples.len(), 4);

        // Verify the paths
        let expected_paths = vec![
            vec![0],       // Top-level u16
            vec![1, 0, 0], // Inner struct's u16
            vec![2, 0],    // First slice u16
            vec![2, 1],    // Second slice u16
        ];

        for path in expected_paths {
            assert!(
                samples.iter().any(|s| s.indices == path),
                "Missing path: {:?}",
                path
            );
        }

        Ok(())
    }
}
