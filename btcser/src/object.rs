use crate::parser::{Descriptor, DescriptorParser, FieldPath, FieldType, IntType};

#[derive(Debug)]
pub struct SerializedValue<'a> {
    bytes: &'a [u8],
    field_type: FieldType,
    nested_values: Vec<SerializedValue<'a>>,
    is_constant: bool,
    length_field_for: Vec<u64>,
}

impl<'a> SerializedValue<'a> {
    pub fn new(bytes: &'a [u8], field_type: FieldType) -> Self {
        Self {
            bytes,
            field_type,
            nested_values: Vec::new(),
            is_constant: false,
            length_field_for: Vec::new(),
        }
    }

    pub fn with_nested(
        bytes: &'a [u8],
        field_type: FieldType,
        nested: Vec<SerializedValue<'a>>,
    ) -> Self {
        Self {
            bytes,
            field_type,
            nested_values: nested,
            is_constant: false,
            length_field_for: Vec::new(),
        }
    }

    pub fn bytes(&self) -> &'a [u8] {
        self.bytes
    }

    pub fn field_type(&self) -> &FieldType {
        &self.field_type
    }

    pub fn nested_values(&self) -> &[SerializedValue<'a>] {
        self.nested_values.as_slice()
    }

    pub fn is_constant(&self) -> bool {
        self.is_constant
    }

    pub fn length_field_for(&self) -> &[u64] {
        self.length_field_for.as_slice()
    }
}

pub fn find_value_in_object<'a>(
    root: &'a [SerializedValue<'a>],
    path: &FieldPath,
) -> Option<&'a SerializedValue<'a>> {
    let mut current = root;
    let mut value = None;

    for (i, &index) in path.indices.iter().enumerate() {
        if index >= current.len() {
            return None;
        }

        if i == path.indices.len() - 1 {
            value = Some(&current[index]);
        } else {
            current = &current[index].nested_values;
        }
    }

    value
}

struct ParseState {
    field_length_values: Vec<Option<u64>>,
}

impl ParseState {
    pub fn new() -> Self {
        Self {
            field_length_values: Vec::new(),
        }
    }

    fn store_field_value(&mut self, value: Option<u64>) {
        self.field_length_values.push(value);
    }

    fn get_field_value(&self, index: usize) -> Option<u64> {
        self.field_length_values.get(index).and_then(|v| *v)
    }

    fn merge(&mut self, other: &Self) {
        self.field_length_values
            .extend(other.field_length_values.clone());
    }
}

pub struct ObjectParser<'p> {
    descriptor: Descriptor,
    parser: &'p DescriptorParser,
}

impl<'p> ObjectParser<'p> {
    pub fn new(descriptor: Descriptor, parser: &'p DescriptorParser) -> Self {
        Self { descriptor, parser }
    }

    pub fn parse<'a>(&self, data: &'a [u8]) -> Result<Vec<SerializedValue<'a>>, String> {
        let mut state = ParseState::new();
        self.parse_with_state(data, &mut state)
    }

    fn parse_with_state<'a>(
        &self,
        data: &'a [u8],
        state: &mut ParseState,
    ) -> Result<Vec<SerializedValue<'a>>, String> {
        let mut values = Vec::new();
        let mut position = 0;

        for field in &self.descriptor.fields {
            let (mut value, consumed) = self.parse_field(
                &field.field_type,
                &data[position..],
                &field.constant_value,
                state,
            )?;
            value.is_constant = field.constant_value.is_some();
            value.length_field_for = field.length_field_for.clone();
            values.push(value);
            position += consumed;
        }

        if position != data.len() {
            return Err("Extra data after message".to_string());
        }

        Ok(values)
    }

    fn parse_field<'a>(
        &self,
        field_type: &FieldType,
        data: &'a [u8],
        constant: &Option<Vec<u8>>,
        state: &mut ParseState,
    ) -> Result<(SerializedValue<'a>, usize), String> {
        let mut field_length_value = None;
        let res = match &field_type {
            FieldType::Bool => {
                if data.is_empty() {
                    return Err("Unexpected end of input: need 1 byte for bool".to_string());
                }
                if data[0] == 0 {
                    state.store_field_value(Some(0));
                } else {
                    state.store_field_value(Some(1));
                }
                self.parse_bool(data)
            }
            FieldType::Int(int_type) => match int_type {
                IntType::U256 | IntType::I256 | IntType::U256BE | IntType::I256BE => {
                    if data.len() < 32 {
                        return Err("Unexpected end of input: need 32 bytes for 256-bit integer"
                            .to_string());
                    }
                    Ok((SerializedValue::new(&data[..32], field_type.clone()), 32))
                }
                _ => {
                    let (value, size) = self
                        .parse_int(int_type, data)
                        .map_err(|e| format!("Failed parsing Int({:?}): {}", int_type, e))?;
                    field_length_value = Some(value);
                    Ok((
                        SerializedValue::new(&data[..size], field_type.clone()),
                        size,
                    ))
                }
            },
            FieldType::Bytes(size) => {
                if data.len() < *size {
                    return Err(format!(
                        "Unexpected end of input: need {} bytes for fixed-size bytes, got {}",
                        size,
                        data.len()
                    ));
                }
                field_length_value = Some(*size as u64);
                self.parse_bytes(*size, data)
            }
            FieldType::Vec(inner_type) => {
                let (length, length_size, _canonical) = decode_compact_size(data)
                    .map_err(|e| format!("Failed parsing Vec length: {}", e))?;
                if data.len() < length_size {
                    return Err(format!(
                        "Unexpected end of input: need {} bytes for Vec length, got {}",
                        length_size,
                        data.len()
                    ));
                }
                field_length_value = Some(length);
                let mut vec_state = ParseState::new();
                self.parse_vec(inner_type, data, constant, &mut vec_state)
                    .map_err(|e| format!("Failed parsing Vec contents: {}", e))
            }
            FieldType::Slice(inner_type, length_field) => {
                let length = state
                    .get_field_value(*length_field as usize)
                    .ok_or_else(|| {
                        format!(
                            "Length field '{}' not found in state for Slice",
                            length_field
                        )
                    })?;
                if data.len() < (length as usize) {
                    return Err(format!(
                        "Unexpected end of input: need {} bytes for Slice (length from field {}), got {}",
                        length,
                        length_field,
                        data.len()
                    ));
                }
                self.parse_slice(inner_type, *length_field, data, constant, state)
                    .map_err(|e| format!("Failed parsing Slice contents: {}", e))
            }
            FieldType::Struct(name) => {
                let mut struct_state = ParseState::new();
                self.parse_struct(name, data, &mut struct_state)
                    .map_err(|e| format!("Failed parsing Struct '{}': {}", name, e))
            }
        };

        state.store_field_value(field_length_value);

        let (value, size) = res?;

        if let Some(ref constant) = constant {
            if value.bytes != constant.as_slice() {
                return Err(format!(
                    "Constant field mismatch: expected {:?}, got {:?}",
                    constant, value.bytes
                ));
            }
        }

        Ok((value, size))
    }

    fn parse_bool<'a>(&self, data: &'a [u8]) -> Result<(SerializedValue<'a>, usize), String> {
        if data.is_empty() {
            return Err("Unexpected end of input".to_string());
        }
        Ok((SerializedValue::new(&data[..1], FieldType::Bool), 1))
    }

    fn parse_int<'a>(&self, int_type: &IntType, data: &'a [u8]) -> Result<(u64, usize), String> {
        match int_type {
            IntType::CompactSize(require_canonical) => {
                let (value, size, is_canonical) = decode_compact_size(data)?;
                if *require_canonical && !is_canonical {
                    return Err("non-canonical CompactSize".to_string());
                }
                Ok((value, size))
            }
            int_type @ (IntType::U8 | IntType::I8) => self.parse_fixed_int(data, 1, int_type),
            int_type @ (IntType::U16 | IntType::U16BE | IntType::I16 | IntType::I16BE) => {
                self.parse_fixed_int(data, 2, int_type)
            }
            int_type @ (IntType::U32 | IntType::U32BE | IntType::I32 | IntType::I32BE) => {
                self.parse_fixed_int(data, 4, int_type)
            }
            int_type @ (IntType::U64 | IntType::U64BE | IntType::I64 | IntType::I64BE) => {
                self.parse_fixed_int(data, 8, int_type)
            }
            int_type @ (IntType::U256 | IntType::U256BE | IntType::I256 | IntType::I256BE) => {
                self.parse_fixed_int(data, 32, int_type)
            }
            IntType::VarInt | IntType::VarIntNonNegative => {
                let (value, size) = self.parse_varint(data)?;
                if matches!(int_type, IntType::VarIntNonNegative) && (value & 1) != 0 {
                    return Err("Negative value not allowed for varint+".to_string());
                }
                Ok((value, size))
            }
        }
    }

    fn parse_varint(&self, data: &[u8]) -> Result<(u64, usize), String> {
        decode_varint(data)
    }

    fn parse_fixed_int<'a>(
        &self,
        data: &'a [u8],
        size: usize,
        int_type: &IntType,
    ) -> Result<(u64, usize), String> {
        if data.len() < size {
            return Err("Unexpected end of input".to_string());
        }

        let value = match int_type {
            IntType::U8 => data[0] as u64,
            IntType::I8 => data[0] as i8 as u64,
            IntType::U16 => u16::from_le_bytes([data[0], data[1]]) as u64,
            IntType::U16BE => u16::from_be_bytes([data[0], data[1]]) as u64,
            IntType::I16 => i16::from_le_bytes([data[0], data[1]]) as u64,
            IntType::I16BE => i16::from_be_bytes([data[0], data[1]]) as u64,
            IntType::U32 => u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as u64,
            IntType::U32BE => u32::from_be_bytes([data[0], data[1], data[2], data[3]]) as u64,
            IntType::I32 => i32::from_le_bytes([data[0], data[1], data[2], data[3]]) as u64,
            IntType::I32BE => i32::from_be_bytes([data[0], data[1], data[2], data[3]]) as u64,
            IntType::U64 => u64::from_le_bytes([
                data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
            ]),
            IntType::U64BE => u64::from_be_bytes([
                data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
            ]),
            IntType::I64 => i64::from_le_bytes([
                data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
            ]) as u64,
            IntType::I64BE => i64::from_be_bytes([
                data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
            ]) as u64,
            _ => return Err("Unsupported fixed int type".to_string()),
        };

        Ok((value, size))
    }

    fn parse_bytes<'a>(
        &self,
        size: usize,
        data: &'a [u8],
    ) -> Result<(SerializedValue<'a>, usize), String> {
        if data.len() < size {
            return Err("Unexpected end of input".to_string());
        }

        Ok((
            SerializedValue::new(&data[..size], FieldType::Bytes(size)),
            size,
        ))
    }

    fn parse_vec<'a>(
        &self,
        inner_type: &FieldType,
        data: &'a [u8],
        constant: &Option<Vec<u8>>,
        state: &mut ParseState,
    ) -> Result<(SerializedValue<'a>, usize), String> {
        let (length, length_size, _canonical) =
            decode_compact_size(data).map_err(|e| format!("Failed parsing Vec length: {}", e))?;

        let mut position = length_size;
        let mut nested_values = Vec::new();

        for _ in 0..length {
            let (value, element_size) =
                self.parse_field(inner_type, &data[position..], constant, state)?;
            nested_values.push(value);
            position += element_size;
        }

        Ok((
            SerializedValue::with_nested(
                &data[..position],
                FieldType::Vec(Box::new(inner_type.clone())),
                nested_values,
            ),
            position,
        ))
    }

    fn parse_slice<'a>(
        &self,
        inner_type: &FieldType,
        length_field: u64,
        data: &'a [u8],
        constant: &Option<Vec<u8>>,
        state: &mut ParseState,
    ) -> Result<(SerializedValue<'a>, usize), String> {
        let length = state
            .get_field_value(length_field as usize)
            .ok_or_else(|| format!("Length field {} not found", length_field))?;

        let mut position = 0;
        let mut nested_values = Vec::new();

        let mut state = ParseState::new(); // Nested elements need a new parse state
        for _ in 0..length {
            let (value, element_size) =
                self.parse_field(inner_type, &data[position..], constant, &mut state)?;
            nested_values.push(value);
            position += element_size;
        }

        Ok((
            SerializedValue::with_nested(
                &data[..position],
                FieldType::Slice(Box::new(inner_type.clone()), length_field),
                nested_values,
            ),
            position,
        ))
    }

    fn parse_struct<'a>(
        &self,
        name: &str,
        data: &'a [u8],
        state: &mut ParseState,
    ) -> Result<(SerializedValue<'a>, usize), String> {
        let struct_descriptor = self.get_struct_descriptor(name)?;

        // Setup a temporary state to merge the results of the alternatives (we avoid poluting the
        // main state with failed alternatives)
        let mut tmp_state = ParseState::new();

        // Try the main descriptor first
        let result = self.try_parse_struct_with_descriptor(struct_descriptor, data, &mut tmp_state);
        if result.is_ok() {
            state.merge(&tmp_state);
            return result;
        }
        let mut last_error = result.err();

        // If main descriptor fails, try alternatives
        for alt_descriptor in struct_descriptor.alternatives.iter() {
            match self.try_parse_struct_with_descriptor(alt_descriptor, data, &mut tmp_state) {
                Ok(result) => {
                    state.merge(&tmp_state);
                    return Ok(result);
                }
                Err(e) => last_error = Some(e),
            }
        }

        // If all attempts fail, return the last error
        Err(last_error.unwrap_or_else(|| format!("Failed to parse struct '{}'", name)))
    }

    // Helper method that contains the actual parsing logic
    fn try_parse_struct_with_descriptor<'a>(
        &self,
        descriptor: &Descriptor,
        data: &'a [u8],
        state: &mut ParseState,
    ) -> Result<(SerializedValue<'a>, usize), String> {
        let mut position = 0;
        let mut nested_values = Vec::new();

        for field in descriptor.fields.iter() {
            let (mut value, consumed) = self.parse_field(
                &field.field_type,
                &data[position..],
                &field.constant_value,
                state,
            )?;
            value.is_constant = field.constant_value.is_some();
            value.length_field_for = field.length_field_for.clone();
            nested_values.push(value);
            position += consumed;
        }

        Ok((
            SerializedValue::with_nested(
                &data[..position],
                FieldType::Struct(descriptor.name.clone()),
                nested_values,
            ),
            position,
        ))
    }

    fn get_struct_descriptor(&self, name: &str) -> Result<&Descriptor, String> {
        self.parser
            .get_descriptor(name)
            .ok_or_else(|| format!("Unknown struct type: {}", name))
    }
}

pub fn encode_compact_size(value: u64) -> Vec<u8> {
    if value <= 252 {
        // Single byte encoding
        vec![value as u8]
    } else if value <= 0xffff {
        // 3-byte encoding
        let mut bytes = vec![253];
        bytes.extend_from_slice(&(value as u16).to_le_bytes());
        bytes
    } else if value <= 0xffffffff {
        // 5-byte encoding
        let mut bytes = vec![254];
        bytes.extend_from_slice(&(value as u32).to_le_bytes());
        bytes
    } else {
        // 9-byte encoding
        let mut bytes = vec![255];
        bytes.extend_from_slice(&value.to_le_bytes());
        bytes
    }
}

pub fn decode_compact_size(data: &[u8]) -> Result<(u64, usize, bool), String> {
    if data.is_empty() {
        return Err("Unexpected end of input".to_string());
    }

    let first_byte = data[0];
    match first_byte {
        0..=252 => Ok((first_byte as u64, 1, true)),
        253 => {
            if data.len() < 3 {
                return Err("Unexpected end of input".to_string());
            }
            let value = u16::from_le_bytes([data[1], data[2]]) as u64;
            Ok((value, 3, value >= 253))
        }
        254 => {
            if data.len() < 5 {
                return Err("Unexpected end of input".to_string());
            }
            let value = u32::from_le_bytes([data[1], data[2], data[3], data[4]]) as u64;
            Ok((value, 5, value >= 0x10000))
        }
        255 => {
            if data.len() < 9 {
                return Err("Unexpected end of input".to_string());
            }
            let value = u64::from_le_bytes([
                data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8],
            ]);
            Ok((value, 9, value >= 0x100000000))
        }
    }
}

pub fn encode_varint(mut value: u64) -> Vec<u8> {
    let mut tmp = Vec::with_capacity(10); // (64 bits + 6) / 7 = max 10 bytes
    let mut len = 0;

    loop {
        tmp.push((value & 0x7F) as u8 | if len > 0 { 0x80 } else { 0x00 });
        if value <= 0x7F {
            break;
        }
        value = (value >> 7) - 1;
        len += 1;
    }

    // Write bytes in reverse order
    let mut result = Vec::with_capacity(len + 1);
    for i in (0..=len).rev() {
        result.push(tmp[i]);
    }

    result
}

pub fn decode_varint(data: &[u8]) -> Result<(u64, usize), String> {
    if data.is_empty() {
        return Err("Unexpected end of input".to_string());
    }

    let mut value: u64 = 0;
    let mut size = 0;

    while size < data.len() {
        let byte = data[size];
        size += 1;

        // Check for overflow before shifting
        if value > (u64::MAX >> 7) {
            return Err("VarInt is too large".to_string());
        }

        value = (value << 7) | (byte & 0x7F) as u64;

        if byte & 0x80 != 0 {
            // More bytes follow - add 1 as per spec
            if value == u64::MAX {
                return Err("VarInt is too large".to_string());
            }
            value += 1;
        } else {
            // Last byte - we're done
            return Ok((value, size));
        }
    }

    Err("Truncated VarInt".to_string())
}

pub fn encode_varint_non_negative(value: u64) -> Vec<u8> {
    encode_varint(value << 1)
}

pub fn decode_varint_non_negative(data: &[u8]) -> Result<(u64, usize), String> {
    let (value, size) = decode_varint(data)?;
    if (value & 1) != 0 {
        return Err("Negative value not allowed for varint+".to_string());
    }
    Ok((value >> 1, size))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::DescriptorParser;

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

    #[test]
    fn test_compact_size() {
        // Valid cases
        assert_eq!(decode_compact_size(&[0]).unwrap(), (0, 1, true));
        assert_eq!(decode_compact_size(&[252]).unwrap(), (252, 1, true));
        assert_eq!(
            decode_compact_size(&[253, 0xfd, 0x00]).unwrap(),
            (253, 3, true)
        );
        assert_eq!(
            decode_compact_size(&[254, 0x00, 0x00, 0x01, 0x00]).unwrap(),
            (0x10000, 5, true)
        );

        // Non-canonical cases
        assert_eq!(
            decode_compact_size(&[253, 0xfc, 0x00]).unwrap(),
            (252, 3, false) // 252 encoded with 3 bytes
        );
        assert_eq!(
            decode_compact_size(&[254, 0xff, 0xff, 0x00, 0x00]).unwrap(),
            (0xffff, 5, false) // < 0x10000 encoded with 5 bytes
        );
        assert_eq!(
            decode_compact_size(&[255, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00]).unwrap(),
            (0xffffffff, 9, false) // < 0x100000000 encoded with 9 bytes
        );
    }

    #[test]
    fn test_parse_vec_u8() -> Result<(), String> {
        let mut parser = DescriptorParser::new();
        let descriptor = parse_descriptor_with_parser("Test { vec<u8> }", &mut parser).unwrap();
        let message_obj_parser = ObjectParser::new(descriptor, &parser);

        // Test empty vector
        let data = [0u8]; // CompactSize(0)
        let values = message_obj_parser.parse(&data).unwrap();
        assert_eq!(values.len(), 1);
        assert_eq!(values[0].bytes, &data[..]);
        assert_eq!(values[0].nested_values.len(), 0);

        // Test vector with elements
        let data = [2u8, 0x42, 0x43]; // CompactSize(2), followed by two bytes
        let values = message_obj_parser.parse(&data).unwrap();
        assert_eq!(values.len(), 1);
        assert_eq!(values[0].bytes, &data[..]);
        assert_eq!(values[0].nested_values.len(), 2);
        assert_eq!(values[0].nested_values[0].bytes, &[0x42]);
        assert_eq!(values[0].nested_values[1].bytes, &[0x43]);

        Ok(())
    }

    #[test]
    fn test_parse_vec_u16() -> Result<(), String> {
        let mut parser = DescriptorParser::new();
        let descriptor = parse_descriptor_with_parser("Test { vec<u16> }", &mut parser).unwrap();
        let obj_parser = ObjectParser::new(descriptor, &parser);

        // Test vector with U16 elements
        let data = [2u8, 0x34, 0x12, 0x78, 0x56]; // CompactSize(2), followed by two U16s
        let values = obj_parser.parse(&data).unwrap();
        assert_eq!(values.len(), 1);
        assert_eq!(values[0].bytes, &data[..]);
        assert_eq!(values[0].nested_values.len(), 2);
        assert_eq!(values[0].nested_values[0].bytes, &[0x34, 0x12]);
        assert_eq!(values[0].nested_values[1].bytes, &[0x78, 0x56]);

        Ok(())
    }

    #[test]
    fn test_parse_vec_errors() {
        let mut parser = DescriptorParser::new();
        let descriptor = parse_descriptor_with_parser("Test { vec<u16> }", &mut parser).unwrap();
        let obj_parser = ObjectParser::new(descriptor, &parser);

        // Test truncated length
        assert!(obj_parser.parse(&[]).is_err());

        // Test truncated data
        assert!(obj_parser.parse(&[1u8, 0x34]).is_err()); // Claims 1 U16 but only has 1 byte

        // Test oversized length
        let data = [253u8, 0xFF, 0xFF]; // CompactSize(65535)
        assert!(obj_parser.parse(&data).is_err());
    }

    #[test]
    fn test_nested_vec() -> Result<(), String> {
        let mut parser = DescriptorParser::new();
        let descriptor =
            parse_descriptor_with_parser("Test { vec<vec<u8>> }", &mut parser).unwrap();
        let obj_parser = ObjectParser::new(descriptor, &parser);

        // Test vector of vectors: [[1, 2], [3, 4]]
        let data = [
            2u8, // Outer vector length
            2u8, 1, 2, // First inner vector
            2u8, 3, 4, // Second inner vector
        ];
        let values = obj_parser.parse(&data).unwrap();
        assert_eq!(values.len(), 1);
        assert_eq!(values[0].bytes, &data[..]);
        assert_eq!(values[0].nested_values.len(), 2);

        // Check first inner vector
        assert_eq!(values[0].nested_values[0].bytes, &data[1..4]);
        assert_eq!(values[0].nested_values[0].nested_values.len(), 2);
        assert_eq!(values[0].nested_values[0].nested_values[0].bytes, &[1]);
        assert_eq!(values[0].nested_values[0].nested_values[1].bytes, &[2]);

        // Check second inner vector
        assert_eq!(values[0].nested_values[1].bytes, &data[4..7]);
        assert_eq!(values[0].nested_values[1].nested_values.len(), 2);
        assert_eq!(values[0].nested_values[1].nested_values[0].bytes, &[3]);
        assert_eq!(values[0].nested_values[1].nested_values[1].bytes, &[4]);

        Ok(())
    }

    #[test]
    fn test_parse_nested_struct() -> Result<(), String> {
        let mut parser = DescriptorParser::new();
        parser
            .parse_file(
                "Inner { u8, u16 }
             Outer { Inner, vec<Inner> }",
            )
            .unwrap();

        let descriptor = parser.descriptors.get("Outer").unwrap().clone();
        let obj_parser = ObjectParser::new(descriptor, &parser);

        // Test parsing a single Inner struct followed by a vector of two Inner structs
        let data = [
            0x42, 0x34, 0x12, // First Inner: u8=0x42, u16=0x1234
            2,    // Vector length (2)
            0x43, 0x56, 0x34, // Second Inner: u8=0x43, u16=0x3456
            0x44, 0x78, 0x56, // Third Inner: u8=0x44, u16=0x5678
        ];

        let values = obj_parser.parse(&data).unwrap();
        assert_eq!(values.len(), 2); // One direct Inner, one vec<Inner>

        // Check the first Inner struct
        assert_eq!(values[0].bytes, &data[0..3]);
        assert_eq!(values[0].nested_values.len(), 2);
        assert_eq!(values[0].nested_values[0].bytes, &[0x42]); // u8
        assert_eq!(values[0].nested_values[1].bytes, &[0x34, 0x12]); // u16

        // Check the vector of Inner structs
        assert_eq!(values[1].bytes, &data[3..]);
        assert_eq!(values[1].nested_values.len(), 2);

        // Check first Inner in vector
        assert_eq!(values[1].nested_values[0].bytes, &data[4..7]);
        assert_eq!(values[1].nested_values[0].nested_values[0].bytes, &[0x43]);
        assert_eq!(
            values[1].nested_values[0].nested_values[1].bytes,
            &[0x56, 0x34]
        );

        // Check second Inner in vector
        assert_eq!(values[1].nested_values[1].bytes, &data[7..10]);
        assert_eq!(values[1].nested_values[1].nested_values[0].bytes, &[0x44]);
        assert_eq!(
            values[1].nested_values[1].nested_values[1].bytes,
            &[0x78, 0x56]
        );

        Ok(())
    }

    #[test]
    fn test_parse_deeply_nested_struct() -> Result<(), String> {
        let mut parser = DescriptorParser::new();
        parser
            .parse_file(
                "Leaf { u8 }
             Branch { Leaf, vec<Leaf> }
             Tree { Branch, vec<Branch> }",
            )
            .unwrap();

        let descriptor = parser.descriptors.get("Tree").unwrap().clone();
        let obj_parser = ObjectParser::new(descriptor, &parser);

        let data = [
            0x01, // First Branch's Leaf: u8=0x01
            1, 0x02, // First Branch's vec<Leaf>: [0x02]
            2,    // Vector length (2) for vec<Branch>
            0x03, // Second Branch's Leaf: u8=0x03
            2, 0x04, 0x05, // Second Branch's vec<Leaf>: [0x04, 0x05]
            0x06, // Third Branch's Leaf: u8=0x06
            1, 0x07, // Third Branch's vec<Leaf>: [0x07]
        ];

        let values = obj_parser.parse(&data).unwrap();
        assert_eq!(values.len(), 2); // One direct Branch, one vec<Branch>

        // Check first Branch
        assert_eq!(values[0].nested_values[0].bytes, &[0x01]); // Leaf
        assert_eq!(values[0].nested_values[1].nested_values.len(), 1); // vec<Leaf>
        assert_eq!(values[0].nested_values[1].nested_values[0].bytes, &[0x02]);

        // Check vec<Branch>
        assert_eq!(values[1].nested_values.len(), 2);

        // Check first Branch in vector
        let first_branch = &values[1].nested_values[0];
        assert_eq!(first_branch.nested_values[0].bytes, &[0x03]); // Leaf
        assert_eq!(first_branch.nested_values[1].nested_values.len(), 2); // vec<Leaf>
        assert_eq!(
            first_branch.nested_values[1].nested_values[0].bytes,
            &[0x04]
        );
        assert_eq!(
            first_branch.nested_values[1].nested_values[1].bytes,
            &[0x05]
        );

        // Check second Branch in vector
        let second_branch = &values[1].nested_values[1];
        assert_eq!(second_branch.nested_values[0].bytes, &[0x06]); // Leaf
        assert_eq!(second_branch.nested_values[1].nested_values.len(), 1); // vec<Leaf>
        assert_eq!(
            second_branch.nested_values[1].nested_values[0].bytes,
            &[0x07]
        );

        Ok(())
    }

    #[test]
    fn test_varint() -> Result<(), String> {
        // Test cases based on the C++ reference examples
        let test_cases = vec![
            (0, vec![0x00]),                 // 0
            (1, vec![0x01]),                 // 1
            (127, vec![0x7F]),               // 127
            (128, vec![0x80, 0x00]),         // 128
            (255, vec![0x80, 0x7F]),         // 255
            (256, vec![0x81, 0x00]),         // 256
            (16383, vec![0xFE, 0x7F]),       // 16383
            (16384, vec![0xFF, 0x00]),       // 16384
            (16511, vec![0xFF, 0x7F]),       // 16511
            (65535, vec![0x82, 0xFE, 0x7F]), // 65535
            // 2^32
            (4294967296, vec![0x8E, 0xFE, 0xFE, 0xFF, 0x00]),
        ];

        for (value, expected_encoding) in test_cases {
            // Test encoding
            let encoded = encode_varint(value);
            assert_eq!(encoded, expected_encoding, "Failed encoding {}", value);

            // Test decoding
            let (decoded, size) = decode_varint(&expected_encoding).unwrap();
            assert_eq!(decoded, value, "Failed decoding {:?}", expected_encoding);
            assert_eq!(size, expected_encoding.len());
        }

        // Test error cases
        assert!(decode_varint(&[]).is_err()); // Empty input
        assert!(decode_varint(&[0x80]).is_err()); // Truncated
        assert!(
            decode_varint(&[0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80]).is_err()
        ); // Too long

        Ok(())
    }

    #[test]
    fn test_varint_non_negative() -> Result<(), String> {
        // Test encoding/decoding positive numbers
        let test_cases = vec![
            (0, vec![0x00]), // 0 -> 0
            (1, vec![0x02]), // 1 -> 2
            (2, vec![0x04]), // 2 -> 4
        ];

        for (value, expected_encoding) in test_cases {
            // Test encoding
            let encoded = encode_varint_non_negative(value);
            assert_eq!(encoded, expected_encoding, "Failed encoding {}", value);

            // Test decoding
            let (decoded, size) = decode_varint_non_negative(&expected_encoding).unwrap();
            assert_eq!(decoded, value, "Failed decoding {:?}", expected_encoding);
            assert_eq!(size, expected_encoding.len());
        }

        // Test decoding invalid (negative) numbers
        assert!(decode_varint_non_negative(&[0x01]).is_err()); // -1
        assert!(decode_varint_non_negative(&[0x03]).is_err()); // -2

        Ok(())
    }

    #[test]
    fn test_parse_slice() {
        // Test slice of u8s with length from previous field
        let mut parser = DescriptorParser::new();
        let descriptor =
            parse_descriptor_with_parser("Test { u8, slice<u8, '0'> }", &mut parser).unwrap();
        let obj_parser = ObjectParser::new(descriptor, &parser);

        // Test length 2 slice
        let data = [
            0x02, // First field: length = 2 (stored as u8)
            0x42, 0x43, // Slice contents: [0x42, 0x43]
        ];
        let values = obj_parser.parse(&data).unwrap();
        assert_eq!(values.len(), 2);
        assert_eq!(values[0].bytes, &[0x02]); // Length field
        assert_eq!(values[1].bytes, &[0x42, 0x43]); // Slice contents
        assert_eq!(values[1].nested_values.len(), 2);
        assert_eq!(values[1].nested_values[0].bytes, &[0x42]);
        assert_eq!(values[1].nested_values[1].bytes, &[0x43]);

        // Test empty slice
        let data = [0x00]; // Length = 0, no contents
        let values = obj_parser.parse(&data).unwrap();
        assert_eq!(values.len(), 2);
        assert_eq!(values[0].bytes, &[0x00]); // Length field
        assert_eq!(values[1].bytes, &[]); // Empty slice
        assert_eq!(values[1].nested_values.len(), 0);

        // Test slice of u16s
        let mut parser = DescriptorParser::new();
        let descriptor =
            parse_descriptor_with_parser("Test { u8, slice<u16, '0'> }", &mut parser).unwrap();
        let obj_parser = ObjectParser::new(descriptor, &parser);

        let data = [
            0x02, // Length = 2 (stored as u8)
            0x34, 0x12, 0x78, 0x56, // Two u16s: [0x1234, 0x5678]
        ];
        let values = obj_parser.parse(&data).unwrap();
        assert_eq!(values.len(), 2);
        assert_eq!(values[1].nested_values.len(), 2);
        assert_eq!(values[1].nested_values[0].bytes, &[0x34, 0x12]);
        assert_eq!(values[1].nested_values[1].bytes, &[0x78, 0x56]);

        // Test error cases
        let mut parser = DescriptorParser::new();
        let descriptor =
            parse_descriptor_with_parser("Test { u8, slice<u8, '0'> }", &mut parser).unwrap();
        let obj_parser = ObjectParser::new(descriptor, &parser);

        // Test truncated data
        let data = [0x02, 0x42]; // Claims length 2 but only has 1 byte
        assert!(obj_parser.parse(&data).is_err());
    }

    //#[test]
    //fn test_nested_slice() {
    //    // Test slice of slices
    //    let descriptor = parse_descriptor_with_parser("Test { u8, u8, slice<slice<u8, '0'>, '1'> }").unwrap();
    //    let parser = DescriptorParser::new();
    //    let obj_parser = ObjectParser::new(descriptor, &parser);
    //
    //    let data = [
    //        0x02, // Inner slice length (stored as u8)
    //        0x02, // Outer slice length (stored as u8)
    //        0x11, 0x22, // First inner slice: [0x11, 0x22]
    //        0x33, 0x44, // Second inner slice: [0x33, 0x44]
    //    ];
    //    let values = obj_parser.parse(&data).unwrap();
    //    assert_eq!(values.len(), 3);
    //
    //    // Check outer slice
    //    assert_eq!(values[2].nested_values.len(), 2);
    //
    //    // Check first inner slice
    //    let first_inner = &values[2].nested_values[0];
    //    assert_eq!(first_inner.nested_values.len(), 2);
    //    assert_eq!(first_inner.nested_values[0].bytes, &[0x11]);
    //    assert_eq!(first_inner.nested_values[1].bytes, &[0x22]);
    //
    //    // Check second inner slice
    //    let second_inner = &values[2].nested_values[1];
    //    assert_eq!(second_inner.nested_values.len(), 2);
    //    assert_eq!(second_inner.nested_values[0].bytes, &[0x33]);
    //    assert_eq!(second_inner.nested_values[1].bytes, &[0x44]);
    //}

    #[test]
    fn test_parse_slice_basic() {
        // Start with just a simple slice of u8s

        let mut parser = DescriptorParser::new();
        let descriptor =
            parse_descriptor_with_parser("Test { u8, slice<u8, '0'> }", &mut parser).unwrap();
        let obj_parser = ObjectParser::new(descriptor, &parser);

        // Test with minimal data
        let data = [
            0x01, // Length field (u8) = 1
            0x42, // Single byte of slice content
        ];

        let values = obj_parser.parse(&data).unwrap();
        assert_eq!(values.len(), 2);
        assert_eq!(values[0].bytes, &[0x01]); // Length field
        assert_eq!(values[1].bytes, &[0x42]); // Slice content
        assert_eq!(values[1].nested_values.len(), 1);
        assert_eq!(values[1].nested_values[0].bytes, &[0x42]);
    }

    #[test]
    fn test_parse_bytes_and_slice() -> Result<(), String> {
        let mut parser = DescriptorParser::new();
        let descriptor =
            parse_descriptor_with_parser("Test { bytes<4>, u8, slice<u8, '1'> }", &mut parser)
                .unwrap();
        let obj_parser = ObjectParser::new(descriptor, &parser);

        // Test case with:
        // - 4 bytes fixed-size field [0xAA, 0xBB, 0xCC, 0xDD]
        // - u8 field with value 2 (length for slice)
        // - slice of 2 bytes [0x11, 0x22]
        let data = [
            0xAA, 0xBB, 0xCC, 0xDD, // First field: fixed size bytes
            0x02, // Second field: length = 2 (stored as u8)
            0x11, 0x22, // Third field: slice contents [0x11, 0x22]
        ];

        let values = obj_parser.parse(&data).unwrap();
        assert_eq!(values.len(), 3);

        // Check fixed-size bytes field
        assert_eq!(values[0].bytes, &[0xAA, 0xBB, 0xCC, 0xDD]);
        assert!(matches!(values[0].field_type(), FieldType::Bytes(4)));

        // Check length field (u8)
        assert_eq!(values[1].bytes, &[0x02]);
        assert!(matches!(
            values[1].field_type(),
            FieldType::Int(IntType::U8)
        ));

        // Check slice field
        assert_eq!(values[2].bytes, &[0x11, 0x22]);
        assert!(matches!(values[2].field_type(),
            FieldType::Slice(_, length_field) if *length_field == 1
        ));
        assert_eq!(values[2].nested_values.len(), 2);
        assert_eq!(values[2].nested_values[0].bytes, &[0x11]);
        assert_eq!(values[2].nested_values[1].bytes, &[0x22]);

        // Test error case: not enough bytes for fixed-size field
        let short_data = [0xAA, 0xBB, 0xCC]; // Missing byte for fixed-size field
        assert!(obj_parser.parse(&short_data).is_err());

        // Test error case: not enough bytes for slice
        let incomplete_data = [
            0xAA, 0xBB, 0xCC, 0xDD, // Fixed bytes
            0x02, // Length = 2
            0x11, // Only 1 byte for slice (should be 2)
        ];
        assert!(obj_parser.parse(&incomplete_data).is_err());

        Ok(())
    }

    #[test]
    fn test_slice_in_nested_struct() -> Result<(), String> {
        let mut parser = DescriptorParser::new();
        parser
            .parse_file(
                "Inner { u8, slice<u8, '0'> }
             Outer { u8, Inner }",
            )
            .unwrap();

        let descriptor = parser.descriptors.get("Outer").unwrap().clone();
        let obj_parser = ObjectParser::new(descriptor, &parser);

        // Test data:
        // Outer.u8 = 0xFF
        // Outer.Inner.u8 = 0x02 (length for slice)
        // Outer.Inner.slice = [0x42, 0x43]
        let data = [
            0xFF, // Outer.u8
            0x02, // Inner.u8 (length)
            0x42, 0x43, // Inner.slice contents
        ];

        let values = obj_parser.parse(&data).unwrap();
        assert_eq!(values.len(), 2); // Outer.u8 and Outer.Inner

        // Check Outer.u8
        assert_eq!(values[0].bytes, &[0xFF]);

        // Check Outer.Inner
        let inner = &values[1];
        assert_eq!(inner.nested_values.len(), 2); // Inner.u8 and Inner.slice

        // Check Inner.u8 (length field)
        assert_eq!(inner.nested_values[0].bytes, &[0x02]);

        // Check Inner.slice
        let inner_slice = &inner.nested_values[1];
        assert_eq!(inner_slice.bytes, &[0x42, 0x43]);
        assert_eq!(inner_slice.nested_values.len(), 2);
        assert_eq!(inner_slice.nested_values[0].bytes, &[0x42]);
        assert_eq!(inner_slice.nested_values[1].bytes, &[0x43]);

        // Test error case: truncated slice
        let truncated_data = [
            0xFF, // Outer.u8
            0x02, // Inner.u8 (length)
            0x42, // Only one byte when two were promised
        ];
        assert!(obj_parser.parse(&truncated_data).is_err());

        Ok(())
    }

    #[test]
    fn test_field_path_sorting() {
        // Create various paths to test sorting
        let paths = vec![
            FieldPath::new(vec![1, 2, 3]),
            FieldPath::new(vec![1, 2]),
            FieldPath::new(vec![1, 3]),
            FieldPath::new(vec![1]),
            FieldPath::new(vec![2]),
            FieldPath::new(vec![0, 1]),
        ];

        // Create expected sorted order
        let expected = vec![
            FieldPath::new(vec![0, 1]),
            FieldPath::new(vec![1]),
            FieldPath::new(vec![1, 2]),
            FieldPath::new(vec![1, 2, 3]),
            FieldPath::new(vec![1, 3]),
            FieldPath::new(vec![2]),
        ];

        // Create a clone to sort
        let mut sorted = paths.clone();
        sorted.sort();

        // Verify sorting matches expected order
        assert_eq!(sorted, expected);

        // Test individual comparisons
        assert!(FieldPath::new(vec![0]) < FieldPath::new(vec![1]));
        assert!(FieldPath::new(vec![0, 1]) < FieldPath::new(vec![1]));
        assert!(FieldPath::new(vec![1, 0]) > FieldPath::new(vec![0, 1]));
        assert!(FieldPath::new(vec![1]) < FieldPath::new(vec![1, 0]));
        assert!(FieldPath::new(vec![1, 1]) > FieldPath::new(vec![1, 0]));
    }

    #[test]
    fn test_field_path_deeply_nested() {
        // Create a more complex nested structure:
        // [
        //   SerializedValue([
        //     SerializedValue([
        //       SerializedValue(1)
        //     ])
        //   ])
        // ]
        let values = vec![SerializedValue::with_nested(
            &[0],
            FieldType::Vec(Box::new(FieldType::Int(IntType::U8))),
            vec![SerializedValue::with_nested(
                &[0],
                FieldType::Vec(Box::new(FieldType::Int(IntType::U8))),
                vec![SerializedValue::new(&[1], FieldType::Int(IntType::U8))],
            )],
        )];

        // Test path to deeply nested value
        let path = FieldPath::new(vec![0, 0, 0]);
        assert_eq!(find_value_in_object(&values, &path).unwrap().bytes, &[1]);

        // Test partial paths
        let path = FieldPath::new(vec![0]);
        assert_eq!(
            find_value_in_object(&values, &path)
                .unwrap()
                .nested_values
                .len(),
            1
        );

        let path = FieldPath::new(vec![0, 0]);
        assert_eq!(
            find_value_in_object(&values, &path)
                .unwrap()
                .nested_values
                .len(),
            1
        );

        // Test invalid deep paths
        let path = FieldPath::new(vec![0, 0, 0, 0]); // Too deep
        assert!(find_value_in_object(&values, &path).is_none());

        let path = FieldPath::new(vec![0, 1, 0]); // Invalid middle index
        assert!(find_value_in_object(&values, &path).is_none());
    }

    #[test]
    fn test_parse_slice_of_structs() -> Result<(), String> {
        let mut parser = DescriptorParser::new();
        parser
            .parse_file(
                "Inner { u16 }
                 Test { u8, slice<Inner, '0'>, bytes<2> }",
            )
            .unwrap();

        let descriptor = parser.descriptors.get("Test").unwrap().clone();
        let obj_parser = ObjectParser::new(descriptor, &parser);

        // Test data:
        // - u8 length field = 2
        // - slice of 2 Inner structs, each containing a u16
        // - trailing 2 bytes
        let data = [
            0x02, // length = 2
            0x34, 0x12, // first Inner.u16 (LE)
            0x78, 0x56, // second Inner.u16 (LE)
            0xAA, 0xBB, // trailing bytes
        ];

        let values = obj_parser.parse(&data).unwrap();
        assert_eq!(values.len(), 3); // length field, slice, trailing bytes

        // Check length field
        assert_eq!(values[0].bytes, &[0x02]);
        assert!(matches!(
            values[0].field_type(),
            FieldType::Int(IntType::U8)
        ));

        // Check slice field
        assert_eq!(values[1].bytes, &data[1..5]); // Both Inner structs
        assert!(matches!(values[1].field_type(),
            FieldType::Slice(ref inner, length_field) if
                matches!(**inner, FieldType::Struct(ref name) if name == "Inner") &&
                *length_field == 0
        ));

        // Check slice contents (the Inner structs)
        assert_eq!(values[1].nested_values.len(), 2);

        // First Inner struct
        let first_inner = &values[1].nested_values[0];
        assert_eq!(first_inner.bytes, &[0x34, 0x12]);
        assert_eq!(first_inner.nested_values.len(), 1);
        assert_eq!(first_inner.nested_values[0].bytes, &[0x34, 0x12]);

        // Second Inner struct
        let second_inner = &values[1].nested_values[1];
        assert_eq!(second_inner.bytes, &[0x78, 0x56]);
        assert_eq!(second_inner.nested_values.len(), 1);
        assert_eq!(second_inner.nested_values[0].bytes, &[0x78, 0x56]);

        // Check trailing bytes
        assert_eq!(values[2].bytes, &[0xAA, 0xBB]);
        assert!(matches!(values[2].field_type(), FieldType::Bytes(2)));

        // Test error case: truncated data
        let truncated_data = [
            0x02, // length = 2
            0x34, 0x12, // first Inner.u16
            0x78, // incomplete second Inner.u16
        ];
        assert!(obj_parser.parse(&truncated_data).is_err());

        Ok(())
    }

    #[test]
    fn test_encode_compact_size() {
        assert_eq!(encode_compact_size(0), vec![0]);
        assert_eq!(encode_compact_size(252), vec![252]);
        assert_eq!(encode_compact_size(253), vec![253, 253, 0]);
        assert_eq!(encode_compact_size(254), vec![253, 254, 0]);
        assert_eq!(encode_compact_size(0xffff), vec![253, 255, 255]);
        assert_eq!(encode_compact_size(0x10000), vec![254, 0, 0, 1, 0]);
        assert_eq!(
            encode_compact_size(0xffffffff),
            vec![254, 255, 255, 255, 255]
        );
        assert_eq!(
            encode_compact_size(0x100000000),
            vec![255, 0, 0, 0, 0, 1, 0, 0, 0]
        );
    }

    #[test]
    fn test_parse_constants() -> Result<(), String> {
        let mut parser = DescriptorParser::new();
        let descriptor = parse_descriptor_with_parser(
            "Test { 
                u8(0xff),           # Constant u8
                u8,                 # Regular u8
                bytes<2>(0xffff),   # Constant bytes
                bytes<2>            # Regular bytes
            }",
            &mut parser,
        )
        .unwrap();
        let obj_parser = ObjectParser::new(descriptor, &parser);

        // Test data matching the descriptor
        let data = [
            0xff, // First field (constant u8)
            0x42, // Second field (regular u8)
            0xff, 0xff, // Third field (constant bytes)
            0x12, 0x34, // Fourth field (regular bytes)
        ];

        let values = obj_parser.parse(&data).unwrap();
        assert_eq!(values.len(), 4);

        // Check first field (constant u8)
        assert_eq!(values[0].bytes, &[0xff]);
        assert!(values[0].is_constant);

        // Check second field (regular u8)
        assert_eq!(values[1].bytes, &[0x42]);
        assert!(!values[1].is_constant);

        // Check third field (constant bytes)
        assert_eq!(values[2].bytes, &[0xff, 0xff]);
        assert!(values[2].is_constant);

        // Check fourth field (regular bytes)
        assert_eq!(values[3].bytes, &[0x12, 0x34]);
        assert!(!values[3].is_constant);

        Ok(())
    }

    #[test]
    fn test_parse_alternatives() -> Result<(), String> {
        let mut parser = DescriptorParser::new();

        // Define a type with two alternative definitions
        parser
            .parse_file(
                "SomeType { u64 }
             SomeType { u32 }
             
             Test { SomeType }",
            )
            .unwrap();

        let descriptor = parser.descriptors.get("Test").unwrap().clone();
        let obj_parser = ObjectParser::new(descriptor, &parser);

        // Test parsing with u32 data (should work with the second alternative)
        let data_u32 = [0x78, 0x56, 0x34, 0x12]; // u32: 0x12345678
        let values_u32 = obj_parser.parse(&data_u32).unwrap();
        assert_eq!(values_u32.len(), 1);
        assert_eq!(values_u32[0].nested_values.len(), 1);
        assert_eq!(
            values_u32[0].nested_values[0].bytes,
            &[0x78, 0x56, 0x34, 0x12]
        );

        // Test parsing with u64 data (should work with the first alternative)
        let data_u64 = [0x78, 0x56, 0x34, 0x12, 0xFF, 0xFF, 0xFF, 0xFF]; // u64
        let values_u64 = obj_parser.parse(&data_u64).unwrap();
        assert_eq!(values_u64.len(), 1);
        assert_eq!(values_u64[0].nested_values.len(), 1);
        assert_eq!(
            values_u64[0].nested_values[0].bytes,
            &[0x78, 0x56, 0x34, 0x12, 0xFF, 0xFF, 0xFF, 0xFF]
        );

        // Test with invalid data length (not matching either alternative)
        let data_invalid = [0x12, 0x34]; // Only 2 bytes
        assert!(obj_parser.parse(&data_invalid).is_err());

        Ok(())
    }

    #[test]
    fn test_parse_constants_with_alternatives() -> Result<(), String> {
        let mut parser = DescriptorParser::new();

        // Define types with constants and alternatives
        parser
            .parse_file(
                "Version { 
                u8(0x01),     # Version 1
                bytes<2>      # Variable payload
             }
             Version {
                u8(0x02),     # Version 2 
                bytes<4>      # Different payload size
             }
             
             Test {
                u8(0xff),           # Fixed header
                Version,           # Version struct with alternatives
                bytes<2>(0xabcd)   # Fixed footer
             }",
            )
            .unwrap();

        let descriptor = parser.descriptors.get("Test").unwrap().clone();
        let obj_parser = ObjectParser::new(descriptor, &parser);

        // Test Version 1 format
        let data_v1 = [
            0xff, // Header constant
            0x01, // Version 1
            0x12, 0x34, // 2-byte payload
            0xab, 0xcd, // Footer constant
        ];
        let values = obj_parser.parse(&data_v1).unwrap();
        assert_eq!(values.len(), 3);
        assert!(values[0].is_constant);
        assert_eq!(values[0].bytes, &[0xff]);
        assert_eq!(values[1].nested_values[0].bytes, &[0x01]);
        assert!(values[1].nested_values[0].is_constant);
        assert_eq!(values[2].bytes, &[0xab, 0xcd]);
        assert!(values[2].is_constant);

        // Test Version 2 format
        let data_v2 = [
            0xff, // Header constant
            0x02, // Version 2
            0x12, 0x34, 0x56, 0x78, // 4-byte payload
            0xab, 0xcd, // Footer constant
        ];
        let values = obj_parser.parse(&data_v2).unwrap();
        assert_eq!(values.len(), 3);
        assert!(values[0].is_constant);
        assert_eq!(values[1].nested_values[0].bytes, &[0x02]);
        assert!(values[1].nested_values[0].is_constant);
        assert_eq!(values[2].bytes, &[0xab, 0xcd]);
        assert!(values[2].is_constant);

        // Test failure cases

        // Wrong header constant
        let data_wrong_header = [
            0xfe, // Wrong header (should be 0xff)
            0x01, // Version 1
            0x12, 0x34, // 2-byte payload
            0xab, 0xcd, // Footer constant
        ];
        assert!(obj_parser.parse(&data_wrong_header).is_err());

        // Wrong version constant
        let data_wrong_version = [
            0xff, // Header constant
            0x03, // Invalid version (neither 1 nor 2)
            0x12, 0x34, // 2-byte payload
            0xab, 0xcd, // Footer constant
        ];
        assert!(obj_parser.parse(&data_wrong_version).is_err());

        // Wrong footer constant
        let data_wrong_footer = [
            0xff, // Header constant
            0x01, // Version 1
            0x12, 0x34, // 2-byte payload
            0xab, 0xce, // Wrong footer (should be 0xabcd)
        ];
        assert!(obj_parser.parse(&data_wrong_footer).is_err());

        Ok(())
    }

    #[test]
    fn test_parse_extra_data() {
        let mut parser = DescriptorParser::new();
        let descriptor = parse_descriptor_with_parser("Test { u8 }", &mut parser).unwrap();
        let obj_parser = ObjectParser::new(descriptor, &parser);

        // Test data with extra bytes at the end
        let data = [0x42, 0xff, 0xff]; // Only first byte should be parsed
        assert!(obj_parser.parse(&data).is_err());
    }

    #[test]
    fn test_parse_truncated_data() {
        let mut parser = DescriptorParser::new();

        // Test bool truncation
        let descriptor = parse_descriptor_with_parser("Test { bool }", &mut parser).unwrap();
        let obj_parser = ObjectParser::new(descriptor, &parser);
        assert!(obj_parser.parse(&[]).is_err());

        // Test u256 truncation
        let descriptor = parse_descriptor_with_parser("Test { u256 }", &mut parser).unwrap();
        let obj_parser = ObjectParser::new(descriptor, &parser);
        let short_data = [0u8; 31]; // Only 31 bytes instead of required 32
        assert!(obj_parser.parse(&short_data).is_err());

        // Test slice truncation
        let descriptor =
            parse_descriptor_with_parser("Test { u8, slice<u8, '0'> }", &mut parser).unwrap();
        let obj_parser = ObjectParser::new(descriptor, &parser);
        let data = [0x02, 0x42]; // Promises 2 bytes but only has 1
        assert!(obj_parser.parse(&data).is_err());

        // Test vec truncation
        let descriptor = parse_descriptor_with_parser("Test { vec<u8> }", &mut parser).unwrap();
        let obj_parser = ObjectParser::new(descriptor, &parser);
        let data = [0x02, 0x42]; // Promises 2 elements but only has 1
        assert!(obj_parser.parse(&data).is_err());
    }

    #[test]
    fn test_parse_u256() -> Result<(), String> {
        let mut parser = DescriptorParser::new();
        let descriptor = parse_descriptor_with_parser("Test { u256, U256 }", &mut parser).unwrap();
        let obj_parser = ObjectParser::new(descriptor, &parser);

        let mut data = [0u8; 64]; // 32 bytes for each u256
        data[0] = 0x42; // Set LSB for LE
        data[31] = 0x43; // Set MSB for LE
        data[32] = 0x44; // Set MSB for BE
        data[63] = 0x45; // Set LSB for BE

        let values = obj_parser.parse(&data)?;
        assert_eq!(values.len(), 2);
        assert_eq!(values[0].bytes[0], 0x42); // Check LE byte order
        assert_eq!(values[0].bytes[31], 0x43);
        assert_eq!(values[1].bytes[0], 0x44); // Check BE byte order
        assert_eq!(values[1].bytes[31], 0x45);

        Ok(())
    }

    #[test]
    fn test_parse_integers() -> Result<(), String> {
        let mut parser = DescriptorParser::new();
        let descriptor = parse_descriptor_with_parser(
            "Test { 
                i8, u8,
                i16, u16, I16, U16,
                i32, u32, I32, U32,
                i64, u64, I64, U64
            }",
            &mut parser,
        )
        .unwrap();
        let obj_parser = ObjectParser::new(descriptor, &parser);

        // Create test data with various integer values
        let mut data = Vec::new();

        // i8/u8
        data.extend_from_slice(&[-42i8 as u8, 42u8]);

        // i16/u16 LE and BE
        data.extend_from_slice(&(-1000i16).to_le_bytes());
        data.extend_from_slice(&1000u16.to_le_bytes());
        data.extend_from_slice(&(-1000i16).to_be_bytes());
        data.extend_from_slice(&1000u16.to_be_bytes());

        // i32/u32 LE and BE
        data.extend_from_slice(&(-100000i32).to_le_bytes());
        data.extend_from_slice(&100000u32.to_le_bytes());
        data.extend_from_slice(&(-100000i32).to_be_bytes());
        data.extend_from_slice(&100000u32.to_be_bytes());

        // i64/u64 LE and BE
        data.extend_from_slice(&(-10000000i64).to_le_bytes());
        data.extend_from_slice(&10000000u64.to_le_bytes());
        data.extend_from_slice(&(-10000000i64).to_be_bytes());
        data.extend_from_slice(&10000000u64.to_be_bytes());

        let values = obj_parser.parse(&data)?;
        assert_eq!(values.len(), 14);

        // Verify i8/u8
        assert_eq!(values[0].bytes, &[-42i8 as u8]);
        assert_eq!(values[1].bytes, &[42u8]);

        // Verify i16/u16
        assert_eq!(values[2].bytes, &(-1000i16).to_le_bytes());
        assert_eq!(values[3].bytes, &1000u16.to_le_bytes());
        assert_eq!(values[4].bytes, &(-1000i16).to_be_bytes());
        assert_eq!(values[5].bytes, &1000u16.to_be_bytes());

        // Verify i32/u32
        assert_eq!(values[6].bytes, &(-100000i32).to_le_bytes());
        assert_eq!(values[7].bytes, &100000u32.to_le_bytes());
        assert_eq!(values[8].bytes, &(-100000i32).to_be_bytes());
        assert_eq!(values[9].bytes, &100000u32.to_be_bytes());

        // Verify i64/u64
        assert_eq!(values[10].bytes, &(-10000000i64).to_le_bytes());
        assert_eq!(values[11].bytes, &10000000u64.to_le_bytes());
        assert_eq!(values[12].bytes, &(-10000000i64).to_be_bytes());
        assert_eq!(values[13].bytes, &10000000u64.to_be_bytes());

        Ok(())
    }

    #[test]
    fn test_parse_varint_edge_cases() -> Result<(), String> {
        // Test edge cases for varint parsing
        let test_cases = vec![
            // (input_bytes, expected_result)
            (vec![0x00], Ok((0, 1))),              // Minimum value
            (vec![0x7F], Ok((127, 1))),            // Max single byte
            (vec![0x80, 0x00], Ok((128, 2))),      // Min two bytes
            (vec![0xFF, 0x7F], Ok((16511, 2))),    // Max two bytes
            (vec![0x80], Err("Truncated VarInt")), // Incomplete
            (vec![0x80, 0x80], Err("Truncated VarInt")), // Still incomplete
                                                   // Add more edge cases as needed
        ];

        for (input, expected) in test_cases {
            let result = decode_varint(&input);
            match (result, expected) {
                (Ok(actual), Ok(expected)) => {
                    assert_eq!(actual, expected, "Mismatch for input: {:?}", input);
                }
                (Err(actual_err), Err(expected_err)) => {
                    assert_eq!(
                        actual_err, expected_err,
                        "Error mismatch for input: {:?}",
                        input
                    );
                }
                _ => panic!("Result mismatch for input: {:?}", input),
            }
        }

        Ok(())
    }
}
