use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum IntType {
    U8,
    I8,
    U16,
    U16BE,
    U32,
    U32BE,
    U64,
    U64BE,
    U256,
    U256BE,
    I16,
    I16BE,
    I32,
    I32BE,
    I64,
    I64BE,
    I256,
    I256BE,
    CompactSize(bool),
    VarInt,
    VarIntNonNegative,
}

#[derive(Debug, Clone)]
pub enum FieldType {
    Bool,
    Int(IntType),
    Bytes(usize),
    Vec(Box<FieldType>),
    Slice(Box<FieldType>, u64),
    Struct(String),
}

#[derive(Debug, Clone)]
pub struct Field {
    pub name: String,
    pub field_type: FieldType,
    pub constant_value: Option<Vec<u8>>,
    pub length_field_for: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct FieldPath {
    pub indices: Vec<usize>,
}

impl std::fmt::Display for FieldPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (i, index) in self.indices.iter().enumerate() {
            if i > 0 {
                write!(f, ".")?;
            }
            write!(f, "{}", index)?;
        }
        write!(f, "]")
    }
}

impl FieldPath {
    pub fn new(indices: Vec<usize>) -> Self {
        Self { indices }
    }
}

#[derive(Debug, Clone)]
pub struct Descriptor {
    pub name: String,
    pub fields: Vec<Field>,
    pub alternatives: Vec<Descriptor>,
}

pub struct DescriptorParser {
    pub descriptors: HashMap<String, Descriptor>,
    current_definition: Option<(String, String)>,
    current_line: usize,
}

impl DescriptorParser {
    pub fn new() -> Self {
        Self {
            descriptors: HashMap::new(),
            current_definition: None,
            current_line: 0,
        }
    }

    fn err(&self, msg: String) -> String {
        format!("Line {}: {}", self.current_line, msg)
    }

    fn parse_line(&mut self, line: &str) -> Result<(), String> {
        self.current_line += 1;

        // Skip comments and empty lines
        let line = if let Some(comment_pos) = line.find('#') {
            line[..comment_pos].trim()
        } else {
            line.trim()
        };

        if line.is_empty() {
            return Ok(());
        }

        // Check if we're in the middle of a multi-line definition
        if let Some((name, partial_fields)) = &self.current_definition {
            if line.ends_with('}') {
                // Complete the definition
                let complete_fields = format!("{}{}", partial_fields, line.trim_end_matches('}'));
                let descriptor = self.parse_fields(name.clone(), &complete_fields)?;
                if let Some(existing) = self.descriptors.get_mut(name) {
                    existing.alternatives.push(descriptor);
                } else {
                    self.descriptors.insert(name.to_string(), descriptor);
                }
                self.current_definition = None;
                return Ok(());
            } else {
                // Continue collecting fields
                self.current_definition =
                    Some((name.clone(), format!("{}{}", partial_fields, line)));
                return Ok(());
            }
        }

        // Start of a new definition
        let parts: Vec<&str> = line.split('{').collect();
        if parts.len() != 2 {
            return Err(self.err(format!("Invalid line format: {}", line)));
        }

        let name = parts[0].trim().to_string();
        let fields_str = parts[1];

        if !fields_str.ends_with('}') {
            // Start of multi-line definition
            self.current_definition = Some((name, fields_str.to_string()));
            return Ok(());
        }

        let fields_str = fields_str.trim_end_matches('}');
        let descriptor = self.parse_fields(name.clone(), fields_str)?;
        if let Some(existing) = self.descriptors.get_mut(&name) {
            existing.alternatives.push(descriptor);
        } else {
            self.descriptors.insert(name, descriptor);
        }

        Ok(())
    }

    pub fn parse_file(&mut self, content: &str) -> Result<(), String> {
        self.current_line = 0;
        for line in content.lines() {
            self.parse_line(line)?;
        }

        if self.current_definition.is_some() {
            return Err(format!(
                "Line {}: Unclosed definition at end of file",
                self.current_line
            ));
        }

        Ok(())
    }

    fn parse_fields(&self, name: String, fields_str: &str) -> Result<Descriptor, String> {
        let mut fields = Vec::new();
        let mut current_field = String::new();
        let mut depth = 0;

        // Parse all fields first
        for c in fields_str.chars() {
            match c {
                '<' => {
                    depth += 1;
                    current_field.push(c);
                }
                '>' => {
                    depth -= 1;
                    current_field.push(c);
                }
                ',' if depth == 0 => {
                    if !current_field.trim().is_empty() {
                        let field = self
                            .parse_field(current_field.trim())
                            .map_err(|e| self.err(e))?;
                        fields.push(field);
                        current_field.clear();
                    }
                }
                _ => {
                    current_field.push(c);
                }
            }
        }

        // Don't forget the last field
        if !current_field.trim().is_empty() {
            let field = self
                .parse_field(current_field.trim())
                .map_err(|e| self.err(e))?;
            fields.push(field);
        }

        // Process length field relationships
        for i in 0..fields.len() {
            match &fields[i].field_type {
                FieldType::Slice(_, length_idx) => {
                    let idx = *length_idx as usize;
                    if idx >= i {
                        return Err(self.err(format!(
                            "Slice reference '{}' points to non-existent field",
                            length_idx
                        )));
                    }
                    // Mark the referenced field as a length field
                    fields[idx].length_field_for = Some(i as u64);
                }
                _ => {}
            }
        }

        Ok(Descriptor {
            name,
            fields,
            alternatives: Vec::new(),
        })
    }

    fn parse_field(&self, field_str: &str) -> Result<Field, String> {
        if field_str.is_empty() {
            return Err("Empty field string".to_string());
        }

        let name = field_str.trim().to_string();
        let (field_type, constant_value) = self.parse_type(field_str.trim())?;

        Ok(Field {
            name,
            field_type,
            constant_value,
            length_field_for: None,
        })
    }

    fn parse_type(&self, type_str: &str) -> Result<(FieldType, Option<Vec<u8>>), String> {
        if type_str.is_empty() {
            return Err("Empty type string".to_string());
        }

        // Check for constant value notation: type(0x...)
        if let Some(paren_idx) = type_str.find('(') {
            let base_type = &type_str[..paren_idx];
            let value_str = type_str[paren_idx + 1..].trim_end_matches(')');

            // Verify hex format and parse the constant value
            if !value_str.starts_with("0x") {
                return Err(format!(
                    "Constant value must be in hex format (0x...): {}",
                    value_str
                ));
            }

            let hex_str = &value_str[2..];
            let value = hex_str
                .as_bytes()
                .chunks(2)
                .map(|chunk| {
                    let hex_byte = std::str::from_utf8(chunk)
                        .map_err(|_| format!("Invalid hex string: {}", hex_str))?;
                    u8::from_str_radix(hex_byte, 16)
                        .map_err(|_| format!("Invalid hex value: {}", hex_byte))
                })
                .collect::<Result<Vec<u8>, String>>()?;

            return Ok((self.parse_base_type(base_type)?, Some(value)));
        }

        // Parse regular types without constant values
        Ok((self.parse_base_type(type_str)?, None))
    }

    fn parse_base_type(&self, type_str: &str) -> Result<FieldType, String> {
        // First handle primitive types
        let primitive_type = match type_str {
            "bool" => Some(FieldType::Bool),
            "u8" => Some(FieldType::Int(IntType::U8)),
            "i8" => Some(FieldType::Int(IntType::I8)),
            "u16" => Some(FieldType::Int(IntType::U16)),
            "u32" => Some(FieldType::Int(IntType::U32)),
            "u64" => Some(FieldType::Int(IntType::U64)),
            "u256" => Some(FieldType::Int(IntType::U256)),
            "i16" => Some(FieldType::Int(IntType::I16)),
            "i32" => Some(FieldType::Int(IntType::I32)),
            "i64" => Some(FieldType::Int(IntType::I64)),
            "i256" => Some(FieldType::Int(IntType::I256)),
            "U16" => Some(FieldType::Int(IntType::U16BE)),
            "U32" => Some(FieldType::Int(IntType::U32BE)),
            "U64" => Some(FieldType::Int(IntType::U64BE)),
            "U256" => Some(FieldType::Int(IntType::U256BE)),
            "I16" => Some(FieldType::Int(IntType::I16BE)),
            "I32" => Some(FieldType::Int(IntType::I32BE)),
            "I64" => Some(FieldType::Int(IntType::I64BE)),
            "I256" => Some(FieldType::Int(IntType::I256BE)),
            "cs64" => Some(FieldType::Int(IntType::CompactSize(true))),
            "varint" => Some(FieldType::Int(IntType::VarInt)),
            "varint+" => Some(FieldType::Int(IntType::VarIntNonNegative)),
            _ => None,
        };

        if let Some(field_type) = primitive_type {
            return Ok(field_type);
        }

        match type_str {
            _ if type_str.starts_with("vec<") => {
                // Find the matching closing bracket
                let mut depth = 0;
                let mut end_pos = 0;

                for (i, c) in type_str[4..].chars().enumerate() {
                    match c {
                        '<' => depth += 1,
                        '>' => {
                            if depth == 0 {
                                end_pos = i + 4; // Add 4 to account for "vec<" prefix
                                break;
                            }
                            depth -= 1;
                        }
                        _ => {}
                    }
                }

                if end_pos == 0 || end_pos != type_str.len() - 1 {
                    return Err(format!("Malformed vector type: {}", type_str));
                }

                let inner = &type_str[4..end_pos];
                let inner_type = self.parse_base_type(inner.trim())?;
                Ok(FieldType::Vec(Box::new(inner_type)))
            }
            _ if type_str.starts_with("bytes<") && type_str.ends_with('>') => {
                let size_str = &type_str[6..type_str.len() - 1];
                let size = size_str
                    .parse::<usize>()
                    .map_err(|_| format!("Invalid bytes size: {}", size_str))?;
                Ok(FieldType::Bytes(size))
            }
            _ if type_str.starts_with("slice<") => {
                let inner = &type_str[6..];
                let mut depth = 0;
                let mut comma_pos = None;

                for (i, c) in inner.chars().enumerate() {
                    match c {
                        '<' => depth += 1,
                        '>' => {
                            if depth == 0 {
                                break;
                            }
                            depth -= 1;
                        }
                        ',' => {
                            if depth == 0 {
                                comma_pos = Some(i);
                                break;
                            }
                        }
                        _ => {}
                    }
                }

                let comma_pos =
                    comma_pos.ok_or_else(|| format!("Invalid slice format: {}", type_str))?;

                let type_part = &inner[..comma_pos].trim();
                let index_part = &inner[comma_pos + 1..].trim_end_matches('>').trim();

                let field_ref = index_part.trim_matches('\'');
                let field_index = field_ref.parse::<u64>().map_err(|_| {
                    format!(
                        "Invalid field reference '{}' - must be a numeric index",
                        field_ref
                    )
                })?;

                Ok(FieldType::Slice(
                    Box::new(self.parse_base_type(type_part)?),
                    field_index,
                ))
            }
            _ => {
                // Only check for struct types if nothing else matched
                if self.descriptors.contains_key(type_str) {
                    Ok(FieldType::Struct(type_str.to_string()))
                } else {
                    Err(format!("Undefined struct type: {}", type_str))
                }
            }
        }
    }

    pub fn get_descriptor(&self, name: &str) -> Option<&Descriptor> {
        self.descriptors.get(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_single_message(content: &str) -> Result<Descriptor, String> {
        let mut parser = DescriptorParser::new();
        parser.parse_file(content)?;
        Ok(parser.descriptors.values().next().unwrap().clone())
    }

    #[test]
    fn test_primitive_types() -> Result<(), String> {
        let msg = parse_single_message("Test { u8, i8, u16, i16, u32, i32, u64, i64, bool }")?;
        assert_eq!(msg.fields.len(), 9);
        assert!(matches!(
            msg.fields[0].field_type,
            FieldType::Int(IntType::U8)
        ));
        assert!(matches!(msg.fields[8].field_type, FieldType::Bool));
        Ok(())
    }

    #[test]
    fn test_nested_vectors() -> Result<(), String> {
        let msg = parse_single_message("Test { vec<vec<u8>>, vec<u32> }")?;
        assert_eq!(msg.fields.len(), 2);

        if let FieldType::Vec(inner) = &msg.fields[0].field_type {
            if let FieldType::Vec(innermost) = &**inner {
                assert!(matches!(&**innermost, FieldType::Int(IntType::U8)));
            } else {
                panic!("Expected nested vector");
            }
        } else {
            panic!("Expected vector");
        }
        Ok(())
    }

    #[test]
    fn test_constant_values() -> Result<(), String> {
        let msg = parse_single_message("Test { u8(0xff), bytes<2>(0xffff) }")?;
        assert_eq!(msg.fields.len(), 2);

        assert_eq!(msg.fields[0].constant_value, Some(vec![0xff]));
        assert_eq!(msg.fields[1].constant_value, Some(vec![0xff, 0xff]));
        Ok(())
    }

    #[test]
    fn test_struct_references() -> Result<(), String> {
        let mut parser = DescriptorParser::new();
        parser.parse_file("Inner { u8 }\nOuter { Inner }")?;

        let outer = parser.descriptors.get("Outer").unwrap();
        assert_eq!(outer.fields.len(), 1);
        assert!(matches!(&outer.fields[0].field_type, FieldType::Struct(name) if name == "Inner"));
        Ok(())
    }

    #[test]
    fn test_error_cases() {
        // Malformed vector
        assert!(parse_single_message("Test { vec<u8 }").is_err());

        // Unknown type
        assert!(parse_single_message("Test { unknown_type }").is_err());

        // Invalid constant value
        assert!(parse_single_message("Test { u8(invalid) }").is_err());

        // Unclosed definition
        assert!(DescriptorParser::new().parse_file("Test { u8").is_err());

        // Invalid slice format
        assert!(parse_single_message("Test { slice<u8> }").is_err());
    }

    #[test]
    fn test_slices() -> Result<(), String> {
        // Test slice with integer length field
        let msg = parse_single_message("Test { u32, slice<u8, '0'> }")?;
        assert_eq!(msg.fields.len(), 2);

        // Verify length field relationship
        assert_eq!(msg.fields[0].length_field_for, Some(1));

        if let FieldType::Slice(inner, length_field) = &msg.fields[1].field_type {
            assert!(matches!(&**inner, FieldType::Int(IntType::U8)));
            assert_eq!(*length_field, 0);
        } else {
            panic!("Expected slice");
        }

        // Test slice with vec length field
        let msg = parse_single_message("Test { vec<u8>, slice<u16, '0'> }")?;
        assert_eq!(msg.fields.len(), 2);
        assert_eq!(msg.fields[0].length_field_for, Some(1));

        if let FieldType::Slice(inner, length_field) = &msg.fields[1].field_type {
            assert!(matches!(&**inner, FieldType::Int(IntType::U16)));
            assert_eq!(*length_field, 0);
        } else {
            panic!("Expected slice");
        }

        // Test slice with bytes length field
        let msg = parse_single_message("Test { bytes<4>, slice<u8, '0'> }")?;
        assert_eq!(msg.fields.len(), 2);
        assert_eq!(msg.fields[0].length_field_for, Some(1));

        if let FieldType::Slice(inner, length_field) = &msg.fields[1].field_type {
            assert!(matches!(&**inner, FieldType::Int(IntType::U8)));
            assert_eq!(*length_field, 0);
        } else {
            panic!("Expected slice");
        }

        Ok(())
    }

    #[test]
    fn test_slice_errors() {
        // Invalid field reference format
        assert!(parse_single_message("Test { u32, slice<u8, 'invalid'> }").is_err());

        // Missing field reference
        assert!(parse_single_message("Test { slice<u8, '0'> }").is_err());

        // Reference to non-existent field
        assert!(parse_single_message("Test { u32, slice<u8, '2'> }").is_err());

        // Invalid slice format
        assert!(parse_single_message("Test { slice<u8> }").is_err());
        assert!(parse_single_message("Test { slice<u8,> }").is_err());
        assert!(parse_single_message("Test { slice<,0> }").is_err());
    }

    #[test]
    fn test_multiline_definitions() -> Result<(), String> {
        let mut parser = DescriptorParser::new();

        // Test multi-line with different indentation and spacing
        parser.parse_file(
            "Message1 {
                u32,
                vec<u8>,
                slice<u16, '0'>
            }",
        )?;

        // Test multiple messages with mixed single/multi-line
        parser.parse_file(
            "Header { u32, u64 }
            Body {
                vec<u8>,
                slice<u32, '0'>,
                bytes<32>
            }
            Footer { u16 }",
        )?;

        // Verify the parsed structures
        let message1 = parser.descriptors.get("Message1").unwrap();
        assert_eq!(message1.fields.len(), 3);
        assert!(matches!(
            message1.fields[0].field_type,
            FieldType::Int(IntType::U32)
        ));

        let body = parser.descriptors.get("Body").unwrap();
        assert_eq!(body.fields.len(), 3);
        if let FieldType::Slice(inner, length_field) = &body.fields[1].field_type {
            assert!(matches!(&**inner, FieldType::Int(IntType::U32)));
            assert_eq!(*length_field, 0);
        } else {
            panic!("Expected slice type");
        }

        Ok(())
    }

    #[test]
    fn test_multiline_errors() {
        let mut parser = DescriptorParser::new();

        // Test unclosed definition
        assert!(parser
            .parse_file(
                "Message {
                    u32,
                    vec<u8>"
            )
            .is_err());

        // Test mismatched brackets
        assert!(parser
            .parse_file(
                "Message {
                    vec<u8>>,
                    u32
                }"
            )
            .is_err());

        // Test incomplete slice reference
        assert!(parser
            .parse_file(
                "Message {
                    u32,
                    slice<u8, >
                }"
            )
            .is_err());
    }

    #[test]
    fn test_comments_and_empty_lines() -> Result<(), String> {
        let mut parser = DescriptorParser::new();

        parser.parse_file(
            "# Header comment
            Message {
                # Field comment
                u32,
                # Another comment
                vec<u8>,
                slice<u16, '0'> # Inline comment
            }
            # Footer comment",
        )?;

        let message = parser.descriptors.get("Message").unwrap();
        assert_eq!(message.fields.len(), 3);
        Ok(())
    }

    #[test]
    fn test_field_path_display() {
        assert_eq!(FieldPath::new(vec![]).to_string(), "[]");
        assert_eq!(FieldPath::new(vec![0]).to_string(), "[0]");
        assert_eq!(FieldPath::new(vec![0, 1, 2]).to_string(), "[0.1.2]");
    }

    #[test]
    fn test_alternative_descriptors() -> Result<(), String> {
        let mut parser = DescriptorParser::new();

        parser.parse_file(
            "Message { u32 }
             Message { u64 }
             Message { bytes<8> }",
        )?;

        let message = parser.descriptors.get("Message").unwrap();
        assert_eq!(message.alternatives.len(), 2);

        // Check main descriptor
        assert_eq!(message.fields.len(), 1);
        assert!(matches!(
            message.fields[0].field_type,
            FieldType::Int(IntType::U32)
        ));

        // Check first alternative
        assert_eq!(message.alternatives[0].fields.len(), 1);
        assert!(matches!(
            message.alternatives[0].fields[0].field_type,
            FieldType::Int(IntType::U64)
        ));

        // Check second alternative
        assert_eq!(message.alternatives[1].fields.len(), 1);
        assert!(matches!(
            message.alternatives[1].fields[0].field_type,
            FieldType::Bytes(8)
        ));

        Ok(())
    }
}