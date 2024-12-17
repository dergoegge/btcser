use crate::lexer::{Lexer, Token};
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
    current_line: usize,
}

impl DescriptorParser {
    pub fn new() -> Self {
        Self {
            descriptors: HashMap::new(),
            current_line: 0,
        }
    }

    fn err(&self, msg: String) -> String {
        format!("Line {}: {}", self.current_line, msg)
    }

    pub fn parse_file(&mut self, content: &str) -> Result<(), String> {
        let mut lexer = Lexer::new(content);
        let mut tokens = Vec::new();

        // Collect all tokens first
        while let Some(token) = lexer.next_token()? {
            match token {
                Token::Comment(_) => continue, // Skip comments
                _ => tokens.push(token),
            }
        }

        let mut position = 0;
        while position < tokens.len() {
            self.current_line = lexer.line();
            position = self.parse_descriptor(&tokens, position)?;
        }

        Ok(())
    }

    fn parse_descriptor(&mut self, tokens: &[Token], start: usize) -> Result<usize, String> {
        let mut pos = start;

        // Expect identifier (struct name)
        let name = match tokens.get(pos) {
            Some(Token::Identifier(name)) => name.clone(),
            _ => return Err(self.err("Expected struct name".to_string())),
        };
        pos += 1;

        // Expect opening brace
        match tokens.get(pos) {
            Some(Token::OpenBrace) => pos += 1,
            _ => return Err(self.err("Expected '{'".to_string())),
        }

        let (fields, new_pos) = self.parse_fields(tokens, pos)?;
        pos = new_pos;

        // Expect closing brace
        match tokens.get(pos) {
            Some(Token::CloseBrace) => pos += 1,
            _ => return Err(self.err("Expected '}'".to_string())),
        }

        let descriptor = Descriptor {
            name: name.clone(),
            fields,
            alternatives: Vec::new(),
        };

        // Handle alternatives
        if let Some(existing) = self.descriptors.get_mut(&name) {
            existing.alternatives.push(descriptor);
        } else {
            self.descriptors.insert(name, descriptor);
        }

        Ok(pos)
    }

    fn parse_fields(&self, tokens: &[Token], start: usize) -> Result<(Vec<Field>, usize), String> {
        let mut pos = start;
        let mut fields = Vec::new();

        while pos < tokens.len() {
            match tokens.get(pos) {
                Some(Token::CloseBrace) => break,
                Some(Token::Comma) => {
                    pos += 1;
                    continue;
                }
                Some(Token::Identifier(_)) => {
                    let (field, new_pos) = self.parse_field(tokens, pos)?;
                    fields.push(field);
                    pos = new_pos;
                }
                err => return Err(self.err(format!("Expected field definition ({:?})", err))),
            }
        }

        // Process length field relationships
        for i in 0..fields.len() {
            if let FieldType::Slice(_, length_idx) = &fields[i].field_type {
                let idx = *length_idx as usize;
                if idx >= i {
                    return Err(self.err(format!(
                        "Slice reference '{}' points to non-existent field",
                        length_idx
                    )));
                }
                fields[idx].length_field_for = Some(i as u64);
            }
        }

        Ok((fields, pos))
    }

    fn parse_field(&self, tokens: &[Token], start: usize) -> Result<(Field, usize), String> {
        let mut pos = start;
        // Get field type identifier
        let type_name = match tokens.get(pos) {
            Some(Token::Identifier(name)) => name.clone(),
            _ => return Err(self.err("Expected type identifier".to_string())),
        };
        pos += 1;

        // Parse the type (including any angle brackets for generics)
        let field_type = self.parse_type(&type_name, tokens, &mut pos)?;

        // Check for constant value
        let mut constant_value = None;
        if let Some(Token::OpenParen) = tokens.get(pos) {
            pos += 1;
            match tokens.get(pos) {
                Some(Token::HexConstant(value)) => {
                    constant_value = Some(value.clone());
                    pos += 1;
                }
                _ => return Err(self.err("Expected hex constant".to_string())),
            }
            match tokens.get(pos) {
                Some(Token::CloseParen) => pos += 1,
                _ => return Err(self.err("Expected ')'".to_string())),
            }
        }

        Ok((
            Field {
                name: type_name,
                field_type,
                constant_value,
                length_field_for: None,
            },
            pos,
        ))
    }

    fn parse_type(
        &self,
        base_type: &str,
        tokens: &[Token],
        pos: &mut usize,
    ) -> Result<FieldType, String> {
        println!("{} {:?} {}", base_type, tokens, pos);
        match base_type {
            "bool" => Ok(FieldType::Bool),
            "u8" => Ok(FieldType::Int(IntType::U8)),
            "i8" => Ok(FieldType::Int(IntType::I8)),
            "u16" => Ok(FieldType::Int(IntType::U16)),
            "u32" => Ok(FieldType::Int(IntType::U32)),
            "u64" => Ok(FieldType::Int(IntType::U64)),
            "u256" => Ok(FieldType::Int(IntType::U256)),
            "i16" => Ok(FieldType::Int(IntType::I16)),
            "i32" => Ok(FieldType::Int(IntType::I32)),
            "i64" => Ok(FieldType::Int(IntType::I64)),
            "i256" => Ok(FieldType::Int(IntType::I256)),
            "U16" => Ok(FieldType::Int(IntType::U16BE)),
            "U32" => Ok(FieldType::Int(IntType::U32BE)),
            "U64" => Ok(FieldType::Int(IntType::U64BE)),
            "U256" => Ok(FieldType::Int(IntType::U256BE)),
            "I16" => Ok(FieldType::Int(IntType::I16BE)),
            "I32" => Ok(FieldType::Int(IntType::I32BE)),
            "I64" => Ok(FieldType::Int(IntType::I64BE)),
            "I256" => Ok(FieldType::Int(IntType::I256BE)),
            "cs64" => Ok(FieldType::Int(IntType::CompactSize(true))),
            "varint" => Ok(FieldType::Int(IntType::VarInt)),
            "varint+" => Ok(FieldType::Int(IntType::VarIntNonNegative)),
            "vec" => self.parse_vector_type(tokens, pos),
            "bytes" => self.parse_bytes_type(tokens, pos),
            "slice" => self.parse_slice_type(tokens, pos),
            _ => {
                if self.descriptors.contains_key(base_type) {
                    Ok(FieldType::Struct(base_type.to_string()))
                } else {
                    Err(format!("Undefined type: {}", base_type))
                }
            }
        }
    }

    fn parse_vector_type(&self, tokens: &[Token], pos: &mut usize) -> Result<FieldType, String> {
        match tokens.get(*pos) {
            Some(Token::OpenAngle) => {
                *pos += 1;
            }
            _ => return Err(self.err("Expected '<' after vec".to_string())),
        };

        let inner_type_name = match tokens.get(*pos) {
            Some(Token::Identifier(inner_type_name)) => inner_type_name.clone(),
            _ => return Err(self.err("Expected type after vec".to_string())),
        };
        *pos += 1;

        if inner_type_name == "slice" {
            return Err(self.err("Can't nest 'slice' in vec".to_string()));
        }

        let inner_type = self.parse_type(&inner_type_name, tokens, pos)?;

        match tokens.get(*pos) {
            Some(Token::CloseAngle) => *pos += 1,
            _ => return Err(self.err("Expected '>' after vec type".to_string())),
        }

        Ok(FieldType::Vec(Box::new(inner_type)))
    }

    fn parse_bytes_type(&self, tokens: &[Token], pos: &mut usize) -> Result<FieldType, String> {
        match tokens.get(*pos) {
            Some(Token::OpenAngle) => {
                *pos += 1;
                match tokens.get(*pos) {
                    Some(Token::Number(size)) => {
                        *pos += 1;
                        match tokens.get(*pos) {
                            Some(Token::CloseAngle) => {
                                *pos += 1;
                                Ok(FieldType::Bytes(*size as usize))
                            }
                            _ => Err(self.err("Expected '>' after size".to_string())),
                        }
                    }
                    _ => Err(self.err("Expected number after bytes<".to_string())),
                }
            }
            _ => Err(self.err("Expected '<' after bytes".to_string())),
        }
    }

    fn parse_slice_type(&self, tokens: &[Token], pos: &mut usize) -> Result<FieldType, String> {
        match tokens.get(*pos) {
            Some(Token::OpenAngle) => {
                *pos += 1;
                let element_type = self.parse_type_until_comma(tokens, pos)?;

                match tokens.get(*pos) {
                    Some(Token::Quote) => {
                        *pos += 1;
                        match tokens.get(*pos) {
                            Some(Token::Number(index)) => {
                                *pos += 1;
                                match tokens.get(*pos) {
                                    Some(Token::Quote) => {
                                        *pos += 1;
                                        match tokens.get(*pos) {
                                            Some(Token::CloseAngle) => {
                                                *pos += 1;
                                                Ok(FieldType::Slice(Box::new(element_type), *index))
                                            }
                                            _ => Err(self.err(
                                                "Expected '>' after slice reference".to_string(),
                                            )),
                                        }
                                    }
                                    _ => Err(self.err(
                                        "Expected closing quote for slice length field".to_string(),
                                    )),
                                }
                            }
                            _ => Err(self.err("Expected length field index as number".to_string())),
                        }
                    }
                    _ => Err(self.err("Expected quote for slice length field".to_string())),
                }
            }
            _ => Err(self.err("Expected '<' after slice".to_string())),
        }
    }

    fn parse_type_until_comma(
        &self,
        tokens: &[Token],
        pos: &mut usize,
    ) -> Result<FieldType, String> {
        let type_token = match tokens.get(*pos) {
            Some(Token::Identifier(name)) => name.clone(),
            _ => return Err(self.err("Expected type identifier".to_string())),
        };
        *pos += 1;

        if type_token == "slice" {
            return Err(self.err("Can't nest slices".to_string()));
        }

        match tokens.get(*pos) {
            Some(Token::Comma) => {
                *pos += 1;
                self.parse_type(&type_token, tokens, pos)
            }
            _ => Err(self.err("Expected ','".to_string())),
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
        let msg = parse_single_message("Test { vec<vec<u8>>, vec<u32> }").unwrap();
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
        let msg = parse_single_message("Test { u8(0xff), bytes<2>(0xffff) }").unwrap();
        assert_eq!(msg.fields.len(), 2);

        assert_eq!(msg.fields[0].constant_value, Some(vec![0xff]));
        assert_eq!(msg.fields[1].constant_value, Some(vec![0xff, 0xff]));
        Ok(())
    }

    #[test]
    fn test_struct_references() -> Result<(), String> {
        let mut parser = DescriptorParser::new();
        parser.parse_file("Inner { u8 }\nOuter { Inner }").unwrap();

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
