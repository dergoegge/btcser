#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    OpenBrace,
    CloseBrace,
    OpenAngle,
    CloseAngle,
    OpenParen,
    CloseParen,
    Comma,
    Identifier(String),
    Number(u64),
    HexConstant(Vec<u8>),
    Quote,
    Comment(String),
}

pub struct Lexer {
    input: Vec<char>,
    position: usize,
    line: usize,
}

impl Lexer {
    pub fn new(input: &str) -> Self {
        Self {
            input: input.chars().collect(),
            position: 0,
            line: 1,
        }
    }

    pub fn next_token(&mut self) -> Result<Option<Token>, String> {
        self.skip_whitespace();

        if self.position >= self.input.len() {
            return Ok(None);
        }

        let token = match self.current_char() {
            '{' => {
                self.advance();
                Token::OpenBrace
            }
            '}' => {
                self.advance();
                Token::CloseBrace
            }
            '<' => {
                self.advance();
                Token::OpenAngle
            }
            '>' => {
                self.advance();
                Token::CloseAngle
            }
            ',' => {
                self.advance();
                Token::Comma
            }
            '\'' => {
                self.advance();
                Token::Quote
            }
            '#' => {
                self.advance();
                let comment = self.read_until_newline();
                Token::Comment(comment)
            }
            '0' if self.peek() == Some('x') => {
                self.advance(); // skip '0'
                self.advance(); // skip 'x'
                let hex_str = self.read_hex();
                let bytes = hex_str
                    .as_bytes()
                    .chunks(2)
                    .map(|chunk| {
                        let hex_byte = std::str::from_utf8(chunk)
                            .map_err(|_| format!("Invalid hex string: {}", hex_str))?;
                        u8::from_str_radix(hex_byte, 16)
                            .map_err(|_| format!("Invalid hex value: {}", hex_byte))
                    })
                    .collect::<Result<Vec<u8>, String>>()?;
                Token::HexConstant(bytes)
            }
            c if c.is_ascii_digit() => {
                let num_str = self.read_number();
                let num = num_str
                    .parse::<u64>()
                    .map_err(|_| format!("Invalid number: {}", num_str))?;
                Token::Number(num)
            }
            c if c.is_ascii_alphabetic() || c == '_' => {
                let ident = self.read_identifier();
                Token::Identifier(ident)
            }
            '(' => {
                self.advance();
                Token::OpenParen
            }
            ')' => {
                self.advance();
                Token::CloseParen
            }
            c => return Err(format!("Unexpected character: {}", c)),
        };

        Ok(Some(token))
    }

    fn current_char(&self) -> char {
        self.input[self.position]
    }

    fn peek(&self) -> Option<char> {
        if self.position + 1 < self.input.len() {
            Some(self.input[self.position + 1])
        } else {
            None
        }
    }

    fn advance(&mut self) {
        if self.position < self.input.len() {
            if self.current_char() == '\n' {
                self.line += 1;
            }
            self.position += 1;
        }
    }

    fn skip_whitespace(&mut self) {
        while self.position < self.input.len() && self.current_char().is_whitespace() {
            self.advance();
        }
    }

    fn read_identifier(&mut self) -> String {
        let mut result = String::new();
        while self.position < self.input.len() {
            let c = self.current_char();
            if c.is_ascii_alphanumeric() || c == '_' {
                result.push(c);
                self.advance();
            } else {
                break;
            }
        }
        result
    }

    fn read_number(&mut self) -> String {
        let mut result = String::new();
        while self.position < self.input.len() {
            let c = self.current_char();
            if c.is_ascii_digit() {
                result.push(c);
                self.advance();
            } else {
                break;
            }
        }
        result
    }

    fn read_hex(&mut self) -> String {
        let mut result = String::new();
        while self.position < self.input.len() {
            let c = self.current_char();
            if c.is_ascii_hexdigit() {
                result.push(c);
                self.advance();
            } else {
                break;
            }
        }
        result
    }

    fn read_until_newline(&mut self) -> String {
        let mut result = String::new();
        while self.position < self.input.len() && self.current_char() != '\n' {
            result.push(self.current_char());
            self.advance();
        }
        result
    }

    pub fn line(&self) -> usize {
        self.line
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenize(input: &str) -> Result<Vec<Token>, String> {
        let mut lexer = Lexer::new(input);
        let mut tokens = Vec::new();
        while let Some(token) = lexer.next_token()? {
            tokens.push(token);
        }
        Ok(tokens)
    }

    #[test]
    fn test_lexer_cases() -> Result<(), String> {
        let test_cases = vec![
            (
                "basic message definition",
                "Message { u32 }",
                vec![
                    Token::Identifier("Message".to_string()),
                    Token::OpenBrace,
                    Token::Identifier("u32".to_string()),
                    Token::CloseBrace,
                ],
            ),
            (
                "complex type expressions",
                "Test { vec<u8>, slice<u16, '0'>, bytes<2>(0xff) }",
                vec![
                    Token::Identifier("Test".to_string()),
                    Token::OpenBrace,
                    Token::Identifier("vec".to_string()),
                    Token::OpenAngle,
                    Token::Identifier("u8".to_string()),
                    Token::CloseAngle,
                    Token::Comma,
                    Token::Identifier("slice".to_string()),
                    Token::OpenAngle,
                    Token::Identifier("u16".to_string()),
                    Token::Comma,
                    Token::Quote,
                    Token::Number(0),
                    Token::Quote,
                    Token::CloseAngle,
                    Token::Comma,
                    Token::Identifier("bytes".to_string()),
                    Token::OpenAngle,
                    Token::Number(2),
                    Token::CloseAngle,
                    Token::OpenParen,
                    Token::HexConstant(vec![0xff]),
                    Token::CloseParen,
                    Token::CloseBrace,
                ],
            ),
            (
                "hex constants",
                "Test { u8(0xff) }",
                vec![
                    Token::Identifier("Test".to_string()),
                    Token::OpenBrace,
                    Token::Identifier("u8".to_string()),
                    Token::OpenParen,
                    Token::HexConstant(vec![0xff]),
                    Token::CloseParen,
                    Token::CloseBrace,
                ],
            ),
            (
                "comments",
                "# This is a comment\nTest { u8 } # inline comment",
                vec![
                    Token::Comment(" This is a comment".to_string()),
                    Token::Identifier("Test".to_string()),
                    Token::OpenBrace,
                    Token::Identifier("u8".to_string()),
                    Token::CloseBrace,
                    Token::Comment(" inline comment".to_string()),
                ],
            ),
            ("empty input", "", vec![]),
            ("whitespace only", "   \n\t   \r\n", vec![]),
            (
                "multiple consecutive comments",
                "# Comment 1\n# Comment 2\nTest",
                vec![
                    Token::Comment(" Comment 1".to_string()),
                    Token::Comment(" Comment 2".to_string()),
                    Token::Identifier("Test".to_string()),
                ],
            ),
            (
                "numbers at identifier boundaries",
                "u32_type123 456",
                vec![
                    Token::Identifier("u32_type123".to_string()),
                    Token::Number(456),
                ],
            ),
            (
                "odd-length hex constant",
                "Test { u8(0x0) }",
                vec![
                    Token::Identifier("Test".to_string()),
                    Token::OpenBrace,
                    Token::Identifier("u8".to_string()),
                    Token::OpenParen,
                    Token::HexConstant(vec![0]),
                    Token::CloseParen,
                    Token::CloseBrace,
                ],
            ),
            (
                "multiple hex bytes",
                "Test { bytes(0xdeadbeef) }",
                vec![
                    Token::Identifier("Test".to_string()),
                    Token::OpenBrace,
                    Token::Identifier("bytes".to_string()),
                    Token::OpenParen,
                    Token::HexConstant(vec![0xde, 0xad, 0xbe, 0xef]),
                    Token::CloseParen,
                    Token::CloseBrace,
                ],
            ),
        ];

        for (test_name, input, expected_tokens) in test_cases {
            let tokens = tokenize(input)?;
            assert_eq!(tokens, expected_tokens, "Failed test case: {}", test_name);
        }

        Ok(())
    }
}
