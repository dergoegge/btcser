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
    fn test_basic_tokens() -> Result<(), String> {
        let tokens = tokenize("Message { u32 }").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Identifier("Message".to_string()),
                Token::OpenBrace,
                Token::Identifier("u32".to_string()),
                Token::CloseBrace,
            ]
        );
        Ok(())
    }

    #[test]
    fn test_complex_tokens() -> Result<(), String> {
        let tokens = tokenize("Test { vec<u8>, slice<u16, '0'>, bytes<2>(0xff) }").unwrap();
        assert!(tokens.len() > 0);
        Ok(())
    }

    #[test]
    fn test_hex_constants() -> Result<(), String> {
        let tokens = tokenize("Test { u8(0xff) }").unwrap();
        assert!(matches!(
            tokens[4],
            Token::HexConstant(ref v) if v == &vec![0xff]
        ));
        Ok(())
    }

    #[test]
    fn test_comments() -> Result<(), String> {
        let tokens = tokenize("# This is a comment\nTest { u8 } # inline comment").unwrap();
        assert!(matches!(
            tokens[0],
            Token::Comment(ref s) if s.trim() == "This is a comment"
        ));
        Ok(())
    }
}
