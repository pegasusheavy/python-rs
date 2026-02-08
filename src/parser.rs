/// Recursive-descent parser with Pratt precedence climbing.
use crate::ast::*;
use crate::error::PythonError;
use crate::lexer::{Token, TokenKind};

/// Parse a token stream into an AST module.
pub fn parse(tokens: Vec<Token>) -> Result<Module, PythonError> {
    let mut parser = Parser::new(tokens);
    let body = parser.parse_module()?;
    Ok(Module { body })
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    fn peek(&self) -> &TokenKind {
        if self.pos < self.tokens.len() {
            &self.tokens[self.pos].kind
        } else {
            &TokenKind::Eof
        }
    }

    fn peek_line(&self) -> u32 {
        if self.pos < self.tokens.len() {
            self.tokens[self.pos].line
        } else {
            0
        }
    }

    fn advance(&mut self) -> &Token {
        let tok = &self.tokens[self.pos];
        self.pos += 1;
        tok
    }

    fn expect(&mut self, expected: &TokenKind) -> Result<(), PythonError> {
        if self.peek() == expected {
            self.advance();
            Ok(())
        } else {
            Err(PythonError::parse(
                format!("expected {expected:?}, got {:?}", self.peek()),
                self.peek_line(),
            ))
        }
    }

    fn eat_newlines(&mut self) {
        while self.peek() == &TokenKind::Newline {
            self.advance();
        }
    }

    fn parse_module(&mut self) -> Result<Vec<Stmt>, PythonError> {
        let mut stmts = Vec::new();
        self.eat_newlines();
        while self.peek() != &TokenKind::Eof {
            stmts.push(self.parse_stmt()?);
            self.eat_newlines();
        }
        Ok(stmts)
    }

    fn parse_block(&mut self) -> Result<Vec<Stmt>, PythonError> {
        self.expect(&TokenKind::Newline)?;
        self.expect(&TokenKind::Indent)?;
        let mut stmts = Vec::new();
        while self.peek() != &TokenKind::Dedent && self.peek() != &TokenKind::Eof {
            stmts.push(self.parse_stmt()?);
            self.eat_newlines();
        }
        if self.peek() == &TokenKind::Dedent {
            self.advance();
        }
        Ok(stmts)
    }

    fn parse_stmt(&mut self) -> Result<Stmt, PythonError> {
        match self.peek().clone() {
            TokenKind::If => self.parse_if(),
            TokenKind::While => self.parse_while(),
            TokenKind::For => self.parse_for(),
            TokenKind::Def => self.parse_function_def(),
            TokenKind::Return => self.parse_return(),
            TokenKind::Pass => self.parse_pass(),
            TokenKind::Break => self.parse_break(),
            TokenKind::Continue => self.parse_continue(),
            _ => self.parse_assign_or_expr(),
        }
    }

    fn parse_if(&mut self) -> Result<Stmt, PythonError> {
        let line = self.peek_line();
        self.advance(); // consume 'if'
        let condition = self.parse_expr()?;
        self.expect(&TokenKind::Colon)?;
        let body = self.parse_block()?;

        let mut elif_clauses = Vec::new();
        let mut else_body = Vec::new();

        loop {
            self.eat_newlines();
            if self.peek() == &TokenKind::Elif {
                self.advance();
                let elif_cond = self.parse_expr()?;
                self.expect(&TokenKind::Colon)?;
                let elif_body = self.parse_block()?;
                elif_clauses.push((elif_cond, elif_body));
            } else if self.peek() == &TokenKind::Else {
                self.advance();
                self.expect(&TokenKind::Colon)?;
                else_body = self.parse_block()?;
                break;
            } else {
                break;
            }
        }

        Ok(Stmt::If { condition, body, elif_clauses, else_body, line })
    }

    fn parse_while(&mut self) -> Result<Stmt, PythonError> {
        let line = self.peek_line();
        self.advance(); // consume 'while'
        let condition = self.parse_expr()?;
        self.expect(&TokenKind::Colon)?;
        let body = self.parse_block()?;
        Ok(Stmt::While { condition, body, line })
    }

    fn parse_for(&mut self) -> Result<Stmt, PythonError> {
        let line = self.peek_line();
        self.advance(); // consume 'for'
        let target = match self.peek().clone() {
            TokenKind::Ident(name) => {
                self.advance();
                name
            }
            _ => return Err(PythonError::parse("expected identifier after 'for'", self.peek_line())),
        };
        self.expect(&TokenKind::In)?;
        let iter = self.parse_expr()?;
        self.expect(&TokenKind::Colon)?;
        let body = self.parse_block()?;
        Ok(Stmt::For { target, iter, body, line })
    }

    fn parse_function_def(&mut self) -> Result<Stmt, PythonError> {
        let line = self.peek_line();
        self.advance(); // consume 'def'
        let name = match self.peek().clone() {
            TokenKind::Ident(name) => {
                self.advance();
                name
            }
            _ => return Err(PythonError::parse("expected function name", self.peek_line())),
        };
        self.expect(&TokenKind::LParen)?;
        let mut params = Vec::new();
        while self.peek() != &TokenKind::RParen {
            if !params.is_empty() {
                self.expect(&TokenKind::Comma)?;
            }
            match self.peek().clone() {
                TokenKind::Ident(p) => {
                    self.advance();
                    params.push(p);
                }
                _ => return Err(PythonError::parse("expected parameter name", self.peek_line())),
            }
        }
        self.expect(&TokenKind::RParen)?;
        self.expect(&TokenKind::Colon)?;
        let body = self.parse_block()?;
        Ok(Stmt::FunctionDef { name, params, body, line })
    }

    fn parse_return(&mut self) -> Result<Stmt, PythonError> {
        let line = self.peek_line();
        self.advance(); // consume 'return'
        let value = if self.peek() == &TokenKind::Newline || self.peek() == &TokenKind::Eof {
            None
        } else {
            Some(self.parse_expr()?)
        };
        if self.peek() == &TokenKind::Newline {
            self.advance();
        }
        Ok(Stmt::Return { value, line })
    }

    fn parse_pass(&mut self) -> Result<Stmt, PythonError> {
        let line = self.peek_line();
        self.advance(); // consume 'pass'
        if self.peek() == &TokenKind::Newline {
            self.advance();
        }
        Ok(Stmt::Pass { line })
    }

    fn parse_break(&mut self) -> Result<Stmt, PythonError> {
        let line = self.peek_line();
        self.advance(); // consume 'break'
        if self.peek() == &TokenKind::Newline {
            self.advance();
        }
        Ok(Stmt::Break { line })
    }

    fn parse_continue(&mut self) -> Result<Stmt, PythonError> {
        let line = self.peek_line();
        self.advance(); // consume 'continue'
        if self.peek() == &TokenKind::Newline {
            self.advance();
        }
        Ok(Stmt::Continue { line })
    }

    fn parse_assign_or_expr(&mut self) -> Result<Stmt, PythonError> {
        let line = self.peek_line();
        let expr = self.parse_expr()?;

        // Check for assignment: name = expr
        if let Expr::Name { id, .. } = &expr {
            let id = id.clone();
            match self.peek() {
                TokenKind::Assign => {
                    self.advance();
                    let value = self.parse_expr()?;
                    if self.peek() == &TokenKind::Newline {
                        self.advance();
                    }
                    return Ok(Stmt::Assign { target: id, value, line });
                }
                TokenKind::PlusAssign
                | TokenKind::MinusAssign
                | TokenKind::StarAssign
                | TokenKind::SlashAssign
                | TokenKind::DoubleSlashAssign
                | TokenKind::PercentAssign
                | TokenKind::DoubleStarAssign => {
                    let op = match self.peek() {
                        TokenKind::PlusAssign => BinOp::Add,
                        TokenKind::MinusAssign => BinOp::Sub,
                        TokenKind::StarAssign => BinOp::Mul,
                        TokenKind::SlashAssign => BinOp::Div,
                        TokenKind::DoubleSlashAssign => BinOp::FloorDiv,
                        TokenKind::PercentAssign => BinOp::Mod,
                        TokenKind::DoubleStarAssign => BinOp::Pow,
                        _ => unreachable!(),
                    };
                    self.advance();
                    let value = self.parse_expr()?;
                    if self.peek() == &TokenKind::Newline {
                        self.advance();
                    }
                    return Ok(Stmt::AugAssign { target: id, op, value, line });
                }
                _ => {}
            }
        }

        if self.peek() == &TokenKind::Newline {
            self.advance();
        }
        Ok(Stmt::ExprStmt { expr, line })
    }

    // Expression parsing using precedence climbing

    fn parse_expr(&mut self) -> Result<Expr, PythonError> {
        self.parse_or()
    }

    fn parse_or(&mut self) -> Result<Expr, PythonError> {
        let mut left = self.parse_and()?;
        while self.peek() == &TokenKind::Or {
            let line = self.peek_line();
            self.advance();
            let right = self.parse_and()?;
            left = Expr::BoolOp {
                op: BoolOpKind::Or,
                left: Box::new(left),
                right: Box::new(right),
                line,
            };
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<Expr, PythonError> {
        let mut left = self.parse_not()?;
        while self.peek() == &TokenKind::And {
            let line = self.peek_line();
            self.advance();
            let right = self.parse_not()?;
            left = Expr::BoolOp {
                op: BoolOpKind::And,
                left: Box::new(left),
                right: Box::new(right),
                line,
            };
        }
        Ok(left)
    }

    fn parse_not(&mut self) -> Result<Expr, PythonError> {
        if self.peek() == &TokenKind::Not {
            let line = self.peek_line();
            self.advance();
            let operand = self.parse_not()?;
            Ok(Expr::UnaryOp {
                op: UnaryOp::Not,
                operand: Box::new(operand),
                line,
            })
        } else {
            self.parse_comparison()
        }
    }

    fn parse_comparison(&mut self) -> Result<Expr, PythonError> {
        let left = self.parse_addition()?;
        let line = self.peek_line();

        let mut ops = Vec::new();
        let mut comparators = Vec::new();

        loop {
            let op = match self.peek() {
                TokenKind::Eq => CmpOp::Eq,
                TokenKind::NotEq => CmpOp::NotEq,
                TokenKind::Lt => CmpOp::Lt,
                TokenKind::LtEq => CmpOp::LtEq,
                TokenKind::Gt => CmpOp::Gt,
                TokenKind::GtEq => CmpOp::GtEq,
                _ => break,
            };
            self.advance();
            ops.push(op);
            comparators.push(self.parse_addition()?);
        }

        if ops.is_empty() {
            Ok(left)
        } else {
            Ok(Expr::Compare {
                left: Box::new(left),
                ops,
                comparators,
                line,
            })
        }
    }

    fn parse_addition(&mut self) -> Result<Expr, PythonError> {
        let mut left = self.parse_multiplication()?;
        loop {
            let op = match self.peek() {
                TokenKind::Plus => BinOp::Add,
                TokenKind::Minus => BinOp::Sub,
                _ => break,
            };
            let line = self.peek_line();
            self.advance();
            let right = self.parse_multiplication()?;
            left = Expr::BinOp {
                left: Box::new(left),
                op,
                right: Box::new(right),
                line,
            };
        }
        Ok(left)
    }

    fn parse_multiplication(&mut self) -> Result<Expr, PythonError> {
        let mut left = self.parse_unary()?;
        loop {
            let op = match self.peek() {
                TokenKind::Star => BinOp::Mul,
                TokenKind::Slash => BinOp::Div,
                TokenKind::DoubleSlash => BinOp::FloorDiv,
                TokenKind::Percent => BinOp::Mod,
                _ => break,
            };
            let line = self.peek_line();
            self.advance();
            let right = self.parse_unary()?;
            left = Expr::BinOp {
                left: Box::new(left),
                op,
                right: Box::new(right),
                line,
            };
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<Expr, PythonError> {
        if self.peek() == &TokenKind::Minus {
            let line = self.peek_line();
            self.advance();
            let operand = self.parse_unary()?;
            Ok(Expr::UnaryOp {
                op: UnaryOp::Neg,
                operand: Box::new(operand),
                line,
            })
        } else {
            self.parse_power()
        }
    }

    fn parse_power(&mut self) -> Result<Expr, PythonError> {
        let base = self.parse_call()?;
        if self.peek() == &TokenKind::DoubleStar {
            let line = self.peek_line();
            self.advance();
            let exp = self.parse_unary()?; // right-associative
            Ok(Expr::BinOp {
                left: Box::new(base),
                op: BinOp::Pow,
                right: Box::new(exp),
                line,
            })
        } else {
            Ok(base)
        }
    }

    fn parse_call(&mut self) -> Result<Expr, PythonError> {
        let mut expr = self.parse_atom()?;

        loop {
            match self.peek() {
                TokenKind::LParen => {
                    let line = self.peek_line();
                    self.advance(); // consume '('
                    let mut args = Vec::new();
                    while self.peek() != &TokenKind::RParen {
                        if !args.is_empty() {
                            self.expect(&TokenKind::Comma)?;
                            // Allow trailing comma
                            if self.peek() == &TokenKind::RParen {
                                break;
                            }
                        }
                        args.push(self.parse_expr()?);
                    }
                    self.expect(&TokenKind::RParen)?;
                    expr = Expr::Call {
                        func: Box::new(expr),
                        args,
                        line,
                    };
                }
                TokenKind::LBracket => {
                    let line = self.peek_line();
                    self.advance(); // consume '['
                    let index = self.parse_expr()?;
                    self.expect(&TokenKind::RBracket)?;
                    expr = Expr::Subscript {
                        value: Box::new(expr),
                        index: Box::new(index),
                        line,
                    };
                }
                _ => break,
            }
        }

        Ok(expr)
    }

    fn parse_atom(&mut self) -> Result<Expr, PythonError> {
        let line = self.peek_line();
        match self.peek().clone() {
            TokenKind::IntLit(v) => {
                self.advance();
                Ok(Expr::IntLit { value: v, line })
            }
            TokenKind::FloatLit(v) => {
                self.advance();
                Ok(Expr::FloatLit { value: v, line })
            }
            TokenKind::StringLit(s) => {
                self.advance();
                Ok(Expr::StringLit { value: s, line })
            }
            TokenKind::True => {
                self.advance();
                Ok(Expr::BoolLit { value: true, line })
            }
            TokenKind::False => {
                self.advance();
                Ok(Expr::BoolLit { value: false, line })
            }
            TokenKind::None => {
                self.advance();
                Ok(Expr::NoneLit { line })
            }
            TokenKind::Ident(name) => {
                self.advance();
                Ok(Expr::Name { id: name, line })
            }
            TokenKind::LParen => {
                self.advance();
                let expr = self.parse_expr()?;
                self.expect(&TokenKind::RParen)?;
                Ok(expr)
            }
            TokenKind::LBracket => {
                self.advance();
                let mut elements = Vec::new();
                while self.peek() != &TokenKind::RBracket {
                    if !elements.is_empty() {
                        self.expect(&TokenKind::Comma)?;
                        if self.peek() == &TokenKind::RBracket {
                            break;
                        }
                    }
                    elements.push(self.parse_expr()?);
                }
                self.expect(&TokenKind::RBracket)?;
                Ok(Expr::List { elements, line })
            }
            _ => Err(PythonError::parse(
                format!("unexpected token {:?}", self.peek()),
                line,
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::tokenize;

    fn parse_str(src: &str) -> Module {
        let tokens = tokenize(src).unwrap();
        parse(tokens).unwrap()
    }

    #[test]
    fn parse_assignment() {
        let m = parse_str("x = 10\n");
        assert_eq!(m.body.len(), 1);
        matches!(&m.body[0], Stmt::Assign { target, .. } if target == "x");
    }

    #[test]
    fn parse_if_elif_else() {
        let m = parse_str("if x:\n    pass\nelif y:\n    pass\nelse:\n    pass\n");
        assert_eq!(m.body.len(), 1);
        if let Stmt::If { elif_clauses, else_body, .. } = &m.body[0] {
            assert_eq!(elif_clauses.len(), 1);
            assert_eq!(else_body.len(), 1);
        } else {
            panic!("expected If");
        }
    }

    #[test]
    fn parse_for_loop() {
        let m = parse_str("for i in range(10):\n    print(i)\n");
        assert_eq!(m.body.len(), 1);
        matches!(&m.body[0], Stmt::For { target, .. } if target == "i");
    }

    #[test]
    fn parse_function_def() {
        let m = parse_str("def foo(a, b):\n    return a + b\n");
        assert_eq!(m.body.len(), 1);
        if let Stmt::FunctionDef { name, params, body, .. } = &m.body[0] {
            assert_eq!(name, "foo");
            assert_eq!(params, &["a", "b"]);
            assert_eq!(body.len(), 1);
        } else {
            panic!("expected FunctionDef");
        }
    }

    #[test]
    fn parse_binary_ops() {
        let m = parse_str("x = 1 + 2 * 3\n");
        if let Stmt::Assign { value, .. } = &m.body[0] {
            // Should be Add(1, Mul(2, 3)) due to precedence
            if let Expr::BinOp { op: BinOp::Add, right, .. } = value {
                matches!(right.as_ref(), Expr::BinOp { op: BinOp::Mul, .. });
            } else {
                panic!("expected BinOp::Add");
            }
        }
    }

    #[test]
    fn parse_comparison() {
        let m = parse_str("x = a == b\n");
        if let Stmt::Assign { value, .. } = &m.body[0] {
            matches!(value, Expr::Compare { .. });
        }
    }

    #[test]
    fn parse_bool_ops() {
        let m = parse_str("x = a and b or c\n");
        // Should be Or(And(a, b), c)
        if let Stmt::Assign { value, .. } = &m.body[0] {
            matches!(value, Expr::BoolOp { op: BoolOpKind::Or, .. });
        }
    }

    #[test]
    fn parse_nested_calls() {
        let m = parse_str("print(len(x))\n");
        assert_eq!(m.body.len(), 1);
        if let Stmt::ExprStmt { expr, .. } = &m.body[0] {
            if let Expr::Call { func, args, .. } = expr {
                matches!(func.as_ref(), Expr::Name { id, .. } if id == "print");
                assert_eq!(args.len(), 1);
            } else {
                panic!("expected Call");
            }
        }
    }
}
