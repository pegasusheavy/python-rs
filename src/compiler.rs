/// Compiles AST into bytecode.
use crate::ast::*;
use crate::bytecode::{self, CodeObject, encode, op};
use crate::error::PythonError;
use crate::object::{HeapObject, Value};

/// Compile a module AST into code objects and initial heap objects.
/// Returns (code_objects, heap_objects, main_code_index).
pub fn compile(module: &Module) -> Result<(Vec<CodeObject>, Vec<HeapObject>), PythonError> {
    let mut compiler = Compiler::new();
    compiler.compile_module(module)?;
    Ok((compiler.code_objects, compiler.heap))
}

struct Compiler {
    code_objects: Vec<CodeObject>,
    heap: Vec<HeapObject>,
    /// Stack of code object indices being compiled.
    code_stack: Vec<usize>,
    /// For loop break/continue: (break_patches, continue_target)
    loop_stack: Vec<LoopContext>,
}

struct LoopContext {
    break_patches: Vec<usize>,
    continue_target: usize,
}

impl Compiler {
    fn new() -> Self {
        Self {
            code_objects: Vec::new(),
            heap: Vec::new(),
            code_stack: Vec::new(),
            loop_stack: Vec::new(),
        }
    }

    fn current_code(&mut self) -> &mut CodeObject {
        let idx = *self.code_stack.last().unwrap();
        &mut self.code_objects[idx]
    }

    fn emit(&mut self, opcode: u8, operand: u32, line: u32) {
        let code = self.current_code();
        code.instructions.push(encode(opcode, operand));
        code.line_table.push(line);
    }

    fn current_offset(&self) -> usize {
        let idx = *self.code_stack.last().unwrap();
        self.code_objects[idx].instructions.len()
    }

    fn patch_jump(&mut self, instr_idx: usize) {
        let target = self.current_offset() as u32;
        let code = self.current_code();
        let old = code.instructions[instr_idx];
        let opcode = bytecode::decode_op(old);
        code.instructions[instr_idx] = encode(opcode, target);
    }

    fn add_const(&mut self, val: Value) -> u32 {
        let code = self.current_code();
        // Check if constant already exists
        for (i, c) in code.constants.iter().enumerate() {
            if *c == val {
                return i as u32;
            }
        }
        let idx = code.constants.len();
        code.constants.push(val);
        idx as u32
    }

    fn add_name(&mut self, name: &str) -> u32 {
        let code = self.current_code();
        for (i, n) in code.names.iter().enumerate() {
            if n == name {
                return i as u32;
            }
        }
        let idx = code.names.len();
        code.names.push(name.to_string());
        idx as u32
    }

    fn add_local(&mut self, name: &str) -> u32 {
        let code = self.current_code();
        for (i, n) in code.local_names.iter().enumerate() {
            if n == name {
                return i as u32;
            }
        }
        let idx = code.local_names.len();
        code.local_names.push(name.to_string());
        code.num_locals = code.local_names.len();
        idx as u32
    }

    fn find_local(&self, name: &str) -> Option<u32> {
        let idx = *self.code_stack.last().unwrap();
        let code = &self.code_objects[idx];
        code.local_names.iter().position(|n| n == name).map(|i| i as u32)
    }

    fn is_module_level(&self) -> bool {
        self.code_stack.len() == 1
    }

    fn add_string_const(&mut self, s: &str) -> u32 {
        // Allocate string on heap and return a Value::str_ref constant
        let heap_idx = self.heap.len();
        self.heap.push(HeapObject::Str(s.into()));
        let val = Value::str_ref(heap_idx);
        self.add_const(val)
    }

    fn compile_module(&mut self, module: &Module) -> Result<(), PythonError> {
        let co_idx = self.code_objects.len();
        self.code_objects.push(CodeObject::new("<module>"));
        self.code_stack.push(co_idx);

        for stmt in &module.body {
            self.compile_stmt(stmt)?;
        }

        // Emit HALT at end of module
        self.emit(op::HALT, 0, 0);

        self.code_stack.pop();
        Ok(())
    }

    fn compile_stmt(&mut self, stmt: &Stmt) -> Result<(), PythonError> {
        match stmt {
            Stmt::Assign { target, value, line } => {
                self.compile_expr(value)?;
                if self.is_module_level() {
                    let name_idx = self.add_name(target);
                    self.emit(op::STORE_GLOBAL, name_idx, *line);
                } else {
                    let local_idx = self.add_local(target);
                    self.emit(op::STORE_FAST, local_idx, *line);
                }
            }
            Stmt::AugAssign { target, op, value, line } => {
                // Load current value
                if self.is_module_level() {
                    let name_idx = self.add_name(target);
                    self.emit(bytecode::op::LOAD_GLOBAL, name_idx, *line);
                } else if let Some(local_idx) = self.find_local(target) {
                    self.emit(bytecode::op::LOAD_FAST, local_idx, *line);
                } else {
                    let local_idx = self.add_local(target);
                    self.emit(bytecode::op::LOAD_FAST, local_idx, *line);
                }
                self.compile_expr(value)?;
                let binop = match op {
                    BinOp::Add => bytecode::op::ADD,
                    BinOp::Sub => bytecode::op::SUB,
                    BinOp::Mul => bytecode::op::MUL,
                    BinOp::Div => bytecode::op::DIV,
                    BinOp::FloorDiv => bytecode::op::FLOOR_DIV,
                    BinOp::Mod => bytecode::op::MOD,
                    BinOp::Pow => bytecode::op::POW,
                };
                self.emit(binop, 0, *line);
                if self.is_module_level() {
                    let name_idx = self.add_name(target);
                    self.emit(bytecode::op::STORE_GLOBAL, name_idx, *line);
                } else {
                    let local_idx = self.add_local(target);
                    self.emit(bytecode::op::STORE_FAST, local_idx, *line);
                }
            }
            Stmt::ExprStmt { expr, line } => {
                self.compile_expr(expr)?;
                self.emit(op::POP_TOP, 0, *line);
            }
            Stmt::If { condition, body, elif_clauses, else_body, line } => {
                self.compile_if(condition, body, elif_clauses, else_body, *line)?;
            }
            Stmt::While { condition, body, line } => {
                self.compile_while(condition, body, *line)?;
            }
            Stmt::For { target, iter, body, line } => {
                self.compile_for(target, iter, body, *line)?;
            }
            Stmt::FunctionDef { name, params, body, line } => {
                self.compile_function_def(name, params, body, *line)?;
            }
            Stmt::Return { value, line } => {
                if let Some(val) = value {
                    self.compile_expr(val)?;
                } else {
                    let none_idx = self.add_const(Value::none());
                    self.emit(op::LOAD_CONST, none_idx, *line);
                }
                self.emit(op::RETURN_VALUE, 0, *line);
            }
            Stmt::Pass { .. } => {
                // No-op
            }
            Stmt::Break { line } => {
                // Emit a JUMP with placeholder, to be patched
                let offset = self.current_offset();
                self.emit(op::JUMP, 0, *line);
                if let Some(ctx) = self.loop_stack.last_mut() {
                    ctx.break_patches.push(offset);
                }
            }
            Stmt::Continue { line } => {
                if let Some(ctx) = self.loop_stack.last() {
                    let target = ctx.continue_target as u32;
                    self.emit(op::JUMP, target, *line);
                }
            }
        }
        Ok(())
    }

    fn compile_if(
        &mut self,
        condition: &Expr,
        body: &[Stmt],
        elif_clauses: &[(Expr, Vec<Stmt>)],
        else_body: &[Stmt],
        line: u32,
    ) -> Result<(), PythonError> {
        self.compile_expr(condition)?;
        let jump_false = self.current_offset();
        self.emit(op::JUMP_IF_FALSE, 0, line);

        for stmt in body {
            self.compile_stmt(stmt)?;
        }

        // Collect end-jumps to patch after all branches
        let mut end_jumps = Vec::new();
        let has_more = !elif_clauses.is_empty() || !else_body.is_empty();
        if has_more {
            let end_jump = self.current_offset();
            self.emit(op::JUMP, 0, line);
            end_jumps.push(end_jump);
        }
        self.patch_jump(jump_false);

        for (elif_cond, elif_body) in elif_clauses {
            self.compile_expr(elif_cond)?;
            let elif_jump_false = self.current_offset();
            self.emit(op::JUMP_IF_FALSE, 0, line);

            for stmt in elif_body {
                self.compile_stmt(stmt)?;
            }

            let end_jump = self.current_offset();
            self.emit(op::JUMP, 0, line);
            end_jumps.push(end_jump);
            self.patch_jump(elif_jump_false);
        }

        for stmt in else_body {
            self.compile_stmt(stmt)?;
        }

        for ej in end_jumps {
            self.patch_jump(ej);
        }

        Ok(())
    }

    fn compile_while(&mut self, condition: &Expr, body: &[Stmt], line: u32) -> Result<(), PythonError> {
        let loop_start = self.current_offset();

        self.loop_stack.push(LoopContext {
            break_patches: Vec::new(),
            continue_target: loop_start,
        });

        self.compile_expr(condition)?;
        let exit_jump = self.current_offset();
        self.emit(op::JUMP_IF_FALSE, 0, line);

        for stmt in body {
            self.compile_stmt(stmt)?;
        }

        self.emit(op::JUMP, loop_start as u32, line);
        self.patch_jump(exit_jump);

        let ctx = self.loop_stack.pop().unwrap();
        for bp in ctx.break_patches {
            self.patch_jump(bp);
        }

        Ok(())
    }

    fn compile_for(&mut self, target: &str, iter: &Expr, body: &[Stmt], line: u32) -> Result<(), PythonError> {
        // Compile the iterable and get an iterator
        self.compile_expr(iter)?;
        self.emit(op::GET_ITER, 0, line);

        // Store iterator in a local/global slot
        let iter_name = format!("__iter_{target}__");
        if self.is_module_level() {
            let name_idx = self.add_name(&iter_name);
            self.emit(op::STORE_GLOBAL, name_idx, line);
        } else {
            let local_idx = self.add_local(&iter_name);
            self.emit(op::STORE_FAST, local_idx, line);
        }

        let loop_start = self.current_offset();

        self.loop_stack.push(LoopContext {
            break_patches: Vec::new(),
            continue_target: loop_start,
        });

        // Load iterator and call FOR_ITER
        if self.is_module_level() {
            let name_idx = self.add_name(&iter_name);
            self.emit(op::LOAD_GLOBAL, name_idx, line);
        } else {
            let local_idx = self.find_local(&iter_name).unwrap();
            self.emit(op::LOAD_FAST, local_idx, line);
        }

        let for_iter = self.current_offset();
        self.emit(op::FOR_ITER, 0, line); // placeholder jump target

        // Store current value in target variable
        if self.is_module_level() {
            let name_idx = self.add_name(target);
            self.emit(op::STORE_GLOBAL, name_idx, line);
        } else {
            let local_idx = self.add_local(target);
            self.emit(op::STORE_FAST, local_idx, line);
        }

        for stmt in body {
            self.compile_stmt(stmt)?;
        }

        self.emit(op::JUMP, loop_start as u32, line);
        self.patch_jump(for_iter);

        let ctx = self.loop_stack.pop().unwrap();
        for bp in ctx.break_patches {
            self.patch_jump(bp);
        }

        Ok(())
    }

    fn compile_function_def(
        &mut self,
        name: &str,
        params: &[String],
        body: &[Stmt],
        line: u32,
    ) -> Result<(), PythonError> {
        // Create a new code object for the function
        let func_co_idx = self.code_objects.len();
        let mut func_co = CodeObject::new(name);
        func_co.num_params = params.len();

        // Pre-populate locals with parameters
        for p in params {
            func_co.local_names.push(p.clone());
        }
        func_co.num_locals = params.len();

        // Pre-scan body for assignment targets to determine locals
        self.prescan_locals(&mut func_co, body);

        self.code_objects.push(func_co);
        self.code_stack.push(func_co_idx);

        for stmt in body {
            self.compile_stmt(stmt)?;
        }

        // Implicit return None
        let none_idx = self.add_const(Value::none());
        self.emit(op::LOAD_CONST, none_idx, line);
        self.emit(op::RETURN_VALUE, 0, line);

        self.code_stack.pop();

        // In the outer code: emit MAKE_FUNCTION + STORE_GLOBAL
        let func_idx_const = self.add_const(Value::int(func_co_idx as i64));
        self.emit(op::MAKE_FUNCTION, func_idx_const, line);
        let name_idx = self.add_name(name);
        self.emit(op::STORE_GLOBAL, name_idx, line);

        Ok(())
    }

    fn prescan_locals(&self, co: &mut CodeObject, stmts: &[Stmt]) {
        for stmt in stmts {
            match stmt {
                Stmt::Assign { target, .. } | Stmt::AugAssign { target, .. } => {
                    if !co.local_names.contains(target) {
                        co.local_names.push(target.clone());
                        co.num_locals = co.local_names.len();
                    }
                }
                Stmt::For { target, body, .. } => {
                    if !co.local_names.contains(target) {
                        co.local_names.push(target.clone());
                        co.num_locals = co.local_names.len();
                    }
                    // Also add iter variable
                    let iter_name = format!("__iter_{target}__");
                    if !co.local_names.contains(&iter_name) {
                        co.local_names.push(iter_name);
                        co.num_locals = co.local_names.len();
                    }
                    self.prescan_locals(co, body);
                }
                Stmt::If { body, elif_clauses, else_body, .. } => {
                    self.prescan_locals(co, body);
                    for (_, elif_body) in elif_clauses {
                        self.prescan_locals(co, elif_body);
                    }
                    self.prescan_locals(co, else_body);
                }
                Stmt::While { body, .. } => {
                    self.prescan_locals(co, body);
                }
                _ => {}
            }
        }
    }

    fn compile_expr(&mut self, expr: &Expr) -> Result<(), PythonError> {
        match expr {
            Expr::IntLit { value, line } => {
                let idx = self.add_const(Value::int(*value));
                self.emit(op::LOAD_CONST, idx, *line);
            }
            Expr::FloatLit { value, line } => {
                let idx = self.add_const(Value::float(*value));
                self.emit(op::LOAD_CONST, idx, *line);
            }
            Expr::StringLit { value, line } => {
                let idx = self.add_string_const(value);
                self.emit(op::LOAD_CONST, idx, *line);
            }
            Expr::BoolLit { value, line } => {
                let idx = self.add_const(Value::bool_val(*value));
                self.emit(op::LOAD_CONST, idx, *line);
            }
            Expr::NoneLit { line } => {
                let idx = self.add_const(Value::none());
                self.emit(op::LOAD_CONST, idx, *line);
            }
            Expr::Name { id, line } => {
                if !self.is_module_level() && let Some(local_idx) = self.find_local(id) {
                    self.emit(op::LOAD_FAST, local_idx, *line);
                    return Ok(());
                }
                let name_idx = self.add_name(id);
                self.emit(op::LOAD_GLOBAL, name_idx, *line);
            }
            Expr::BinOp { left, op: binop, right, line } => {
                self.compile_expr(left)?;
                self.compile_expr(right)?;
                let opcode = match binop {
                    BinOp::Add => op::ADD,
                    BinOp::Sub => op::SUB,
                    BinOp::Mul => op::MUL,
                    BinOp::Div => op::DIV,
                    BinOp::FloorDiv => op::FLOOR_DIV,
                    BinOp::Mod => op::MOD,
                    BinOp::Pow => op::POW,
                };
                self.emit(opcode, 0, *line);
            }
            Expr::UnaryOp { op: unop, operand, line } => {
                self.compile_expr(operand)?;
                let opcode = match unop {
                    UnaryOp::Neg => op::UNARY_NEG,
                    UnaryOp::Not => op::UNARY_NOT,
                };
                self.emit(opcode, 0, *line);
            }
            Expr::Compare { left, ops, comparators, line } => {
                // For now, handle single comparison (chained comparisons need more work)
                if ops.len() == 1 {
                    self.compile_expr(left)?;
                    self.compile_expr(&comparators[0])?;
                    let opcode = match ops[0] {
                        CmpOp::Eq => op::COMPARE_EQ,
                        CmpOp::NotEq => op::COMPARE_NE,
                        CmpOp::Lt => op::COMPARE_LT,
                        CmpOp::LtEq => op::COMPARE_LE,
                        CmpOp::Gt => op::COMPARE_GT,
                        CmpOp::GtEq => op::COMPARE_GE,
                    };
                    self.emit(opcode, 0, *line);
                } else {
                    // Chained comparison: a < b < c → (a < b) and (b < c)
                    // Compile as short-circuit chain
                    self.compile_expr(left)?;
                    let mut end_jumps = Vec::new();

                    for (i, (cmp_op, comparator)) in ops.iter().zip(comparators.iter()).enumerate() {
                        if i < ops.len() - 1 {
                            self.emit(op::DUP_TOP, 0, *line);
                        }
                        self.compile_expr(comparator)?;
                        if i < ops.len() - 1 {
                            // We need to handle the intermediate value
                            // For simplicity, just do sequential comparisons with AND
                        }
                        let opcode = match cmp_op {
                            CmpOp::Eq => op::COMPARE_EQ,
                            CmpOp::NotEq => op::COMPARE_NE,
                            CmpOp::Lt => op::COMPARE_LT,
                            CmpOp::LtEq => op::COMPARE_LE,
                            CmpOp::Gt => op::COMPARE_GT,
                            CmpOp::GtEq => op::COMPARE_GE,
                        };
                        self.emit(opcode, 0, *line);

                        if i < ops.len() - 1 {
                            let jump = self.current_offset();
                            self.emit(op::JUMP_IF_FALSE, 0, *line);
                            end_jumps.push(jump);
                        }
                    }

                    for ej in end_jumps {
                        self.patch_jump(ej);
                    }
                }
            }
            Expr::BoolOp { op: boolop, left, right, line } => {
                self.compile_expr(left)?;
                match boolop {
                    BoolOpKind::And => {
                        self.emit(op::DUP_TOP, 0, *line);
                        let jump = self.current_offset();
                        self.emit(op::JUMP_IF_FALSE, 0, *line);
                        self.emit(op::POP_TOP, 0, *line);
                        self.compile_expr(right)?;
                        self.patch_jump(jump);
                    }
                    BoolOpKind::Or => {
                        self.emit(op::DUP_TOP, 0, *line);
                        let jump = self.current_offset();
                        self.emit(op::JUMP_IF_TRUE, 0, *line);
                        self.emit(op::POP_TOP, 0, *line);
                        self.compile_expr(right)?;
                        self.patch_jump(jump);
                    }
                }
            }
            Expr::Call { func, args, line } => {
                self.compile_expr(func)?;
                let argc = args.len();
                for arg in args {
                    self.compile_expr(arg)?;
                }
                self.emit(op::CALL_FUNCTION, argc as u32, *line);
            }
            Expr::Subscript { value, index, line } => {
                self.compile_expr(value)?;
                self.compile_expr(index)?;
                self.emit(op::SUBSCRIPT, 0, *line);
            }
            Expr::List { elements, line } => {
                for elem in elements {
                    self.compile_expr(elem)?;
                }
                self.emit(op::BUILD_LIST, elements.len() as u32, *line);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::tokenize;
    use crate::parser::parse;

    fn compile_src(src: &str) -> (Vec<CodeObject>, Vec<HeapObject>) {
        let tokens = tokenize(src).unwrap();
        let module = parse(tokens).unwrap();
        compile(&module).unwrap()
    }

    #[test]
    fn compile_simple_assignment() {
        let (cos, _) = compile_src("x = 42\n");
        assert_eq!(cos.len(), 1);
        assert!(cos[0].instructions.len() >= 2); // LOAD_CONST + STORE_GLOBAL + HALT
    }

    #[test]
    fn compile_function() {
        let (cos, _) = compile_src("def foo(x):\n    return x\n");
        assert_eq!(cos.len(), 2); // module + foo
        assert_eq!(cos[1].name, "foo");
        assert_eq!(cos[1].num_params, 1);
    }

    #[test]
    fn compile_for_loop() {
        let (cos, _) = compile_src("for i in range(10):\n    print(i)\n");
        assert_eq!(cos.len(), 1);
        // Should contain GET_ITER, FOR_ITER, JUMP
        let ops: Vec<u8> = cos[0].instructions.iter().map(|i| bytecode::decode_op(*i)).collect();
        assert!(ops.contains(&op::GET_ITER));
        assert!(ops.contains(&op::FOR_ITER));
    }
}
