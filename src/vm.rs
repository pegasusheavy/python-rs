/// Stackless bytecode VM with explicit frame stack.
use crate::builtins;
use crate::bytecode::{self, CodeObject, op};
use crate::error::PythonError;
use crate::object::{HeapObject, Value};
use std::collections::HashMap;

const MAX_STACK: usize = 256;
const MAX_LOCALS: usize = 128;

/// A single execution frame.
struct Frame {
    code_index: usize,
    ip: usize,
    stack: [Value; MAX_STACK],
    sp: usize,
    locals: [Value; MAX_LOCALS],
}

impl Frame {
    fn new(code_index: usize) -> Self {
        Self {
            code_index,
            ip: 0,
            // SAFETY: Value is Copy and none() is a valid bit pattern for initialization.
            stack: [Value::none(); MAX_STACK],
            sp: 0,
            locals: [Value::none(); MAX_LOCALS],
        }
    }

    fn push(&mut self, val: Value) {
        // SAFETY: sp is always < MAX_STACK when correctly balanced bytecode is executed.
        // The compiler ensures push/pop balance and MAX_STACK=256 is sufficient for
        // expression evaluation depth in Phase 1.
        unsafe {
            *self.stack.get_unchecked_mut(self.sp) = val;
        }
        self.sp += 1;
    }

    fn pop(&mut self) -> Value {
        self.sp -= 1;
        // SAFETY: sp was > 0 before decrement (balanced bytecode).
        unsafe { *self.stack.get_unchecked(self.sp) }
    }

    fn peek(&self) -> Value {
        // SAFETY: sp > 0 (at least one value on stack).
        unsafe { *self.stack.get_unchecked(self.sp - 1) }
    }
}

/// The virtual machine.
pub struct VM {
    frames: Vec<Frame>,
    code_objects: Vec<CodeObject>,
    globals: HashMap<String, Value>,
    pub heap: Vec<HeapObject>,
    pub output: Vec<String>,
}

impl VM {
    /// Create a new VM with compiled code objects and initial heap.
    pub fn new(code_objects: Vec<CodeObject>, heap: Vec<HeapObject>) -> Self {
        let mut vm = Self {
            frames: Vec::with_capacity(64),
            code_objects,
            globals: HashMap::new(),
            heap,
            output: Vec::new(),
        };
        builtins::register_builtins(&mut vm.globals, &mut vm.heap);
        vm
    }

    /// Run the main (first) code object.
    pub fn run(&mut self) -> Result<(), PythonError> {
        self.frames.push(Frame::new(0));
        self.execute()
    }

    fn execute(&mut self) -> Result<(), PythonError> {
        loop {
            let frame_idx = self.frames.len() - 1;

            // Read instruction — borrow frame briefly, extract all needed data, then drop borrow.
            let (instr, line, code_index) = {
                let frame = &self.frames[frame_idx];
                let code = &self.code_objects[frame.code_index];
                if frame.ip >= code.instructions.len() {
                    return Err(PythonError::runtime("instruction pointer out of bounds", 0));
                }
                // SAFETY: ip is bounds-checked above.
                let instr = unsafe { *code.instructions.get_unchecked(frame.ip) };
                let line = unsafe { *code.line_table.get_unchecked(frame.ip) };
                (instr, line, frame.code_index)
            };

            self.frames[frame_idx].ip += 1;

            let opcode = bytecode::decode_op(instr);
            let operand = bytecode::decode_operand(instr);

            match opcode {
                op::LOAD_CONST => {
                    // SAFETY: operand is a valid constant index (set by compiler).
                    let val = self.code_objects[code_index].constants[operand as usize];
                    self.frames[frame_idx].push(val);
                }
                op::LOAD_FAST => {
                    // SAFETY: operand is a valid local index (set by compiler).
                    let val = unsafe { *self.frames[frame_idx].locals.get_unchecked(operand as usize) };
                    self.frames[frame_idx].push(val);
                }
                op::STORE_FAST => {
                    let val = self.frames[frame_idx].pop();
                    // SAFETY: operand is a valid local index (set by compiler).
                    unsafe { *self.frames[frame_idx].locals.get_unchecked_mut(operand as usize) = val; }
                }
                op::LOAD_GLOBAL => {
                    let name = &self.code_objects[code_index].names[operand as usize];
                    if let Some(&val) = self.globals.get(name) {
                        self.frames[frame_idx].push(val);
                    } else {
                        return Err(PythonError::runtime(
                            format!("name '{name}' is not defined"),
                            line,
                        ));
                    }
                }
                op::STORE_GLOBAL => {
                    let val = self.frames[frame_idx].pop();
                    let name = self.code_objects[code_index].names[operand as usize].clone();
                    self.globals.insert(name, val);
                }
                op::ADD => {
                    let right = self.frames[frame_idx].pop();
                    let left = self.frames[frame_idx].pop();
                    let result = binary_add(left, right, &mut self.heap, line)?;
                    self.frames[frame_idx].push(result);
                }
                op::SUB => {
                    let right = self.frames[frame_idx].pop();
                    let left = self.frames[frame_idx].pop();
                    let result = binary_arith(left, right, line, |a, b| a - b, |a, b| a - b)?;
                    self.frames[frame_idx].push(result);
                }
                op::MUL => {
                    let right = self.frames[frame_idx].pop();
                    let left = self.frames[frame_idx].pop();
                    let result = binary_mul(left, right, line)?;
                    self.frames[frame_idx].push(result);
                }
                op::DIV => {
                    let right = self.frames[frame_idx].pop();
                    let left = self.frames[frame_idx].pop();
                    let result = binary_div(left, right, line)?;
                    self.frames[frame_idx].push(result);
                }
                op::FLOOR_DIV => {
                    let right = self.frames[frame_idx].pop();
                    let left = self.frames[frame_idx].pop();
                    let result = binary_floor_div(left, right, line)?;
                    self.frames[frame_idx].push(result);
                }
                op::MOD => {
                    let right = self.frames[frame_idx].pop();
                    let left = self.frames[frame_idx].pop();
                    let result = binary_mod(left, right, line)?;
                    self.frames[frame_idx].push(result);
                }
                op::POW => {
                    let right = self.frames[frame_idx].pop();
                    let left = self.frames[frame_idx].pop();
                    let result = binary_pow(left, right, line)?;
                    self.frames[frame_idx].push(result);
                }
                op::UNARY_NEG => {
                    let val = self.frames[frame_idx].pop();
                    let result = if let Some(i) = val.as_int() {
                        Value::int(-i)
                    } else if let Some(f) = val.as_float() {
                        Value::float(-f)
                    } else {
                        return Err(PythonError::runtime("bad operand for unary -", line));
                    };
                    self.frames[frame_idx].push(result);
                }
                op::UNARY_NOT => {
                    let val = self.frames[frame_idx].pop();
                    let truthy = is_truthy(val, &self.heap);
                    self.frames[frame_idx].push(Value::bool_val(!truthy));
                }
                op::COMPARE_EQ => {
                    let right = self.frames[frame_idx].pop();
                    let left = self.frames[frame_idx].pop();
                    let r = compare(left, right, &self.heap, |a, b| a == b, |a, b| a == b);
                    self.frames[frame_idx].push(Value::bool_val(r));
                }
                op::COMPARE_NE => {
                    let right = self.frames[frame_idx].pop();
                    let left = self.frames[frame_idx].pop();
                    let r = compare(left, right, &self.heap, |a, b| a != b, |a, b| a != b);
                    self.frames[frame_idx].push(Value::bool_val(r));
                }
                op::COMPARE_LT => {
                    let right = self.frames[frame_idx].pop();
                    let left = self.frames[frame_idx].pop();
                    let r = compare(left, right, &self.heap, |a, b| a < b, |a, b| a < b);
                    self.frames[frame_idx].push(Value::bool_val(r));
                }
                op::COMPARE_LE => {
                    let right = self.frames[frame_idx].pop();
                    let left = self.frames[frame_idx].pop();
                    let r = compare(left, right, &self.heap, |a, b| a <= b, |a, b| a <= b);
                    self.frames[frame_idx].push(Value::bool_val(r));
                }
                op::COMPARE_GT => {
                    let right = self.frames[frame_idx].pop();
                    let left = self.frames[frame_idx].pop();
                    let r = compare(left, right, &self.heap, |a, b| a > b, |a, b| a > b);
                    self.frames[frame_idx].push(Value::bool_val(r));
                }
                op::COMPARE_GE => {
                    let right = self.frames[frame_idx].pop();
                    let left = self.frames[frame_idx].pop();
                    let r = compare(left, right, &self.heap, |a, b| a >= b, |a, b| a >= b);
                    self.frames[frame_idx].push(Value::bool_val(r));
                }
                op::JUMP => {
                    self.frames[frame_idx].ip = operand as usize;
                }
                op::JUMP_IF_FALSE => {
                    let val = self.frames[frame_idx].pop();
                    if !is_truthy(val, &self.heap) {
                        self.frames[frame_idx].ip = operand as usize;
                    }
                }
                op::JUMP_IF_TRUE => {
                    let val = self.frames[frame_idx].pop();
                    if is_truthy(val, &self.heap) {
                        self.frames[frame_idx].ip = operand as usize;
                    }
                }
                op::CALL_FUNCTION => {
                    let argc = operand as usize;

                    // Pop args then function from the stack
                    let mut args = Vec::with_capacity(argc);
                    for _ in 0..argc {
                        args.push(self.frames[frame_idx].pop());
                    }
                    args.reverse();
                    let func_val = self.frames[frame_idx].pop();

                    if let Some(heap_idx) = func_val.as_builtin_ref() {
                        let id = if let HeapObject::BuiltinFn { id, .. } = &self.heap[heap_idx] {
                            *id
                        } else {
                            return Err(PythonError::runtime("not a callable", line));
                        };
                        let result = builtins::call_builtin(
                            id,
                            &args,
                            &mut self.heap,
                            &mut self.output,
                        )?;
                        self.frames[frame_idx].push(result);
                    } else if let Some(heap_idx) = func_val.as_func_ref() {
                        let (func_code_index, arity) = if let HeapObject::Function { code_index, arity, .. } = &self.heap[heap_idx] {
                            (*code_index, *arity as usize)
                        } else {
                            return Err(PythonError::runtime("not a callable", line));
                        };

                        if argc != arity {
                            let name = if let HeapObject::Function { name, .. } = &self.heap[heap_idx] {
                                name.clone()
                            } else {
                                "???".to_string()
                            };
                            return Err(PythonError::runtime(
                                format!("{name}() takes {arity} argument(s) but {argc} were given"),
                                line,
                            ));
                        }

                        let mut new_frame = Frame::new(func_code_index);
                        for (i, arg) in args.iter().enumerate() {
                            new_frame.locals[i] = *arg;
                        }
                        self.frames.push(new_frame);
                        continue;
                    } else {
                        return Err(PythonError::runtime(
                            format!("'{}' is not callable", func_val.display(&self.heap)),
                            line,
                        ));
                    }
                }
                op::RETURN_VALUE => {
                    let return_val = self.frames[frame_idx].pop();
                    self.frames.pop();

                    if self.frames.is_empty() {
                        return Ok(());
                    }

                    let caller = self.frames.last_mut().unwrap();
                    caller.push(return_val);
                    continue;
                }
                op::MAKE_FUNCTION => {
                    let code_idx_val = self.code_objects[code_index].constants[operand as usize];
                    let func_code_index = code_idx_val.as_int().unwrap() as usize;
                    let func_name = self.code_objects[func_code_index].name.clone();
                    let arity = self.code_objects[func_code_index].num_params as u8;

                    let heap_idx = self.heap.len();
                    self.heap.push(HeapObject::Function {
                        name: func_name,
                        code_index: func_code_index,
                        arity,
                    });
                    self.frames[frame_idx].push(Value::func_ref(heap_idx));
                }
                op::GET_ITER => {
                    // range() already returns a RangeIter — pass through.
                }
                op::FOR_ITER => {
                    let iter_val = self.frames[frame_idx].pop();

                    if let Some(heap_idx) = iter_val.as_range_ref() {
                        let (current, stop, step) = if let HeapObject::RangeIter { current, stop, step } = &self.heap[heap_idx] {
                            (*current, *stop, *step)
                        } else {
                            return Err(PythonError::runtime("expected iterator", line));
                        };

                        let exhausted = if step > 0 { current >= stop } else { current <= stop };

                        if exhausted {
                            self.frames[frame_idx].ip = operand as usize;
                        } else {
                            self.frames[frame_idx].push(Value::int(current));
                            if let HeapObject::RangeIter { current: c, .. } = &mut self.heap[heap_idx] {
                                *c = current + step;
                            }
                        }
                    } else {
                        return Err(PythonError::runtime("expected iterator", line));
                    }
                }
                op::POP_TOP => {
                    self.frames[frame_idx].pop();
                }
                op::DUP_TOP => {
                    let val = self.frames[frame_idx].peek();
                    self.frames[frame_idx].push(val);
                }
                op::BUILD_LIST => {
                    let count = operand as usize;
                    let mut elements = Vec::with_capacity(count);
                    for _ in 0..count {
                        elements.push(self.frames[frame_idx].pop());
                    }
                    elements.reverse();
                    let heap_idx = self.heap.len();
                    self.heap.push(HeapObject::List(elements));
                    self.frames[frame_idx].push(Value::list_ref(heap_idx));
                }
                op::LIST_APPEND => {
                    let val = self.frames[frame_idx].pop();
                    let list_val = self.frames[frame_idx].pop();
                    if let Some(heap_idx) = list_val.as_list_ref() && let HeapObject::List(items) = &mut self.heap[heap_idx] {
                        items.push(val);
                    }
                    self.frames[frame_idx].push(list_val);
                }
                op::SUBSCRIPT => {
                    let index = self.frames[frame_idx].pop();
                    let obj = self.frames[frame_idx].pop();
                    if let Some(heap_idx) = obj.as_list_ref() {
                        if let Some(i) = index.as_int() {
                            if let HeapObject::List(items) = &self.heap[heap_idx] {
                                let idx = if i < 0 { items.len() as i64 + i } else { i } as usize;
                                if idx < items.len() {
                                    let val = items[idx];
                                    self.frames[frame_idx].push(val);
                                } else {
                                    return Err(PythonError::runtime("list index out of range", line));
                                }
                            }
                        } else {
                            return Err(PythonError::runtime("list indices must be integers", line));
                        }
                    } else if let Some(heap_idx) = obj.as_str_ref() {
                        if let Some(i) = index.as_int() {
                            let s = self.heap[heap_idx].as_str().unwrap();
                            let len = s.len() as i64;
                            let idx = if i < 0 { len + i } else { i } as usize;
                            if idx < s.len() {
                                let ch: String = s.chars().nth(idx).unwrap().to_string();
                                let new_heap_idx = self.heap.len();
                                self.heap.push(HeapObject::Str(ch.into()));
                                self.frames[frame_idx].push(Value::str_ref(new_heap_idx));
                            } else {
                                return Err(PythonError::runtime("string index out of range", line));
                            }
                        } else {
                            return Err(PythonError::runtime("string indices must be integers", line));
                        }
                    } else {
                        return Err(PythonError::runtime("object is not subscriptable", line));
                    }
                }
                op::HALT => {
                    return Ok(());
                }
                _ => {
                    return Err(PythonError::runtime(
                        format!("unknown opcode {opcode}"),
                        line,
                    ));
                }
            }
        }
    }
}

// --- Free functions for arithmetic/comparison (avoids borrow conflicts) ---

fn is_truthy(val: Value, heap: &[HeapObject]) -> bool {
    if let Some(idx) = val.as_str_ref() && let Some(s) = heap[idx].as_str() {
        return !s.is_empty();
    }
    if let Some(idx) = val.as_list_ref() && let HeapObject::List(items) = &heap[idx] {
        return !items.is_empty();
    }
    val.is_truthy()
}

fn binary_add(left: Value, right: Value, heap: &mut Vec<HeapObject>, line: u32) -> Result<Value, PythonError> {
    if let (Some(a), Some(b)) = (left.as_int(), right.as_int()) {
        return Ok(Value::int(a + b));
    }
    if let (Some(a), Some(b)) = (left.to_f64(), right.to_f64()) && (left.is_float() || right.is_float()) {
        return Ok(Value::float(a + b));
    }
    if let (Some(a_idx), Some(b_idx)) = (left.as_str_ref(), right.as_str_ref()) {
        let a = heap[a_idx].as_str().unwrap().to_string();
        let b = heap[b_idx].as_str().unwrap();
        let result = format!("{a}{b}");
        let heap_idx = heap.len();
        heap.push(HeapObject::Str(result.into()));
        return Ok(Value::str_ref(heap_idx));
    }
    Err(PythonError::runtime("unsupported operand type(s) for +", line))
}

fn binary_arith(
    left: Value,
    right: Value,
    line: u32,
    int_op: impl Fn(i64, i64) -> i64,
    float_op: impl Fn(f64, f64) -> f64,
) -> Result<Value, PythonError> {
    if let (Some(a), Some(b)) = (left.as_int(), right.as_int()) {
        return Ok(Value::int(int_op(a, b)));
    }
    if let (Some(a), Some(b)) = (left.to_f64(), right.to_f64()) && (left.is_float() || right.is_float()) {
        return Ok(Value::float(float_op(a, b)));
    }
    Err(PythonError::runtime("unsupported operand types", line))
}

fn binary_mul(left: Value, right: Value, line: u32) -> Result<Value, PythonError> {
    if let (Some(a), Some(b)) = (left.as_int(), right.as_int()) {
        return Ok(Value::int(a * b));
    }
    if let (Some(a), Some(b)) = (left.to_f64(), right.to_f64()) && (left.is_float() || right.is_float()) {
        return Ok(Value::float(a * b));
    }
    Err(PythonError::runtime("unsupported operand type(s) for *", line))
}

fn binary_div(left: Value, right: Value, line: u32) -> Result<Value, PythonError> {
    let a = left.to_f64().ok_or_else(|| PythonError::runtime("unsupported operand type(s) for /", line))?;
    let b = right.to_f64().ok_or_else(|| PythonError::runtime("unsupported operand type(s) for /", line))?;
    if b == 0.0 {
        return Err(PythonError::runtime("division by zero", line));
    }
    Ok(Value::float(a / b))
}

fn binary_floor_div(left: Value, right: Value, line: u32) -> Result<Value, PythonError> {
    if let (Some(a), Some(b)) = (left.as_int(), right.as_int()) {
        if b == 0 {
            return Err(PythonError::runtime("integer division or modulo by zero", line));
        }
        return Ok(Value::int(a.div_euclid(b)));
    }
    if let (Some(a), Some(b)) = (left.to_f64(), right.to_f64()) {
        if b == 0.0 {
            return Err(PythonError::runtime("float floor division by zero", line));
        }
        return Ok(Value::float((a / b).floor()));
    }
    Err(PythonError::runtime("unsupported operand type(s) for //", line))
}

fn binary_mod(left: Value, right: Value, line: u32) -> Result<Value, PythonError> {
    if let (Some(a), Some(b)) = (left.as_int(), right.as_int()) {
        if b == 0 {
            return Err(PythonError::runtime("integer division or modulo by zero", line));
        }
        return Ok(Value::int(a.rem_euclid(b)));
    }
    if let (Some(a), Some(b)) = (left.to_f64(), right.to_f64()) {
        if b == 0.0 {
            return Err(PythonError::runtime("float modulo by zero", line));
        }
        let result = ((a % b) + b) % b;
        return Ok(Value::float(result));
    }
    Err(PythonError::runtime("unsupported operand type(s) for %", line))
}

fn binary_pow(left: Value, right: Value, line: u32) -> Result<Value, PythonError> {
    if let (Some(a), Some(b)) = (left.as_int(), right.as_int()) {
        if b >= 0 {
            return Ok(Value::int(a.pow(b as u32)));
        } else {
            return Ok(Value::float((a as f64).powi(b as i32)));
        }
    }
    if let (Some(a), Some(b)) = (left.to_f64(), right.to_f64()) {
        return Ok(Value::float(a.powf(b)));
    }
    Err(PythonError::runtime("unsupported operand type(s) for **", line))
}

fn compare(
    left: Value,
    right: Value,
    heap: &[HeapObject],
    int_cmp: impl Fn(i64, i64) -> bool,
    float_cmp: impl Fn(f64, f64) -> bool,
) -> bool {
    if let (Some(a), Some(b)) = (left.as_int(), right.as_int()) {
        int_cmp(a, b)
    } else if let (Some(a), Some(b)) = (left.to_f64(), right.to_f64()) {
        float_cmp(a, b)
    } else if let (Some(a), Some(b)) = (left.as_bool(), right.as_bool()) {
        int_cmp(a as i64, b as i64)
    } else if left.is_none() && right.is_none() {
        true
    } else if let (Some(a_idx), Some(b_idx)) = (left.as_str_ref(), right.as_str_ref()) {
        let a = heap[a_idx].as_str().unwrap();
        let b = heap[b_idx].as_str().unwrap();
        let ord = a.cmp(b) as i64;
        int_cmp(ord, 0)
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler;
    use crate::lexer;
    use crate::parser;

    fn run_and_capture(src: &str) -> Vec<String> {
        let tokens = lexer::tokenize(src).unwrap();
        let module = parser::parse(tokens).unwrap();
        let (code_objects, heap) = compiler::compile(&module).unwrap();
        let mut vm = VM::new(code_objects, heap);
        vm.run().unwrap();
        vm.output
    }

    #[test]
    fn test_hello_world() {
        let output = run_and_capture("print(\"hello world\")\n");
        assert_eq!(output, vec!["hello world"]);
    }

    #[test]
    fn test_arithmetic() {
        let output = run_and_capture("print(2 + 3)\nprint(10 - 4)\nprint(3 * 7)\n");
        assert_eq!(output, vec!["5", "6", "21"]);
    }

    #[test]
    fn test_variables() {
        let output = run_and_capture("x = 10\ny = 3\nprint(x + y)\n");
        assert_eq!(output, vec!["13"]);
    }

    #[test]
    fn test_comparison() {
        let output = run_and_capture("print(10 > 3)\nprint(1 == 2)\n");
        assert_eq!(output, vec!["True", "False"]);
    }

    #[test]
    fn test_if_else() {
        let output = run_and_capture("x = 5\nif x > 3:\n    print(\"yes\")\nelse:\n    print(\"no\")\n");
        assert_eq!(output, vec!["yes"]);
    }

    #[test]
    fn test_for_range() {
        let output = run_and_capture("for i in range(3):\n    print(i)\n");
        assert_eq!(output, vec!["0", "1", "2"]);
    }

    #[test]
    fn test_function() {
        let output = run_and_capture("def add(a, b):\n    return a + b\nprint(add(3, 4))\n");
        assert_eq!(output, vec!["7"]);
    }

    #[test]
    fn test_while_loop() {
        let output = run_and_capture("x = 0\nwhile x < 3:\n    print(x)\n    x += 1\n");
        assert_eq!(output, vec!["0", "1", "2"]);
    }

    #[test]
    fn test_nested_function_calls() {
        let src = "def double(x):\n    return x * 2\ndef add_one(x):\n    return x + 1\nprint(add_one(double(5)))\n";
        let output = run_and_capture(src);
        assert_eq!(output, vec!["11"]);
    }

    #[test]
    fn test_fizzbuzz() {
        let src = r#"for i in range(1, 21):
    if i % 15 == 0:
        print("FizzBuzz")
    elif i % 3 == 0:
        print("Fizz")
    elif i % 5 == 0:
        print("Buzz")
    else:
        print(i)
"#;
        let output = run_and_capture(src);
        let expected = vec![
            "1", "2", "Fizz", "4", "Buzz", "Fizz", "7", "8", "Fizz", "Buzz",
            "11", "Fizz", "13", "14", "FizzBuzz", "16", "17", "Fizz", "19", "Buzz",
        ];
        assert_eq!(output, expected);
    }

    #[test]
    fn test_fibonacci() {
        let src = r#"def fib(n):
    a = 0
    b = 1
    for i in range(n):
        temp = a
        a = b
        b = temp + b
    return a
print(fib(10))
"#;
        let output = run_and_capture(src);
        assert_eq!(output, vec!["55"]);
    }

    #[test]
    fn test_full_target_script() {
        let src = r#"x = 10
y = 3
print(x + y)
print(x > y)

for i in range(1, 21):
    if i % 15 == 0:
        print("FizzBuzz")
    elif i % 3 == 0:
        print("Fizz")
    elif i % 5 == 0:
        print("Buzz")
    else:
        print(i)

def fib(n):
    a = 0
    b = 1
    for i in range(n):
        temp = a
        a = b
        b = temp + b
    return a

print(fib(10))
"#;
        let output = run_and_capture(src);
        assert_eq!(output[0], "13");
        assert_eq!(output[1], "True");
        let fizzbuzz = &output[2..22];
        assert_eq!(fizzbuzz, &[
            "1", "2", "Fizz", "4", "Buzz", "Fizz", "7", "8", "Fizz", "Buzz",
            "11", "Fizz", "13", "14", "FizzBuzz", "16", "17", "Fizz", "19", "Buzz",
        ]);
        assert_eq!(output[22], "55");
    }
}
