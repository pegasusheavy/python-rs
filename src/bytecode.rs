/// Bytecode instruction set and code objects.
use crate::object::Value;

/// Opcode constants (upper 8 bits of a u32 instruction).
pub mod op {
    pub const LOAD_CONST: u8 = 0;
    pub const LOAD_FAST: u8 = 1;
    pub const STORE_FAST: u8 = 2;
    pub const LOAD_GLOBAL: u8 = 3;
    pub const STORE_GLOBAL: u8 = 4;
    pub const ADD: u8 = 10;
    pub const SUB: u8 = 11;
    pub const MUL: u8 = 12;
    pub const DIV: u8 = 13;
    pub const FLOOR_DIV: u8 = 14;
    pub const MOD: u8 = 15;
    pub const POW: u8 = 16;
    pub const UNARY_NEG: u8 = 20;
    pub const UNARY_NOT: u8 = 21;
    pub const COMPARE_EQ: u8 = 30;
    pub const COMPARE_NE: u8 = 31;
    pub const COMPARE_LT: u8 = 32;
    pub const COMPARE_LE: u8 = 33;
    pub const COMPARE_GT: u8 = 34;
    pub const COMPARE_GE: u8 = 35;
    pub const JUMP: u8 = 40;
    pub const JUMP_IF_FALSE: u8 = 41;
    pub const JUMP_IF_TRUE: u8 = 42;
    pub const CALL_FUNCTION: u8 = 50;
    pub const RETURN_VALUE: u8 = 51;
    pub const MAKE_FUNCTION: u8 = 52;
    pub const GET_ITER: u8 = 60;
    pub const FOR_ITER: u8 = 61;
    pub const POP_TOP: u8 = 70;
    pub const DUP_TOP: u8 = 71;
    pub const HALT: u8 = 255;
    pub const BUILD_LIST: u8 = 80;
    pub const LIST_APPEND: u8 = 81;
    pub const SUBSCRIPT: u8 = 82;
}

/// Encode an instruction: upper 8 bits opcode, lower 24 bits operand.
pub fn encode(opcode: u8, operand: u32) -> u32 {
    ((opcode as u32) << 24) | (operand & 0x00FF_FFFF)
}

/// Decode opcode from instruction.
pub fn decode_op(instr: u32) -> u8 {
    (instr >> 24) as u8
}

/// Decode operand from instruction.
pub fn decode_operand(instr: u32) -> u32 {
    instr & 0x00FF_FFFF
}

/// A compiled code object (one per function + one for module-level).
#[derive(Debug, Clone)]
pub struct CodeObject {
    /// Name of this code object (function name or "<module>").
    pub name: String,
    /// Bytecode instructions.
    pub instructions: Vec<u32>,
    /// Constant pool.
    pub constants: Vec<Value>,
    /// Global variable names (indexed by operand of LOAD_GLOBAL/STORE_GLOBAL).
    pub names: Vec<String>,
    /// Local variable names (indexed by operand of LOAD_FAST/STORE_FAST).
    pub local_names: Vec<String>,
    /// Number of local variable slots.
    pub num_locals: usize,
    /// Number of parameters (first N locals).
    pub num_params: usize,
    /// Line number for each instruction (parallel to instructions vec).
    pub line_table: Vec<u32>,
}

impl CodeObject {
    /// Create a new empty code object.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            instructions: Vec::new(),
            constants: Vec::new(),
            names: Vec::new(),
            local_names: Vec::new(),
            num_locals: 0,
            num_params: 0,
            line_table: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode() {
        let instr = encode(op::LOAD_CONST, 42);
        assert_eq!(decode_op(instr), op::LOAD_CONST);
        assert_eq!(decode_operand(instr), 42);
    }

    #[test]
    fn max_operand() {
        let instr = encode(op::JUMP, 0x00FF_FFFF);
        assert_eq!(decode_operand(instr), 0x00FF_FFFF);
    }

    #[test]
    fn code_object_new() {
        let co = CodeObject::new("<module>");
        assert_eq!(co.name, "<module>");
        assert!(co.instructions.is_empty());
    }
}
