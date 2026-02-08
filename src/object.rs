//! NaN-boxed value representation and heap objects.
//!
//! Layout: every Value is a u64. IEEE 754 doubles are stored as-is.
//! Non-float values use the NaN space: when bits match the tag pattern
//! `(bits & 0x7FFC_0000_0000_0000) == 0x7FFC_0000_0000_0000`, the value
//! is tagged. The tag is encoded in sign bit + bits 49:48 (3 bits total).
//!
//! Tags:
//!   0 = Int(i48)       — sign-extended 48-bit integer
//!   1 = Bool           — payload 0 or 1
//!   2 = None           — singleton
//!   3 = Str(heap idx)  — index into heap
//!   4 = List(heap idx)
//!   5 = Function(heap idx)
//!   6 = RangeIter(heap idx)
//!   7 = BuiltinFn(heap idx)

use std::fmt;

/// Quiet NaN with tag bits set — base for all tagged values.
const QNAN: u64 = 0x7FFC_0000_0000_0000;
/// Mask for the 48-bit payload.
const PAYLOAD_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;
/// Mask for tag bits 49:48 (within the NaN space).
const TAG_BITS_MASK: u64 = 0x0003_0000_0000_0000;

/// Tag values (3 bits: sign + bits 49:48).
const TAG_INT: u64 = 0;       // sign=0, bits=00
const TAG_BOOL: u64 = 1;      // sign=0, bits=01
const TAG_NONE: u64 = 2;      // sign=0, bits=10
const TAG_STR: u64 = 3;       // sign=0, bits=11
const TAG_LIST: u64 = 4;      // sign=1, bits=00
const TAG_FUNC: u64 = 5;      // sign=1, bits=01
const TAG_RANGE: u64 = 6;     // sign=1, bits=10
const TAG_BUILTIN: u64 = 7;   // sign=1, bits=11

/// A NaN-boxed Python value — 8 bytes, Copy.
#[derive(Clone, Copy, PartialEq)]
pub struct Value(u64);

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_float() {
            write!(f, "Value(float={})", self.as_float().unwrap())
        } else if self.is_int() {
            write!(f, "Value(int={})", self.as_int().unwrap())
        } else if self.is_bool() {
            write!(f, "Value(bool={})", self.as_bool().unwrap())
        } else if self.is_none() {
            write!(f, "Value(None)")
        } else {
            write!(f, "Value(tagged=0x{:016X})", self.0)
        }
    }
}

impl Value {
    /// Create a float value.
    pub fn float(v: f64) -> Self {
        // SAFETY: f64 and u64 have the same size and alignment.
        let bits = v.to_bits();
        // If the float happens to look like our tag pattern, use canonical NaN.
        if (bits & QNAN) == QNAN {
            // It's a NaN — store canonical NaN to avoid collision with tags.
            // SAFETY: this is a valid f64 bit pattern (quiet NaN).
            Self(0x7FF8_0000_0000_0000)
        } else {
            Self(bits)
        }
    }

    /// Create an integer value (i48 range).
    pub fn int(v: i64) -> Self {
        // Truncate to 48 bits (sign-extended on read).
        let payload = (v as u64) & PAYLOAD_MASK;
        Self(make_tagged(TAG_INT, payload))
    }

    /// Create a boolean value.
    pub fn bool_val(v: bool) -> Self {
        Self(make_tagged(TAG_BOOL, v as u64))
    }

    /// Create the None singleton.
    pub fn none() -> Self {
        Self(make_tagged(TAG_NONE, 0))
    }

    /// Create a string reference (heap index).
    pub fn str_ref(heap_idx: usize) -> Self {
        Self(make_tagged(TAG_STR, heap_idx as u64))
    }

    /// Create a list reference (heap index).
    pub fn list_ref(heap_idx: usize) -> Self {
        Self(make_tagged(TAG_LIST, heap_idx as u64))
    }

    /// Create a function reference (heap index).
    pub fn func_ref(heap_idx: usize) -> Self {
        Self(make_tagged(TAG_FUNC, heap_idx as u64))
    }

    /// Create a range iterator reference (heap index).
    pub fn range_ref(heap_idx: usize) -> Self {
        Self(make_tagged(TAG_RANGE, heap_idx as u64))
    }

    /// Create a builtin function reference (heap index).
    pub fn builtin_ref(heap_idx: usize) -> Self {
        Self(make_tagged(TAG_BUILTIN, heap_idx as u64))
    }

    /// Check if this is a float (not a tagged NaN value).
    pub fn is_float(&self) -> bool {
        (self.0 & QNAN) != QNAN
    }

    /// Check if this is a tagged value.
    fn is_tagged(&self) -> bool {
        (self.0 & QNAN) == QNAN
    }

    /// Extract the 3-bit tag from a tagged value.
    fn tag(&self) -> u64 {
        debug_assert!(self.is_tagged());
        let sign_bit = (self.0 >> 63) << 2; // bit 63 → bit 2 of tag
        let mid = (self.0 & TAG_BITS_MASK) >> 48; // bits 49:48 → bits 1:0
        sign_bit | mid
    }

    /// Check if this value has the given tag.
    fn has_tag(&self, t: u64) -> bool {
        self.is_tagged() && self.tag() == t
    }

    /// Extract the 48-bit payload.
    fn payload(&self) -> u64 {
        self.0 & PAYLOAD_MASK
    }

    /// Check if this is an integer.
    pub fn is_int(&self) -> bool {
        self.has_tag(TAG_INT)
    }

    /// Check if this is a boolean.
    pub fn is_bool(&self) -> bool {
        self.has_tag(TAG_BOOL)
    }

    /// Check if this is None.
    pub fn is_none(&self) -> bool {
        self.has_tag(TAG_NONE)
    }

    /// Check if this is a string reference.
    pub fn is_str(&self) -> bool {
        self.has_tag(TAG_STR)
    }

    /// Check if this is a list reference.
    pub fn is_list(&self) -> bool {
        self.has_tag(TAG_LIST)
    }

    /// Check if this is a function reference.
    pub fn is_func(&self) -> bool {
        self.has_tag(TAG_FUNC)
    }

    /// Check if this is a range iterator reference.
    pub fn is_range(&self) -> bool {
        self.has_tag(TAG_RANGE)
    }

    /// Check if this is a builtin function reference.
    pub fn is_builtin(&self) -> bool {
        self.has_tag(TAG_BUILTIN)
    }

    /// Extract as f64.
    pub fn as_float(&self) -> Option<f64> {
        if self.is_float() {
            // SAFETY: we verified this is a float (not a tagged NaN).
            Some(f64::from_bits(self.0))
        } else {
            None
        }
    }

    /// Extract as i64 (sign-extended from i48).
    pub fn as_int(&self) -> Option<i64> {
        if self.is_int() {
            let raw = self.payload();
            // Sign-extend from 48 bits.
            let shifted = (raw as i64) << 16;
            Some(shifted >> 16)
        } else {
            None
        }
    }

    /// Extract as bool.
    pub fn as_bool(&self) -> Option<bool> {
        if self.is_bool() {
            Some(self.payload() != 0)
        } else {
            None
        }
    }

    /// Extract heap index for string.
    pub fn as_str_ref(&self) -> Option<usize> {
        if self.is_str() { Some(self.payload() as usize) } else { None }
    }

    /// Extract heap index for list.
    pub fn as_list_ref(&self) -> Option<usize> {
        if self.is_list() { Some(self.payload() as usize) } else { None }
    }

    /// Extract heap index for function.
    pub fn as_func_ref(&self) -> Option<usize> {
        if self.is_func() { Some(self.payload() as usize) } else { None }
    }

    /// Extract heap index for range iterator.
    pub fn as_range_ref(&self) -> Option<usize> {
        if self.is_range() { Some(self.payload() as usize) } else { None }
    }

    /// Extract heap index for builtin function.
    pub fn as_builtin_ref(&self) -> Option<usize> {
        if self.is_builtin() { Some(self.payload() as usize) } else { None }
    }

    /// Get a numeric value as f64 (works for int and float).
    pub fn to_f64(self) -> Option<f64> {
        if let Some(f) = self.as_float() {
            Some(f)
        } else {
            self.as_int().map(|i| i as f64)
        }
    }

    /// Python truthiness.
    pub fn is_truthy(&self) -> bool {
        if let Some(b) = self.as_bool() {
            b
        } else if let Some(i) = self.as_int() {
            i != 0
        } else if let Some(f) = self.as_float() {
            f != 0.0
        } else if self.is_none() {
            false
        } else {
            // Heap objects (str, list, etc.) — truthy by default.
            // Actual truthiness for strings/lists checked in VM with heap access.
            true
        }
    }

    /// Display this value using the heap for string/list lookup.
    pub fn display(&self, heap: &[HeapObject]) -> String {
        if let Some(f) = self.as_float() {
            format_float(f)
        } else if let Some(i) = self.as_int() {
            i.to_string()
        } else if let Some(b) = self.as_bool() {
            if b { "True".to_string() } else { "False".to_string() }
        } else if self.is_none() {
            "None".to_string()
        } else if let Some(idx) = self.as_str_ref() {
            heap[idx].as_str().unwrap_or("???").to_string()
        } else if let Some(idx) = self.as_list_ref() {
            if let HeapObject::List(items) = &heap[idx] {
                let parts: Vec<String> = items.iter().map(|v| v.repr(heap)).collect();
                format!("[{}]", parts.join(", "))
            } else {
                "[???]".to_string()
            }
        } else if let Some(idx) = self.as_func_ref() {
            if let HeapObject::Function { name, .. } = &heap[idx] {
                format!("<function {name}>")
            } else {
                "<function>".to_string()
            }
        } else if let Some(idx) = self.as_builtin_ref() {
            if let HeapObject::BuiltinFn { name, .. } = &heap[idx] {
                format!("<built-in function {name}>")
            } else {
                "<builtin>".to_string()
            }
        } else {
            format!("<object 0x{:016X}>", self.0)
        }
    }

    /// Python repr (strings get quotes).
    pub fn repr(&self, heap: &[HeapObject]) -> String {
        if let Some(idx) = self.as_str_ref() {
            let s = heap[idx].as_str().unwrap_or("???");
            format!("'{s}'")
        } else {
            self.display(heap)
        }
    }
}

/// Format a float like Python does.
fn format_float(f: f64) -> String {
    if f.is_infinite() {
        if f > 0.0 { "inf".to_string() } else { "-inf".to_string() }
    } else if f.is_nan() {
        "nan".to_string()
    } else if f == f.trunc() && f.abs() < 1e16 {
        format!("{f:.1}")
    } else {
        format!("{f}")
    }
}

/// Construct a tagged NaN-boxed value from a 3-bit tag and 48-bit payload.
fn make_tagged(tag: u64, payload: u64) -> u64 {
    let sign = ((tag >> 2) & 1) << 63; // bit 2 of tag → bit 63 (sign bit)
    let mid = (tag & 0b011) << 48;     // bits 0-1 of tag → bits 49:48
    QNAN | sign | mid | (payload & PAYLOAD_MASK)
}

/// Heap-allocated Python objects.
#[derive(Debug, Clone)]
pub enum HeapObject {
    /// Immutable string.
    Str(Box<str>),
    /// Mutable list.
    List(Vec<Value>),
    /// User-defined function.
    Function {
        name: String,
        code_index: usize,
        arity: u8,
    },
    /// Range iterator state.
    RangeIter {
        current: i64,
        stop: i64,
        step: i64,
    },
    /// Built-in function.
    BuiltinFn {
        name: String,
        id: BuiltinId,
    },
}

impl HeapObject {
    /// Get as string slice if this is a Str variant.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(s) => Some(s),
            _ => None,
        }
    }
}

/// Identifies which built-in function to dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinId {
    Print,
    Range,
    Len,
    Type,
    Int,
    Str,
    Bool,
    Float,
    Abs,
    Min,
    Max,
    #[allow(dead_code)]
    Append,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int_roundtrip() {
        for v in [0, 1, -1, 42, -42, 100_000, -100_000, (1 << 47) - 1, -(1 << 47)] {
            let val = Value::int(v);
            assert!(val.is_int(), "expected int for {v}");
            assert_eq!(val.as_int(), Some(v), "roundtrip failed for {v}");
        }
    }

    #[test]
    fn float_roundtrip() {
        for v in [0.0, 1.5, -3.14, f64::INFINITY, f64::NEG_INFINITY] {
            let val = Value::float(v);
            assert!(val.is_float(), "expected float for {v}");
            assert_eq!(val.as_float(), Some(v));
        }
    }

    #[test]
    fn nan_becomes_canonical() {
        let val = Value::float(f64::NAN);
        assert!(val.is_float());
        assert!(val.as_float().unwrap().is_nan());
    }

    #[test]
    fn bool_roundtrip() {
        assert_eq!(Value::bool_val(true).as_bool(), Some(true));
        assert_eq!(Value::bool_val(false).as_bool(), Some(false));
        assert!(Value::bool_val(true).is_bool());
    }

    #[test]
    fn none_works() {
        let v = Value::none();
        assert!(v.is_none());
        assert!(!v.is_int());
        assert!(!v.is_float());
    }

    #[test]
    fn heap_refs() {
        let v = Value::str_ref(42);
        assert!(v.is_str());
        assert_eq!(v.as_str_ref(), Some(42));

        let v = Value::list_ref(7);
        assert!(v.is_list());
        assert_eq!(v.as_list_ref(), Some(7));

        let v = Value::func_ref(3);
        assert!(v.is_func());
        assert_eq!(v.as_func_ref(), Some(3));
    }

    #[test]
    fn truthiness() {
        assert!(Value::bool_val(true).is_truthy());
        assert!(!Value::bool_val(false).is_truthy());
        assert!(Value::int(1).is_truthy());
        assert!(!Value::int(0).is_truthy());
        assert!(Value::float(1.0).is_truthy());
        assert!(!Value::float(0.0).is_truthy());
        assert!(!Value::none().is_truthy());
    }

    #[test]
    fn display_values() {
        let heap = vec![HeapObject::Str("hello".into())];
        assert_eq!(Value::int(42).display(&heap), "42");
        assert_eq!(Value::float(3.14).display(&heap), "3.14");
        assert_eq!(Value::bool_val(true).display(&heap), "True");
        assert_eq!(Value::none().display(&heap), "None");
        assert_eq!(Value::str_ref(0).display(&heap), "hello");
    }

    #[test]
    fn tags_dont_collide() {
        let values = vec![
            Value::int(0),
            Value::bool_val(false),
            Value::none(),
            Value::str_ref(0),
            Value::list_ref(0),
            Value::func_ref(0),
            Value::range_ref(0),
            Value::builtin_ref(0),
        ];
        // Each should only match its own type.
        for (i, v) in values.iter().enumerate() {
            let checks = [
                v.is_int(),
                v.is_bool(),
                v.is_none(),
                v.is_str(),
                v.is_list(),
                v.is_func(),
                v.is_range(),
                v.is_builtin(),
            ];
            for (j, &check) in checks.iter().enumerate() {
                if i == j {
                    assert!(check, "tag {i} should match check {j}");
                } else {
                    assert!(!check, "tag {i} should NOT match check {j}");
                }
            }
        }
    }
}
