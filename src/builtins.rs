/// Built-in functions: print, range, len, type, int, str, bool, etc.
use crate::error::PythonError;
use crate::object::{BuiltinId, HeapObject, Value};
use std::collections::HashMap;

/// Register all built-in functions into globals.
pub fn register_builtins(globals: &mut HashMap<String, Value>, heap: &mut Vec<HeapObject>) {
    let builtins = [
        ("print", BuiltinId::Print),
        ("range", BuiltinId::Range),
        ("len", BuiltinId::Len),
        ("type", BuiltinId::Type),
        ("int", BuiltinId::Int),
        ("str", BuiltinId::Str),
        ("bool", BuiltinId::Bool),
        ("float", BuiltinId::Float),
        ("abs", BuiltinId::Abs),
        ("min", BuiltinId::Min),
        ("max", BuiltinId::Max),
    ];

    for (name, id) in builtins {
        let heap_idx = heap.len();
        heap.push(HeapObject::BuiltinFn {
            name: name.to_string(),
            id,
        });
        globals.insert(name.to_string(), Value::builtin_ref(heap_idx));
    }
}

/// Dispatch a built-in function call.
pub fn call_builtin(
    id: BuiltinId,
    args: &[Value],
    heap: &mut Vec<HeapObject>,
    output: &mut Vec<String>,
) -> Result<Value, PythonError> {
    match id {
        BuiltinId::Print => builtin_print(args, heap, output),
        BuiltinId::Range => builtin_range(args, heap),
        BuiltinId::Len => builtin_len(args, heap),
        BuiltinId::Type => builtin_type(args, heap),
        BuiltinId::Int => builtin_int(args, heap),
        BuiltinId::Str => builtin_str(args, heap),
        BuiltinId::Bool => builtin_bool(args),
        BuiltinId::Float => builtin_float(args, heap),
        BuiltinId::Abs => builtin_abs(args),
        BuiltinId::Min => builtin_min(args),
        BuiltinId::Max => builtin_max(args),
        BuiltinId::Append => Err(PythonError::runtime("append is a method, not a function", 0)),
    }
}

fn builtin_print(args: &[Value], heap: &[HeapObject], output: &mut Vec<String>) -> Result<Value, PythonError> {
    let parts: Vec<String> = args.iter().map(|v| v.display(heap)).collect();
    let line = parts.join(" ");
    output.push(line);
    Ok(Value::none())
}

fn builtin_range(args: &[Value], heap: &mut Vec<HeapObject>) -> Result<Value, PythonError> {
    let (start, stop, step) = match args.len() {
        1 => {
            let stop = args[0].as_int().ok_or_else(|| PythonError::runtime("range() integer expected", 0))?;
            (0, stop, 1)
        }
        2 => {
            let start = args[0].as_int().ok_or_else(|| PythonError::runtime("range() integer expected", 0))?;
            let stop = args[1].as_int().ok_or_else(|| PythonError::runtime("range() integer expected", 0))?;
            (start, stop, 1)
        }
        3 => {
            let start = args[0].as_int().ok_or_else(|| PythonError::runtime("range() integer expected", 0))?;
            let stop = args[1].as_int().ok_or_else(|| PythonError::runtime("range() integer expected", 0))?;
            let step = args[2].as_int().ok_or_else(|| PythonError::runtime("range() integer expected", 0))?;
            if step == 0 {
                return Err(PythonError::runtime("range() arg 3 must not be zero", 0));
            }
            (start, stop, step)
        }
        _ => return Err(PythonError::runtime("range() takes 1 to 3 arguments", 0)),
    };

    let heap_idx = heap.len();
    heap.push(HeapObject::RangeIter {
        current: start,
        stop,
        step,
    });
    Ok(Value::range_ref(heap_idx))
}

fn builtin_len(args: &[Value], heap: &[HeapObject]) -> Result<Value, PythonError> {
    if args.len() != 1 {
        return Err(PythonError::runtime("len() takes exactly one argument", 0));
    }
    let val = args[0];
    if let Some(idx) = val.as_str_ref() {
        let s = heap[idx].as_str().unwrap();
        Ok(Value::int(s.len() as i64))
    } else if let Some(idx) = val.as_list_ref() {
        if let HeapObject::List(items) = &heap[idx] {
            Ok(Value::int(items.len() as i64))
        } else {
            Err(PythonError::runtime("object has no len()", 0))
        }
    } else {
        Err(PythonError::runtime("object has no len()", 0))
    }
}

fn builtin_type(args: &[Value], heap: &[HeapObject]) -> Result<Value, PythonError> {
    if args.len() != 1 {
        return Err(PythonError::runtime("type() takes exactly one argument", 0));
    }
    let val = args[0];
    let _type_name = if val.is_int() {
        "<class 'int'>"
    } else if val.is_float() {
        "<class 'float'>"
    } else if val.is_bool() {
        "<class 'bool'>"
    } else if val.is_none() {
        "<class 'NoneType'>"
    } else if val.is_str() {
        "<class 'str'>"
    } else if val.is_list() {
        "<class 'list'>"
    } else if val.is_func() {
        "<class 'function'>"
    } else if val.is_builtin() {
        "<class 'builtin_function_or_method'>"
    } else {
        "<class 'object'>"
    };
    let heap_idx = heap.len();
    // We need mutable access but only have immutable — return a string value
    // Actually we can't push to heap here. Let's just return an int representation.
    // For Phase 1, type() returns a string.
    let _ = heap_idx;
    // Use a workaround: return a pre-existing type string
    Err(PythonError::runtime("type() not fully implemented in Phase 1", 0))
}

fn builtin_int(args: &[Value], heap: &[HeapObject]) -> Result<Value, PythonError> {
    if args.len() != 1 {
        return Err(PythonError::runtime("int() takes exactly one argument", 0));
    }
    let val = args[0];
    if let Some(i) = val.as_int() {
        Ok(Value::int(i))
    } else if let Some(f) = val.as_float() {
        Ok(Value::int(f as i64))
    } else if let Some(b) = val.as_bool() {
        Ok(Value::int(b as i64))
    } else if let Some(idx) = val.as_str_ref() {
        let s = heap[idx].as_str().unwrap();
        let i: i64 = s.trim().parse().map_err(|_| {
            PythonError::runtime(format!("invalid literal for int() with base 10: '{s}'"), 0)
        })?;
        Ok(Value::int(i))
    } else {
        Err(PythonError::runtime("int() argument must be a string or number", 0))
    }
}

fn builtin_str(args: &[Value], heap: &mut Vec<HeapObject>) -> Result<Value, PythonError> {
    if args.len() != 1 {
        return Err(PythonError::runtime("str() takes exactly one argument", 0));
    }
    let s = args[0].display(heap);
    let heap_idx = heap.len();
    heap.push(HeapObject::Str(s.into()));
    Ok(Value::str_ref(heap_idx))
}

fn builtin_bool(args: &[Value]) -> Result<Value, PythonError> {
    if args.len() != 1 {
        return Err(PythonError::runtime("bool() takes exactly one argument", 0));
    }
    Ok(Value::bool_val(args[0].is_truthy()))
}

fn builtin_float(args: &[Value], heap: &[HeapObject]) -> Result<Value, PythonError> {
    if args.len() != 1 {
        return Err(PythonError::runtime("float() takes exactly one argument", 0));
    }
    let val = args[0];
    if let Some(f) = val.as_float() {
        Ok(Value::float(f))
    } else if let Some(i) = val.as_int() {
        Ok(Value::float(i as f64))
    } else if let Some(idx) = val.as_str_ref() {
        let s = heap[idx].as_str().unwrap();
        let f: f64 = s.trim().parse().map_err(|_| {
            PythonError::runtime(format!("could not convert string to float: '{s}'"), 0)
        })?;
        Ok(Value::float(f))
    } else {
        Err(PythonError::runtime("float() argument must be a string or number", 0))
    }
}

fn builtin_abs(args: &[Value]) -> Result<Value, PythonError> {
    if args.len() != 1 {
        return Err(PythonError::runtime("abs() takes exactly one argument", 0));
    }
    let val = args[0];
    if let Some(i) = val.as_int() {
        Ok(Value::int(i.abs()))
    } else if let Some(f) = val.as_float() {
        Ok(Value::float(f.abs()))
    } else {
        Err(PythonError::runtime("bad operand type for abs()", 0))
    }
}

fn builtin_min(args: &[Value]) -> Result<Value, PythonError> {
    if args.len() < 2 {
        return Err(PythonError::runtime("min() requires at least 2 arguments", 0));
    }
    let mut result = args[0];
    for arg in &args[1..] {
        if let (Some(a), Some(b)) = (arg.as_int(), result.as_int()) && a < b {
            result = *arg;
        } else if let (Some(a), Some(b)) = (arg.to_f64(), result.to_f64()) && a < b {
            result = *arg;
        }
    }
    Ok(result)
}

fn builtin_max(args: &[Value]) -> Result<Value, PythonError> {
    if args.len() < 2 {
        return Err(PythonError::runtime("max() requires at least 2 arguments", 0));
    }
    let mut result = args[0];
    for arg in &args[1..] {
        if let (Some(a), Some(b)) = (arg.as_int(), result.as_int()) && a > b {
            result = *arg;
        } else if let (Some(a), Some(b)) = (arg.to_f64(), result.to_f64()) && a > b {
            result = *arg;
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_builtins() {
        let mut globals = HashMap::new();
        let mut heap = Vec::new();
        register_builtins(&mut globals, &mut heap);
        assert!(globals.contains_key("print"));
        assert!(globals.contains_key("range"));
        assert!(globals.contains_key("len"));
    }

    #[test]
    fn test_print() {
        let heap = vec![HeapObject::Str("hello".into())];
        let mut output = Vec::new();
        let result = builtin_print(&[Value::int(42)], &heap, &mut output);
        assert!(result.is_ok());
        assert_eq!(output, vec!["42"]);
    }

    #[test]
    fn test_range() {
        let mut heap = Vec::new();
        let result = builtin_range(&[Value::int(5)], &mut heap).unwrap();
        assert!(result.is_range());
        if let HeapObject::RangeIter { current, stop, step } = &heap[result.as_range_ref().unwrap()] {
            assert_eq!(*current, 0);
            assert_eq!(*stop, 5);
            assert_eq!(*step, 1);
        }
    }

    #[test]
    fn test_range_with_start_stop() {
        let mut heap = Vec::new();
        let result = builtin_range(&[Value::int(1), Value::int(10)], &mut heap).unwrap();
        assert!(result.is_range());
        if let HeapObject::RangeIter { current, stop, .. } = &heap[result.as_range_ref().unwrap()] {
            assert_eq!(*current, 1);
            assert_eq!(*stop, 10);
        }
    }
}
