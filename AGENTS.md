# AGENTS.md — python-rs

Minimal **stackless** Python 3 interpreter written in Rust. Proof-of-concept focused on outperforming CPython, with the target of running Pandas.

## Goal

Build a stackless Python 3 engine in Rust capable of **executing Pandas workloads** — `import pandas as pd; df = pd.DataFrame(...)` and common operations (indexing, filtering, groupby, apply). The primary objective is **beating CPython's execution speed** on equivalent programs. Every design decision should favor runtime performance. Correctness is required, but when there are multiple correct approaches, choose the faster one.

### What "Running Pandas" Means Concretely

The PoC is done when this script executes correctly and faster than CPython:

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie", "Alice", "Bob"],
    "score": [85, 92, 78, 95, 88],
    "grade": ["A", "A", "B", "A", "B"],
})

print(df.head())
print(df.describe())
print(df.groupby("name")["score"].mean())
print(df[df["score"] > 80])
print(df.sort_values("score", ascending=False))
df["curved"] = df["score"].apply(lambda x: x + 5)
print(df)
```

This requires: the full class system, import machinery, C extension loading (NumPy + Pandas are C extensions), exception handling, dunder protocols, comprehensions, closures, and most of the Python data model.

## Stackless Design

This interpreter is **stackless** — Python function calls must NOT map to Rust stack frames. This is a hard architectural constraint.

### Why Stackless
- CPython uses recursive C function calls for Python calls, limiting recursion depth and making deep call chains expensive. We avoid this entirely.
- Stackless execution enables flat iteration in the VM loop — no Rust recursion means no stack overflow risk and better cache behavior.
- Critical for Pandas: deep call stacks through decorators, property access, and dunder chains would blow the Rust stack without this.

### How It Works
- The VM maintains an explicit **call stack** of `Frame` objects on the heap (or in a pre-allocated `Vec<Frame>`).
- A `Frame` holds: the bytecode reference, instruction pointer, local variable slots, and the operand stack (or register file).
- `CALL_FUNCTION` pushes a new `Frame` onto the call stack and continues the VM loop — no Rust recursion.
- `RETURN_VALUE` pops the current `Frame`, writes the return value into the caller's frame, and continues — no Rust `return` unwind.
- The VM loop is a single flat `loop { match opcode { ... } }` that switches between frames by updating a `current_frame` index.
- Built-in functions and C extension calls are the only exception — they may be called as Rust functions directly since they don't yield or recurse into Python (unless they call back into Python, which must trampoline through the frame stack).

## Architecture

```
Source → Lexer → Parser → AST → Compiler → Bytecode → VM
                                                       ↑
                                              C Extension API (cpyext)
                                                       ↑
                                              NumPy / Pandas .so
```

### Modules

| Module | File(s) | Responsibility |
|---|---|---|
| Lexer | `src/lexer.rs` | Tokenize Python source into a token stream. Handle INDENT/DEDENT, f-strings, all string prefixes (`b""`, `r""`, `f""`). |
| Parser | `src/parser.rs` | Recursive-descent parser producing an AST. No parser generators. Full Python 3.11+ grammar for the supported subset. |
| AST | `src/ast.rs` | Data types representing the syntax tree — expressions, statements, patterns, comprehensions, decorators, class bodies. |
| Compiler | `src/compiler.rs` | Walk the AST and emit bytecode. Constant folding, peephole optimizations, scope analysis for closures. |
| Bytecode | `src/bytecode.rs` | Instruction set and code object representation. Fixed-width instructions for cache-friendly dispatch. |
| VM | `src/vm.rs` | **Stackless** VM. Single flat dispatch loop with explicit frame stack. Hot loop — see performance section. |
| Objects | `src/object.rs` | Core `PyObject` representation. NaN-boxing for small values, heap objects for the rest. |
| Type System | `src/types/` | Python type implementations — `int`, `float`, `str`, `list`, `dict`, `tuple`, `set`, `bytes`, `type`, `object`, `NoneType`, `bool`. Each in its own file under `src/types/`. |
| Class System | `src/class.rs` | `class` statement, MRO (C3 linearization), `super()`, descriptors, properties, `__slots__`, metaclasses. |
| Dunder Dispatch | `src/dunder.rs` | Operator overloading dispatch — `__add__`, `__getitem__`, `__setitem__`, `__len__`, `__repr__`, `__str__`, `__call__`, `__iter__`, `__next__`, `__enter__`, `__exit__`, `__hash__`, `__eq__`, `__bool__`, etc. |
| Import System | `src/import.rs` | `import`, `from ... import`, relative imports, `__init__.py`, `sys.path` search, `.pyc` caching (optional), finder/loader protocol. |
| Exceptions | `src/exceptions.rs` | `try`/`except`/`finally`/`else`, `raise`, `with` statement, exception hierarchy (`BaseException` → `Exception` → ...), traceback objects. |
| Closures | `src/closure.rs` | Free variable capture, `nonlocal`/`global` declarations, cell objects. |
| Generators | `src/generator.rs` | `yield`, `yield from`, `send()`, `throw()`, `close()`. Generator frames that suspend/resume within the stackless VM. |
| C Extension API | `src/cpyext/` | **CPython stable ABI compatibility layer** — enough of `Python.h` to load NumPy and Pandas `.so` files. See dedicated section below. |
| Builtins | `src/builtins.rs` | Built-in functions and the `builtins` module — `print`, `len`, `range`, `type`, `isinstance`, `issubclass`, `getattr`, `setattr`, `hasattr`, `iter`, `next`, `map`, `filter`, `zip`, `enumerate`, `sorted`, `reversed`, `any`, `all`, `sum`, `min`, `max`, `abs`, `round`, `id`, `hash`, `repr`, `str`, `int`, `float`, `bool`, `list`, `dict`, `tuple`, `set`, `bytes`, `super`, `property`, `staticmethod`, `classmethod`, `open`, `input`, `vars`, `dir`, `globals`, `locals`, `callable`, `chr`, `ord`. |
| Standard Library | `src/stdlib/` | Minimal pure-Python stdlib shims needed by Pandas: `os.path`, `sys`, `io`, `functools`, `itertools`, `collections`, `abc`, `copy`, `warnings`, `re` (stub or binding), `datetime` (stub). |
| Bench | `benches/interpreter.rs` | Criterion 0.8 benchmark suite. |
| Main | `src/main.rs` | CLI entry point — file execution, REPL, module path setup. |

### C Extension API (`src/cpyext/`)

This is the critical path to running Pandas. NumPy and Pandas are C extensions — they ship `.so`/`.pyd` files compiled against CPython's C API. We must provide a compatibility shim.

#### Strategy

Implement a **minimal subset of the CPython stable ABI** (PEP 384) — only the functions NumPy and Pandas actually call. This is the same approach PyPy uses with `cpyext`.

#### Structure

```
src/cpyext/
├── mod.rs           # Public API, ABI symbol table
├── object.rs        # PyObject, Py_INCREF/DECREF, type checking
├── number.rs        # PyLong_*, PyFloat_*, numeric protocol
├── sequence.rs      # PyList_*, PyTuple_*, sequence protocol
├── mapping.rs       # PyDict_*, mapping protocol
├── unicode.rs       # PyUnicode_* string functions
├── type_object.rs   # PyType_*, tp_* slots, type creation
├── module.rs        # PyModule_*, module initialization (multi-phase init PEP 489)
├── err.rs           # PyErr_*, exception handling across the FFI boundary
├── import.rs        # PyImport_*, extension module loading
├── buffer.rs        # Buffer protocol (PEP 3118) — NumPy depends on this heavily
├── memory.rs        # PyMem_*, object allocation
└── capsule.rs       # PyCapsule API — NumPy uses this for inter-extension communication
```

#### Key Constraints
- Load `.so` files with `dlopen`/`dlsym` at runtime.
- Expose our `PyObject` layout behind a C-compatible `#[repr(C)]` wrapper so C extensions can manipulate it.
- Reference counting is **required at the cpyext boundary** even though our internal runtime avoids it — C extensions expect `Py_INCREF`/`Py_DECREF` to work.
- The buffer protocol (PEP 3118) is non-negotiable — NumPy's ndarray is a buffer object.
- `unsafe` is heavily required here. Every function is `extern "C"` and deals with raw pointers.

#### Phased Approach
1. **Phase 1**: Load a trivial C extension (hand-written test `.so` with one function).
2. **Phase 2**: Load NumPy — focus on `ndarray` creation, basic arithmetic, buffer protocol.
3. **Phase 3**: Load Pandas — DataFrame construction, column access, basic operations.

## Performance Strategy

The entire point of this project is to be faster than CPython on the supported subset. These are the key techniques, ordered by impact:

### Object Representation
- **NaN-boxing**: Pack `int` (i48 or tagged i64), `float`, `bool`, and `None` into a single 8-byte value with no heap allocation. Only `str`, `list`, `dict`, `tuple`, and heap objects require pointers. This eliminates CPython's 28-byte `PyLongObject` overhead for integers.
- Use Rust ownership and `Rc<T>` for internal heap objects. Reference counting is required at the cpyext boundary but avoided in pure-Python paths.

### Stackless VM Dispatch
- **Single flat loop**: The entire interpreter runs in one `loop { match opcode { ... } }` — no recursive `execute()` calls. `CALL_FUNCTION` and `RETURN_VALUE` manipulate the frame stack inline.
- **Pre-allocated frame pool**: Allocate a `Vec<Frame>` at startup. `CALL_FUNCTION` grabs the next slot, `RETURN_VALUE` releases it. No heap allocation per call.
- **Computed goto / match dispatch**: Mark the dispatch function `#[inline(never)]` to help the branch predictor. Keep the opcode match arms small so they stay in L1 icache.
- **Register-based VM** (preferred over stack-based): Reduces the number of bytecode instructions per operation. If register allocation is too complex initially, use a stack VM but minimize redundant push/pop sequences in the compiler.
- **Specialized opcodes**: Type-specialized instructions for the hot path. `ADD_INT`, `ADD_FLOAT`, `LOAD_ATTR_CACHED` (inline cache for attribute lookups — critical for Pandas method chains).

### Compiler Optimizations
- **Constant folding**: Evaluate constant expressions at compile time.
- **Peephole optimization**: Eliminate redundant load/store pairs, fold `NOT`+branch into inverted branch.
- **Scope analysis**: Determine local/free/global variables at compile time. Emit indexed `LOAD_FAST`/`STORE_FAST` for locals, `LOAD_DEREF` for closures.
- **Inline caching for attribute access**: Pandas code is method-chain heavy (`df.groupby(...).mean()`). Cache the attribute lookup result keyed by type version — skip the MRO search on repeated access.

### Memory & Allocation
- **Arena allocation**: Bump allocator for objects created during a single function call. Reset on return.
- **String interning**: Intern all identifier strings and short literals.
- **Pre-sized stack**: Allocate the VM stack once at startup with a fixed capacity.
- **Dict optimization**: Compact dict layout (like CPython 3.6+) with key-sharing for instances of the same class. Critical for Pandas — DataFrames are dict-backed.

### Where CPython Is Slow (exploit these)
| CPython bottleneck | python-rs advantage |
|---|---|
| 28+ bytes per int object, heap allocated | NaN-boxed i64, zero allocation |
| Reference counting on every assignment | Rust ownership, Rc only where needed |
| Dictionary-based variable lookup (`LOAD_NAME`) | Indexed local variable slots |
| Generic `BINARY_ADD` with type dispatch every call | Specialized `ADD_INT` / `ADD_FLOAT` opcodes |
| GIL overhead (even single-threaded) | No GIL, no thread synchronization |
| `PyObject *` pointer indirection everywhere | NaN-boxed values inline on the stack |
| Recursive C calls for Python function calls | Stackless — flat loop, explicit frame stack |
| MRO lookup on every attribute access | Inline caching for hot attribute paths |

### Benchmarking with Criterion 0.8

Use [Criterion.rs](https://github.com/bheisler/criterion.rs) **0.8** as the benchmark framework. This gives us statistical rigor, regression detection, and HTML reports out of the box.

#### Setup

Add to `Cargo.toml`:
```toml
[dev-dependencies]
criterion = { version = "0.8", features = ["html_reports"] }

[[bench]]
name = "interpreter"
harness = false
```

#### Benchmark Structure

Benchmarks live in `benches/interpreter.rs`. Each benchmark runs a Python source string through the full pipeline (parse → compile → execute) and measures execution time.

```
benches/
├── interpreter.rs       # Criterion benchmark entry point
└── scripts/             # Python source files used by benchmarks
    ├── fib_recursive.py
    ├── fib_iterative.py
    ├── loop_sum.py
    ├── string_concat.py
    ├── function_calls.py
    ├── nested_loops.py
    ├── list_operations.py
    ├── dict_operations.py
    ├── class_creation.py
    ├── method_dispatch.py
    ├── exception_handling.py
    ├── comprehensions.py
    ├── closure_calls.py
    ├── generator_iteration.py
    ├── pandas_dataframe.py  # target benchmark
    └── pandas_groupby.py    # target benchmark
```

#### Benchmark Groups

| Group | Scripts | What it measures |
|---|---|---|
| `arithmetic` | `loop_sum.py`, `fib_iterative.py` | Integer arithmetic throughput in tight loops |
| `function_calls` | `fib_recursive.py`, `function_calls.py` | Stackless call/return overhead |
| `control_flow` | `nested_loops.py` | Branch + loop performance |
| `strings` | `string_concat.py` | String allocation and concatenation |
| `collections` | `list_operations.py`, `dict_operations.py` | Container create/access/mutate |
| `object_model` | `class_creation.py`, `method_dispatch.py` | Class instantiation, attribute lookup, dunder dispatch |
| `closures_generators` | `closure_calls.py`, `generator_iteration.py` | Closure variable access, generator suspend/resume |
| `exceptions` | `exception_handling.py` | try/except overhead, exception creation |
| `pandas` | `pandas_dataframe.py`, `pandas_groupby.py` | End-to-end Pandas workloads — the target |
| `full_pipeline` | All scripts | Lex + parse + compile + execute end-to-end |

#### Benchmark Requirements
- Every benchmark script must also be valid CPython 3 — use the companion `bench/cpython_compare.sh` script to run the same files under `python3` with `time` for manual cross-interpreter comparison.
- Use `criterion::black_box` on inputs and outputs to prevent dead-code elimination.
- Benchmark both the **VM execution only** (pre-compiled bytecode) and the **full pipeline** (source → result) as separate groups.
- Run with: `cargo bench` (generates HTML reports in `target/criterion/`).
- CI must run `cargo bench -- --no-plot` to catch regressions without generating reports.
- Target: **5-20x faster than CPython** on pure-Python loops, **2-5x** on string-heavy workloads, **1.5-3x** on Pandas operations (constrained by C extension call overhead).
- Track regressions: if a change makes any Criterion benchmark regress beyond the noise threshold, it must be justified.

## PoC Language Subset

Everything below is **required** to run Pandas. Nothing optional.

### Types
- `int` (Rust `i64` packed via NaN-boxing, no big-int)
- `float` (Rust `f64`, native in NaN-box)
- `bool` (`True`, `False`, NaN-boxed tag)
- `str` (immutable, interned where possible, `Rc<str>` on heap)
- `bytes` / `bytearray`
- `None` (NaN-boxed singleton tag)
- `list` (mutable, heterogeneous)
- `tuple` (immutable, heterogeneous)
- `dict` (ordered, compact layout)
- `set` / `frozenset`
- `slice` objects
- `type` (the metaclass)
- `object` (base class)
- `function`, `method`, `builtin_function`
- `module`
- `property`, `staticmethod`, `classmethod`
- `generator`
- `NotImplemented`, `Ellipsis`
- `range`
- `enumerate`, `zip`, `map`, `filter` (lazy iterators)
- `super`
- Exception types: `BaseException`, `Exception`, `TypeError`, `ValueError`, `KeyError`, `IndexError`, `AttributeError`, `StopIteration`, `RuntimeError`, `ImportError`, `NotImplementedError`, `NameError`, `OSError`, `FileNotFoundError`

### Expressions
- Arithmetic: `+`, `-`, `*`, `/`, `//`, `%`, `**`, unary `-`, unary `+`
- Bitwise: `&`, `|`, `^`, `~`, `<<`, `>>`
- Comparison: `==`, `!=`, `<`, `>`, `<=`, `>=`, `is`, `is not`, `in`, `not in`
- Boolean: `and`, `or`, `not`
- Conditional: `x if cond else y`
- String: concatenation, repetition, f-strings (`f"hello {name}"`)
- Subscript: `a[i]`, `a[i:j]`, `a[i:j:k]`, `a[key]`
- Attribute access: `a.b`
- Call: `f(a, b, *args, key=val, **kwargs)`
- Lambda: `lambda x: x + 1`
- Comprehensions: `[x for x in xs if cond]`, `{k: v for ...}`, `{x for ...}`, `(x for x in xs)`
- Starred: `*a` in assignments, calls, and comprehensions
- Walrus: `:=`
- `yield`, `yield from`

### Statements
- Assignment: `x = expr`, `a, b = expr`, `a, *b, c = expr`
- Augmented assignment: `+=`, `-=`, `*=`, `/=`, `//=`, `%=`, `**=`, `&=`, `|=`, `^=`, `<<=`, `>>=`
- `del`
- `print(...)` (it's a function, but used everywhere)
- `if` / `elif` / `else`
- `while` with `break`, `continue`
- `for x in iterable` with `break`, `continue`, `else`
- `def` — full signature: positional, `*args`, keyword-only, `**kwargs`, defaults, annotations (parsed and stored, not enforced)
- `return`
- `class` — single and multiple inheritance, `super()`, metaclass, `__init__`, `__new__`, `@staticmethod`, `@classmethod`, `@property`, decorators, `__slots__`
- `import` / `from ... import ...` / `from . import ...`
- `try` / `except` / `except ... as e` / `finally` / `else`
- `raise` / `raise ... from ...`
- `with` / `with ... as ...` (single and multiple context managers)
- `assert`
- `pass`
- `global`, `nonlocal`
- Decorators: `@decorator` on functions and classes

### Dunder Methods (minimum required)

These are the operator and protocol dunders that must work for Pandas:

| Category | Methods |
|---|---|
| Construction | `__init__`, `__new__`, `__del__` |
| Representation | `__repr__`, `__str__`, `__format__`, `__bytes__` |
| Comparison | `__eq__`, `__ne__`, `__lt__`, `__le__`, `__gt__`, `__ge__` |
| Hashing | `__hash__` |
| Boolean | `__bool__`, `__len__` |
| Arithmetic | `__add__`, `__radd__`, `__iadd__`, `__sub__`, `__rsub__`, `__isub__`, `__mul__`, `__rmul__`, `__imul__`, `__truediv__`, `__rtruediv__`, `__floordiv__`, `__mod__`, `__pow__`, `__neg__`, `__pos__`, `__abs__` |
| Bitwise | `__and__`, `__or__`, `__xor__`, `__invert__`, `__lshift__`, `__rshift__` |
| Container | `__getitem__`, `__setitem__`, `__delitem__`, `__contains__`, `__len__`, `__iter__`, `__next__`, `__reversed__` |
| Attribute | `__getattr__`, `__getattribute__`, `__setattr__`, `__delattr__`, `__dir__` |
| Descriptor | `__get__`, `__set__`, `__delete__`, `__set_name__` |
| Callable | `__call__` |
| Context Manager | `__enter__`, `__exit__` |
| Class | `__init_subclass__`, `__class_getitem__`, `__instancecheck__`, `__subclasscheck__` |
| Iterator/Generator | `__iter__`, `__next__`, `send`, `throw`, `close` |
| Index | `__index__`, `__int__`, `__float__` |

### NOT in scope for PoC
- `async`/`await`, `match`/`case` (PEP 634), annotations enforcement, `__prepare__`, multiple metaclass resolution, `__slots__` with inheritance diamond edge cases, `__init_subclass__` with complex kwargs, structural pattern matching

## Coding Conventions

- Rust edition 2024.
- `unsafe` is allowed in: `src/object.rs` (NaN-boxing), `src/vm.rs` (unchecked stack access in hot loop), and `src/cpyext/` (C FFI — inherently unsafe). Every `unsafe` block must have a `// SAFETY:` comment. No `unsafe` anywhere else.
- External crates allowed: `criterion` 0.8 (dev-dependency for benchmarks), `libc` (for cpyext FFI), `libloading` (for `dlopen`/`dlsym`), and optionally `rustyline` (REPL) and `clap` (CLI args). No other crates.
- All public types and functions get a one-line doc comment.
- Use `thiserror`-style manual error enums — define a `PythonError` enum in `src/error.rs` with variants for `LexError`, `ParseError`, `CompileError`, `RuntimeError`. All error variants carry a line number and traceback context.
- `#[derive(Debug, Clone, PartialEq)]` on AST nodes and tokens.
- Format with `cargo fmt`, lint with `cargo clippy -- -D warnings`.
- Every commit must pass `cargo build` and `cargo test`.

## Testing Strategy

- Unit tests go in each module file behind `#[cfg(test)]`.
- Lexer tests: verify token sequences for representative snippets.
- Parser tests: parse source strings and assert AST structure.
- VM integration tests: run Python source through the full pipeline and assert stdout output.
- Add a `tests/` directory with `.py` files and expected output `.txt` files for end-to-end tests.
- Every test that runs a computation should also pass when executed with `python3` — correctness is non-negotiable.
- **Milestone tests** (must pass in order):
  1. FizzBuzz — basic control flow
  2. Recursive fibonacci — function calls, stackless validation
  3. Class with methods and inheritance — object model
  4. Generator-based iteration — stackless generator frames
  5. `try`/`except` with custom exceptions — exception handling
  6. `import json; json.dumps({"a": 1})` — import system + stdlib
  7. Load a trivial C extension `.so` — cpyext foundation
  8. `import numpy; a = numpy.array([1,2,3]); print(a + a)` — NumPy loading
  9. The Pandas target script above — PoC complete

## Build & Run

```sh
cargo build --release              # always benchmark with release builds
cargo run --release -- script.py   # execute a file
cargo run --release                # REPL (stretch goal)
cargo test                         # all tests
cargo clippy -- -D warnings        # lint
cargo bench                        # Criterion benchmarks (HTML reports in target/criterion/)
cargo bench -- --no-plot           # CI-friendly, no HTML generation
```

Profile-guided optimization for final benchmarks:

```sh
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

## Suggested Implementation Order

### Phase 1 — Core Language (pure Python scripts)
1. `src/error.rs` — error types with traceback support
2. `src/object.rs` — NaN-boxed `Value` type
3. `src/lexer.rs` — tokenizer including INDENT/DEDENT, f-strings
4. `src/ast.rs` — full AST node types
5. `src/parser.rs` — recursive-descent parser (full grammar)
6. `src/bytecode.rs` — instruction set with specialized opcodes
7. `src/compiler.rs` — AST → bytecode with constant folding, scope analysis
8. `src/vm.rs` — stackless bytecode execution
9. `src/builtins.rs` — built-in functions
10. `src/types/` — `int`, `float`, `str`, `bool`, `list`, `tuple`, `dict`, `set`, `bytes`, `slice`, `range`
11. `src/main.rs` — file execution, module path setup

### Phase 2 — Object Model
12. `src/class.rs` — `class` statement, MRO, `super()`, descriptors, properties, metaclasses
13. `src/dunder.rs` — operator overloading dispatch
14. `src/closure.rs` — closures and `nonlocal`/`global`
15. `src/generator.rs` — `yield`, generator frames within the stackless VM
16. `src/exceptions.rs` — `try`/`except`/`finally`, exception hierarchy, traceback

### Phase 3 — Import System & stdlib
17. `src/import.rs` — `import`/`from`, finder/loader protocol, `sys.path`, `__init__.py`
18. `src/stdlib/` — minimal shims: `sys`, `os`, `os.path`, `io`, `functools`, `itertools`, `collections`, `abc`, `copy`, `warnings`, `re`, `datetime`, `json`

### Phase 4 — C Extension Compatibility
19. `src/cpyext/object.rs` — `PyObject` C-compatible wrapper, `Py_INCREF`/`Py_DECREF`
20. `src/cpyext/module.rs` — extension module loading (`dlopen`, multi-phase init)
21. `src/cpyext/number.rs` — `PyLong_*`, `PyFloat_*`
22. `src/cpyext/sequence.rs` — `PyList_*`, `PyTuple_*`
23. `src/cpyext/mapping.rs` — `PyDict_*`
24. `src/cpyext/unicode.rs` — `PyUnicode_*`
25. `src/cpyext/buffer.rs` — buffer protocol (PEP 3118) — NumPy hard dependency
26. `src/cpyext/err.rs` — `PyErr_*`
27. `src/cpyext/type_object.rs` — `PyType_*`, `tp_*` slots
28. `src/cpyext/capsule.rs` — `PyCapsule` (NumPy inter-module communication)

### Phase 5 — Benchmarking & Optimization
29. `benches/` — Criterion 0.8 benchmark suite with CPython comparison scripts
30. Inline caching for attribute lookups
31. Specialized opcodes based on profiling data
32. Arena allocator tuning
33. The Pandas target script passes and benchmarks favorably

### Phase 6 — Stretch
34. REPL with `rustyline`
35. `.pyc` bytecode caching
36. `async`/`await` (if needed by Pandas dependencies)

Each phase should be fully tested before starting the next. Criterion benchmarks should be added as early as Phase 1 to track performance throughout.

## Future TODO — Embeddable Python Engine for Rust

After the PoC is complete, the next major goal is exposing python-rs as a **library crate** that lets Rust programs execute Python scripts directly — an embedded scripting engine, like Lua's C API but for Python.

### Vision

```rust
use python_rs::Runtime;

fn main() {
    let mut rt = Runtime::new();

    // Execute a script
    rt.exec("x = 1 + 2").unwrap();

    // Evaluate an expression and get the result
    let val: i64 = rt.eval("x * 10").unwrap();
    assert_eq!(val, 30);

    // Call a Python function from Rust
    rt.exec("def greet(name): return f'hello {name}'").unwrap();
    let msg: String = rt.call("greet", ("world",)).unwrap();
    assert_eq!(msg, "hello world");

    // Pass Rust data into Python
    rt.set("items", vec![1, 2, 3]).unwrap();
    rt.exec("total = sum(items)").unwrap();
    let total: i64 = rt.get("total").unwrap();
    assert_eq!(total, 6);

    // Run a .py file
    rt.exec_file("script.py").unwrap();
}
```

### Embedding API Surface

| API | Description |
|---|---|
| `Runtime::new()` | Create an isolated interpreter instance with its own global state. |
| `Runtime::with_config(cfg)` | Create with options: memory limits, allowed imports, stdout capture, etc. |
| `rt.exec(source)` | Execute Python statements. Returns `Result<(), PythonError>`. |
| `rt.eval(expr)` | Evaluate a Python expression, return the result converted to a Rust type. |
| `rt.exec_file(path)` | Execute a `.py` file. |
| `rt.call(name, args)` | Call a Python function by name with Rust arguments. |
| `rt.get::<T>(name)` | Read a Python variable, converting to Rust type `T`. |
| `rt.set(name, value)` | Set a Python variable from a Rust value. |
| `rt.module(name)` | Access or create a Python module to inject Rust functions into. |
| `rt.scope()` | Create a child scope with its own locals but shared globals. |

### Type Conversion Trait

A `FromPython` / `IntoPython` trait pair for zero-copy or minimal-copy conversion between Rust and Python types:

```rust
/// Convert a Python Value to a Rust type.
pub trait FromPython: Sized {
    fn from_python(value: Value) -> Result<Self, PythonError>;
}

/// Convert a Rust type to a Python Value.
pub trait IntoPython {
    fn into_python(self) -> Value;
}
```

Built-in impls for: `i64`, `f64`, `bool`, `String`, `&str`, `Vec<T>`, `HashMap<K, V>`, `Option<T>`, `()`. Derive macro as a stretch goal.

### Exposing Rust Functions to Python

```rust
rt.module("mymod").unwrap()
    .add_function("add", |a: i64, b: i64| -> i64 { a + b })
    .add_function("transform", |s: String| -> String { s.to_uppercase() });

rt.exec("import mymod; print(mymod.add(1, 2))").unwrap();
```

Under the hood this registers a Rust closure as a built-in function callable from Python. Arguments are auto-converted via `FromPython`, return values via `IntoPython`.

### Crate Structure

```
python-rs/
├── Cargo.toml          # [lib] + [[bin]] targets
├── src/
│   ├── lib.rs          # Public API: Runtime, Value, FromPython, IntoPython, PythonError
│   ├── runtime.rs      # Runtime struct, exec/eval/call/get/set implementation
│   ├── convert.rs      # FromPython / IntoPython trait and built-in impls
│   ├── embed.rs        # Module injection, Rust function registration
│   └── ...             # All existing interpreter modules
└── examples/
    ├── hello.rs         # Minimal embedding example
    ├── pandas_from_rust.rs  # Run Pandas from Rust
    └── scripting.rs     # Use Python as a config/scripting language
```

### Key Design Constraints
- **Multiple runtimes**: Each `Runtime` is fully isolated — separate global state, separate frame stacks, separate import caches. No global mutable state.
- **Thread safety**: A single `Runtime` is `!Send` + `!Sync` (single-threaded, like CPython). Multiple runtimes can exist on different threads. A `Send` wrapper with a mutex is provided as opt-in.
- **No `unsafe` in the public API**: All unsafety is internal. The embedding API is fully safe Rust.
- **Error propagation**: Python exceptions convert to `Result<T, PythonError>` at the Rust boundary. Rust panics do not propagate into Python.
- **Memory**: The `Runtime` owns all Python objects. When it drops, everything is freed. No leaked references.
- **Performance**: `exec`/`eval` compile on every call. Provide `rt.compile(source) -> CompiledCode` for repeated execution of the same script without re-parsing.

## Future TODO — Package Manager (`pyrs`)

A built-in package manager that replaces `pip` + `requirements.txt` with a single manifest file and first-class separation of production, dev, and optional dependency groups. Think Cargo for Python.

### Manifest: `pyproject.rs.toml`

One file, no ambiguity. Dependency groups are top-level sections, not ad-hoc extras:

```toml
[package]
name = "my-app"
version = "0.1.0"
python = ">=3.11"

[dependencies]
pandas = ">=2.0,<3"
numpy = ">=1.24"
requests = "~=2.31"

[dev-dependencies]
pytest = ">=7.0"
pytest-cov = "*"
mypy = ">=1.5"
ruff = ">=0.1"

[optional-dependencies.postgres]
psycopg2 = ">=2.9"
sqlalchemy = ">=2.0"

[optional-dependencies.redis]
redis = ">=5.0"

[optional-dependencies.ml]
scikit-learn = ">=1.3"
torch = ">=2.0"
```

### Dependency Groups

| Section | Installed by | Purpose |
|---|---|---|
| `[dependencies]` | `pyrs install` | Production requirements — always installed. |
| `[dev-dependencies]` | `pyrs install --dev` | Test runners, linters, type checkers, formatters. Never shipped to production. |
| `[optional-dependencies.<name>]` | `pyrs install --feature <name>` | Feature-gated extras. Users opt in per deployment. Multiple can be combined: `pyrs install --feature postgres,redis`. |

### Lockfile: `pyrs.lock`

Deterministic, cross-platform lockfile generated from the manifest. Contains resolved versions, hashes, and dependency graph for all groups:

```toml
# Auto-generated by pyrs. Do not edit.
[[package]]
name = "pandas"
version = "2.1.4"
sha256 = "..."
source = "pypi"
group = "dependencies"
requires = ["numpy>=1.24", "python-dateutil>=2.8.2", "pytz>=2020.1"]

[[package]]
name = "pytest"
version = "7.4.3"
sha256 = "..."
source = "pypi"
group = "dev-dependencies"
```

- `pyrs install` reads the lockfile if present (fast, reproducible). Regenerates only when `pyproject.rs.toml` changes.
- `pyrs lock` explicitly regenerates the lockfile.
- The lockfile is committed to version control.

### CLI Commands

```sh
pyrs init                          # create pyproject.rs.toml in current directory
pyrs install                       # install production deps from lockfile
pyrs install --dev                 # install production + dev deps
pyrs install --feature postgres    # install production + optional group
pyrs install --all                 # install everything (production + dev + all optional)
pyrs add pandas                    # add to [dependencies], update lockfile
pyrs add --dev pytest              # add to [dev-dependencies]
pyrs add --feature ml torch        # add to [optional-dependencies.ml]
pyrs remove pandas                 # remove from manifest + lockfile
pyrs lock                          # regenerate lockfile from manifest
pyrs update                        # update all deps to latest compatible versions
pyrs update pandas                 # update a single package
pyrs tree                          # print dependency tree
pyrs tree --why numpy              # show why a transitive dep is included
pyrs run script.py                 # run with the project's installed environment
pyrs run --feature ml train.py     # run with optional group activated
pyrs check                         # verify lockfile matches manifest, all hashes valid
```

### Content-Addressable Store + Symlinks (pnpm model)

Like pnpm, `pyrs` uses a **global content-addressable store** with per-project symlink trees. Packages are never duplicated on disk.

#### How It Works

```
~/.pyrs/store/                              # global store, shared across all projects
├── pandas@2.1.4/                           # one copy per version, ever
│   ├── pandas/
│   └── ...
├── numpy@1.26.2/
├── pytest@7.4.3/
└── ...

my-project/.pyrs/                           # project-local environment
├── site-packages/
│   ├── pandas -> ~/.pyrs/store/pandas@2.1.4/pandas     # symlink
│   ├── numpy -> ~/.pyrs/store/numpy@1.26.2/numpy       # symlink
│   └── ...
└── bin/
    └── python -> <python-rs binary>
```

- **Global store** (`~/.pyrs/store/`): Every package version is downloaded and unpacked exactly once, content-addressed by `name@version`. Shared across every project on the machine.
- **Project symlinks** (`.pyrs/site-packages/`): Each project gets a symlink tree pointing into the global store. `pyrs install` creates/updates symlinks — no file copying.
- **Hardlinks for performance**: On filesystems that support it, use hardlinks instead of symlinks for individual files within packages. Falls back to symlinks on filesystems that don't support cross-device hardlinks.

#### Why This Matters

| Traditional (pip/venv) | pyrs (pnpm model) |
|---|---|
| 10 projects using pandas = 10 full copies on disk | 10 projects using pandas = 1 copy + 10 symlinks |
| `pip install` copies files into each venv | `pyrs install` creates symlinks in milliseconds |
| Updating a shared dependency re-downloads and re-extracts for every project | Updating fetches once to the store, then repoints symlinks |
| A typical ML environment is 2-5 GB (NumPy, Pandas, PyTorch all copied) | Same environment is ~10 MB of symlinks + one shared store |

#### Disk Savings Example

```
# Traditional: 5 projects with pandas+numpy+scipy
5 × ~300 MB = ~1.5 GB

# pyrs: same 5 projects
1 × ~300 MB (store) + 5 × ~4 KB (symlinks) ≈ 300 MB
```

#### Store Management

```sh
pyrs store status                  # show store location, size, package count
pyrs store gc                      # remove packages not referenced by any project
pyrs store gc --dry-run            # preview what gc would remove
pyrs store path                    # print store path
pyrs store path pandas@2.1.4      # print path to a specific package in the store
```

- The store is append-only during installs. Garbage collection (`pyrs store gc`) scans all known project lockfiles and removes unreferenced packages.
- Store location defaults to `~/.pyrs/store/`, configurable via `PYRS_STORE` env var or `~/.pyrs/config.toml`.

#### Isolation Guarantees

A project can only import packages that are in its own symlink tree. Unlike pip where any globally-installed package leaks in, pyrs enforces strict boundaries:

- The import path is `.pyrs/site-packages/` and nothing else (no system site-packages, no user site-packages).
- If a package isn't in `pyproject.rs.toml`, it doesn't get a symlink, and `import` fails — even if it exists in the global store. This is identical to pnpm's strictness: phantom dependencies are impossible.

### Virtual Environments

`pyrs` manages environments automatically — no manual `venv` activation:

- `pyrs install` creates `.pyrs/` in the project root with a symlink tree into the global store.
- `pyrs run` always uses the project environment. No `source activate` ceremony.
- Environments are per-project, never global. No system Python contamination.
- Instant environment creation — symlinking is O(number of deps), not O(total file size).

### Resolution Strategy

- Resolve from PyPI (default registry) or configured private registries.
- Use the same wheel/sdist formats as pip — full compatibility with existing PyPI packages.
- Prefer binary wheels. Fall back to sdist only when no compatible wheel exists.
- Resolution must handle the dependency groups independently: production deps are resolved first, then dev and optional groups are resolved against the production solution (no conflicts between groups).

### Why Not pip/poetry/pdm

| Existing tool | Problem |
|---|---|
| pip + requirements.txt | No dependency groups, no lockfile, no resolution guarantees. `requirements-dev.txt` is convention, not enforcement. |
| Poetry | `pyproject.toml` overload, slow resolver, groups are second-class (`--with`/`--without` flags). |
| pdm | Closest to what we want, but still piggybacks on PEP 621 optional-dependencies which were designed for library extras, not deployment features. |
| uv | Fast, but inherits pip's flat model. No manifest-level group separation. |

`pyrs` treats dependency groups as a first-class concept in the manifest format, not an afterthought bolted onto library metadata.

### Implementation Notes
- Written in Rust as a separate binary in the workspace (`pyrs-pkg/` or similar).
- Uses the same `python-rs` runtime for running `setup.py` / build backends when building sdists.
- Package resolution is a SAT problem — use a DPLL-based resolver or port `resolvelib`'s backtracking algorithm.
- Hash verification on every install (SHA-256 from lockfile vs downloaded artifact).
- Parallel downloads and installs.
