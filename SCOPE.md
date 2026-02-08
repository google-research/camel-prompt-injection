# CaMeL Supported Python Features

CaMeL is a **secure Python interpreter** designed for sandboxed execution of code in LLM tool-calling environments. This document defines what Python features are supported.

## Design Philosophy

CaMeL intentionally supports a **safe subset of Python** rather than full language compatibility. This:
- Keeps the codebase maintainable
- Reduces attack surface
- Makes security guarantees easier to reason about
- Prevents users from writing code that won't work elsewhere

**If a feature isn't listed here, it's unsupported by design.**

---

## ✅ Fully Supported

### Variables & Assignment
- Simple assignment: `x = 1`
- Multiple assignment: `x, y = 1, 2`
- Variable lookup and updates

### Data Types
- **Primitives**: `int`, `float`, `str`, `bool`, `None`
- **Collections**: `list`, `dict`, `set`, `tuple`
- **Type conversions**: `int()`, `float()`, `str()`, `bool()`, `list()`, `dict()`, `set()`, `tuple()`

### Operators
- **Arithmetic**: `+`, `-`, `*`, `/`, `//`, `%`, `**`
- **Comparison**: `==`, `!=`, `<`, `>`, `<=`, `>=`, `in`, `not in`
- **Boolean**: `and`, `or`, `not`
- **Bitwise**: `&`, `|`, `^`, `~`, `<<`, `>>`
- **Membership**: `in`, `not in`

### Control Flow
- **Conditionals**: `if`, `elif`, `else`
- **Loops**: `for item in iterable:` (standard for loops only)
- **Comprehensions**: list, set, dict comprehensions with filters

### Functions & Methods
- Function calls with positional and keyword arguments
- Method calls on objects
- Unpacking arguments: `*args`, `**kwargs`
- Built-in functions: `len()`, `range()`, `enumerate()`, `zip()`, `sorted()`, `reversed()`, `sum()`, `min()`, `max()`, `any()`, `all()`, `print()`, `type()`, `repr()`, `str()`, `abs()`, `round()`

### Classes & Objects
- Class definitions (basic)
- Instance creation
- Instance attributes and methods
- `@dataclass` decorator for data classes
- Class inheritance from `pydantic.BaseModel`

### Attribute & Item Access
- Attribute access: `obj.attr`
- Index access: `list[0]`, `dict["key"]`
- Item iteration within for loops and comprehensions

---

## ❌ Intentionally Unsupported

### Control Flow
- ❌ `while` loops (use `for` instead)
- ❌ `break` and `continue` statements
- ❌ `try`/`except`/`finally` blocks (errors propagate)
- ❌ `match` statements (Python 3.10+)

### Functions & Definitions
- ❌ Function definitions in code (provide via API)
- ❌ Lambda functions
- ❌ Decorators (except `@dataclass`)
- ❌ Generators and `yield`
- ❌ Async/await and coroutines

### Advanced Features
- ❌ Imports (`import`, `from ... import`)
- ❌ Context managers (`with` statements)
- ❌ Slicing: `x[1:3]`, `x[::2]` (use `x[1]`, `x[2]` instead)
- ❌ Global/nonlocal declarations
- ❌ Python's `eval()`, `exec()`, or reflection APIs
- ❌ Module introspection (`getattr`, `setattr`, `delattr` on non-data-class objects)
- ❌ Operator overloading beyond basic types

### Type System
- ❌ Type hints and annotations in code
- ❌ Generic types

---

## Why These Restrictions?

### Security
- No imports → no external code execution
- No exec/eval → no arbitrary code generation
- No try/except → errors are visible and don't hide failures

### Simplicity
- No while loops → easier to reason about termination
- No slicing → reduces cognitive load on users
- No complex decorators → easier to understand code flow

### Sandboxing
- Limited operator support → prevents complex attacks
- Restricted builtins → only what's needed for data transformation
- No global state → each execution is isolated

---

## Error Handling

When code uses unsupported features, CaMeL raises a `SyntaxError` or `TypeError` explaining the limitation. Example:

```python
# This fails:
code = "while True: pass"
# Error: While statements are not supported. Use a for loop instead.

# This fails:
code = "import os"
# Error: You can't import modules. Instead, use what you have been provided...
```

---

## Metadata Tracking

CaMeL uniquely tracks **data provenance** through the interpreter:
- Which values came from user input vs. tool output
- Which operations were performed on data
- Dependencies between values

This enables security policies to make decisions like:
- "Only allow changes to field X if the user provided it"
- "Reject operations mixing user data with untrusted sources"

---

## For Developers

If you need a feature not listed here:

1. **Check if it can be worked around** within the supported set
2. **Open an issue** describing your use case
3. **Consider the security implications** — we may say no

We prioritize:
- **Security over compatibility**
- **Maintainability over features**
- **Clear errors over silent failures**
