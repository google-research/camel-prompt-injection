# CaMeL: Secure Python Execution for LLM Tool Sandboxing

A fast, safe Python interpreter for sandboxing code execution in LLM tool-calling environments. CaMeL enforces security policies and tracks data provenance through program execution.

**Research paper**: [Defeating Prompt Injections by Design](https://arxiv.org/abs/2503.18813)

Original authors: Edoardo Debenedetti, Ilia Shumailov, Tianqi Fan, Jamie Hayes, Nicholas Carlini, Daniel Fabian, Christoph Kern, Chongyang Shi, Florian Tramèr (Google, Google DeepMind, ETH Zurich)

## Why CaMeL?

When LLMs call tools, they execute untrusted code. CaMeL provides:

- 🔒 **Security by Design**: Intentionally restricted Python subset (no imports, eval, or dangerous operations)
- 📊 **Data Provenance Tracking**: Knows which values came from user input vs. tool responses
- 🛡️ **Security Policies**: Define fine-grained rules like "only allow this operation on user-provided data"
- ⚡ **Fast Execution**: Custom interpreter optimized for tool-calling workloads
- 🐍 **Familiar Syntax**: Standard Python semantics for supported features

## Quick Start

### Installation

```bash
pip install camel-interpreter
```

### Basic Usage

```python
import ast
from camel.interpreter.interpreter import camel_eval, EvalArgs
from camel.interpreter.namespace import Namespace
from camel.security_policy import NoSecurityPolicyEngine
from camel.interpreter.interpreter import MetadataEvalMode

# Code the LLM wants to execute
code = """
prices = [10, 20, 30]
total = sum(prices)
"""

# Evaluate it safely
ast_tree = ast.parse(code)
result, namespace, _, _ = camel_eval(
    ast_tree,
    Namespace.with_builtins(),
    [],
    [],
    EvalArgs(NoSecurityPolicyEngine(), MetadataEvalMode.NORMAL)
)

print(namespace.variables["total"])  # CaMeLInt(60, ...)
```

## What's Supported?

See [SCOPE.md](SCOPE.md) for a complete list of supported Python features.

**Quick summary**:
- ✅ Variables, assignments, arithmetic
- ✅ Lists, dicts, sets, tuples
- ✅ If/elif/else, for loops, comprehensions
- ✅ Function calls and methods
- ✅ Classes and `@dataclass` decorator
- ❌ Imports, eval/exec, try/except
- ❌ While loops, async/await, decorators (except @dataclass)
- ❌ Slicing (`x[1:3]`)

## Security Policies

Define which operations are allowed based on data provenance:

```python
from camel.security_policy import SecurityPolicyEngine

class MyPolicy(SecurityPolicyEngine):
    def check_policy(self, function_name, args, dependencies):
        # Allow user-provided data to be modified
        # Block tool-controlled data manipulation
        pass
```

See [docs/security-policies.md](docs/security-policies.md) for examples.

## Development

### Requirements

- Python 3.10+
- `uv` for dependency management

### Setup

```bash
git clone https://github.com/yourusername/camel-interpreter.git
cd camel-interpreter
uv sync
```

### Running Tests

```bash
uv run pytest tests/
uv run ruff check --fix
uv run pyright
```

### Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the [Apache License 2.0](LICENSE).

## Citation

If you use CaMeL in research, please cite the original paper:

```bibtex
@article{debenedetti2025camel,
  title={CaMeL: Defeating Prompt Injections by Design},
  author={Debenedetti, Edoardo and others},
  journal={arXiv preprint arXiv:2503.18813},
  year={2025}
}
```

## Acknowledgments

This is a community-maintained fork of the original research artifact by Google / Google DeepMind / ETH Zurich.
