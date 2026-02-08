# Contributing to CaMeL

Thank you for your interest in contributing to CaMeL! We welcome bug reports, feature requests, and pull requests.

## Code of Conduct

Be respectful and constructive. We're building this together.

## Reporting Bugs

Found a bug? Please open an issue with:

1. **Minimum reproducible example** — code that demonstrates the problem
2. **Expected behavior** — what should happen
3. **Actual behavior** — what does happen
4. **Environment** — Python version, OS, installation method

## Feature Requests

Before requesting a feature, **check** [SCOPE.md](SCOPE.md):
- If it's listed as "Intentionally Unsupported", we probably won't add it
- If you have a strong use case, open an issue describing it

## How to Contribute Code

### 1. Fork & Setup

```bash
git clone https://github.com/yourusername/camel-interpreter.git
cd camel-interpreter
uv sync
```

### 2. Create a Branch

```bash
git checkout -b fix/my-bug-fix
```

### 3. Make Changes & Test

```bash
uv run pytest tests/
uv run ruff check --fix
uv run pyright
```

### 4. Commit & Push

```bash
git commit -m "fix(interpreter): describe your fix"
git push origin fix/my-bug-fix
```

### 5. Submit Pull Request

Reference any related issues: "Fixes #123"

## Guidelines

✅ **We Accept**:
- Bug fixes with tests
- Documentation improvements
- Test coverage improvements
- Performance improvements

❌ **Check First**:
- New unsupported Python features (see [SCOPE.md](SCOPE.md))
- Large refactors
- Breaking API changes

**Questions?** Open an issue to discuss before coding.

