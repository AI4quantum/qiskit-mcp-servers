# Publishing to PyPI

This guide covers how to publish the Qiskit MCP servers to PyPI, both manually and via automated workflows.

## Packages

This repository contains two separate PyPI packages:

1. **qiskit-code-assistant-mcp-server** - MCP server for Qiskit Code Assistant
2. **qiskit-ibm-runtime-mcp-server** - MCP server for IBM Quantum Runtime

## Automated Publishing (Recommended)

### Prerequisites

To create releases using the automated approach, you'll need:

- **Git**: For creating and pushing tags
- **GitHub CLI (`gh`)**: For creating releases from the command line
  - Install from https://cli.github.com/ or via package manager:
    ```bash
    # macOS
    brew install gh

    # Linux
    sudo apt install gh  # Debian/Ubuntu
    sudo dnf install gh  # Fedora

    # Windows
    winget install GitHub.cli
    ```
  - Authenticate with: `gh auth login`

Alternatively, you can create releases manually through the GitHub web interface instead of using the `gh` CLI.

### Setup: Configure Trusted Publishing

**One-time setup** - Configure trusted publishing on PyPI (no API tokens needed):

1. Go to PyPI and create the project (if it doesn't exist):
   - For `qiskit-code-assistant-mcp-server`: https://pypi.org/manage/project/qiskit-code-assistant-mcp-server/settings/publishing/
   - For `qiskit-ibm-runtime-mcp-server`: https://pypi.org/manage/project/qiskit-ibm-runtime-mcp-server/settings/publishing/

2. Add a "trusted publisher" with these settings:
   - **PyPI Project Name**: `qiskit-code-assistant-mcp-server` (or `qiskit-ibm-runtime-mcp-server`)
   - **Owner**: `AI4quantum`
   - **Repository**: `qiskit-mcp-servers`
   - **Workflow name**: `publish-pypi.yml`
   - **Environment name**: (leave blank)

### Publishing via GitHub Releases

The workflow automatically publishes when you create a GitHub release:

#### For Code Assistant Server:
```bash
# Update version in qiskit-code-assistant-mcp-server/pyproject.toml
# Then create and push a tag
git tag -a code-assistant-v0.1.1 -m "Release qiskit-code-assistant-mcp-server v0.1.1"
git push origin code-assistant-v0.1.1

# Create a GitHub release from this tag
gh release create code-assistant-v0.1.1 --title "qiskit-code-assistant-mcp-server v0.1.1" --notes "Release notes here"
```

#### For Runtime Server:
```bash
# Update version in qiskit-ibm-runtime-mcp-server/pyproject.toml
# Then create and push a tag
git tag -a runtime-v0.1.1 -m "Release qiskit-ibm-runtime-mcp-server v0.1.1"
git push origin runtime-v0.1.1

# Create a GitHub release from this tag
gh release create runtime-v0.1.1 --title "qiskit-ibm-runtime-mcp-server v0.1.1" --notes "Release notes here"
```

### Manual Workflow Trigger

You can also trigger publishing manually via GitHub Actions using the CLI:

```bash
# Publish both packages
gh workflow run "Publish to PyPI" -f package=both

# Publish only code-assistant
gh workflow run "Publish to PyPI" -f package=code-assistant

# Publish only runtime
gh workflow run "Publish to PyPI" -f package=runtime
```

Alternatively, you can trigger via the GitHub web interface:

1. Go to **Actions** → **Publish to PyPI**
2. Click **Run workflow**
3. Select which package to publish: `both`, `code-assistant`, or `runtime`

## Manual Publishing

### Prerequisites

**Python version**: Python 3.10 or higher is required (as specified in each mcp-server `pyproject.toml`).

Install build tools:
```bash
pip install build twine
```

Or use `uv` (recommended):
```bash
pip install uv
```

### Step-by-Step Manual Publishing

#### 1. Update Version

Edit the version in `pyproject.toml`:
- **Code Assistant**: `qiskit-code-assistant-mcp-server/pyproject.toml`
- **Runtime**: `qiskit-ibm-runtime-mcp-server/pyproject.toml`

#### 2. Build the Package

**For Code Assistant:**
```bash
cd qiskit-code-assistant-mcp-server

# Build with uv (recommended)
uv build

# Or with build
python -m build
```

**For Runtime:**
```bash
cd qiskit-ibm-runtime-mcp-server

# Build with uv (recommended)
uv build

# Or with build
python -m build
```

This creates `.whl` and `.tar.gz` files in the `dist/` directory.

#### 3. Verify the Build

Check the contents:
```bash
# List files in the wheel
unzip -l dist/*.whl

# Check package metadata
twine check dist/*
```

#### 4. Upload to PyPI

**Test on TestPyPI first (recommended):**
```bash
# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ qiskit-code-assistant-mcp-server
# or
pip install --index-url https://test.pypi.org/simple/ qiskit-ibm-runtime-mcp-server
```

**Upload to production PyPI:**
```bash
# With twine
twine upload dist/*

# Or with uv
uv publish
```

You'll be prompted for your PyPI username and password (or API token).

#### 5. Verify Installation

```bash
# For Code Assistant
pip install qiskit-code-assistant-mcp-server

# For Runtime
pip install qiskit-ibm-runtime-mcp-server
```

## Version Management

### Versioning Strategy

Both packages use **semantic versioning**: `MAJOR.MINOR.PATCH`

- **MAJOR**: Breaking changes to the API
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

### Current Versions

The current version for each package is defined in their respective `pyproject.toml` files:

- **qiskit-code-assistant-mcp-server**: See [qiskit-code-assistant-mcp-server/pyproject.toml](qiskit-code-assistant-mcp-server/pyproject.toml) (search for `version =`)
- **qiskit-ibm-runtime-mcp-server**: See [qiskit-ibm-runtime-mcp-server/pyproject.toml](qiskit-ibm-runtime-mcp-server/pyproject.toml) (search for `version =`)

## Pre-Publication Checklist

Before publishing, ensure:

- [ ] Version number updated in `pyproject.toml`
- [ ] All tests pass: `./run_tests.sh`
- [ ] Code is formatted: `uv run ruff format`
- [ ] Linting passes: `uv run ruff check`
- [ ] Type checking passes: `uv run mypy src`
- [ ] README is up to date
- [ ] CHANGELOG updated (if you have one)
- [ ] Git commit and tag created

## Troubleshooting

### "Package already exists" error

You cannot overwrite a version on PyPI. You must:
1. Increment the version number in `pyproject.toml`
2. Rebuild and upload

### Authentication issues

For manual uploads, create a PyPI API token:
1. Go to https://pypi.org/manage/account/token/
2. Create a token with upload permissions
3. Use `__token__` as username and the token as password

Or configure in `~/.pypirc`:
```ini
[pypi]
username = __token__
password = pypi-YOUR-API-TOKEN-HERE
```

### Build artifacts in wrong location

Make sure you're running build commands from the package directory:
```bash
cd qiskit-code-assistant-mcp-server  # or qiskit-ibm-runtime-mcp-server
uv build
```

## Resources

- [PyPI Publishing Guide](https://packaging.python.org/tutorials/packaging-projects/)
- [Trusted Publishers (PyPI)](https://docs.pypi.org/trusted-publishers/)
- [Semantic Versioning](https://semver.org/)
- [GitHub Actions - PyPI Publish](https://github.com/marketplace/actions/pypi-publish)
