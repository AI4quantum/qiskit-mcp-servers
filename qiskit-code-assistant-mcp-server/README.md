# mcp-qiskit-code-assistant

MCP server for Qiskit Code Assistant


## Components


### Tools

The server implements one tool:
- `qca_completion`: Get completion for a given prompt
  - Takes a "prompt" as a required string argument
  - Connects to a Qiskit Code Assistant service and returns a code completion based on the prompt


## Prerequisites

- Python 3.10 or higher
- [uv](https://astral.sh/uv) package manager (recommended)
- IBM Quantum account and API token
- Access to Qiskit Code Assistant service

## Installation

This project uses [uv](https://astral.sh/uv) for virtual environments and dependencies management. If you don't have `uv` installed, check out the instructions in <https://docs.astral.sh/uv/getting-started/installation/>

### Setting up the Project with uv

1. **Initialize or sync the project**:
   ```bash
   # This will create a virtual environment and install dependencies
   uv sync
   ```

2. **Configure environment variables**:
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env and add your IBM Quantum API token
   # Get your token from: https://quantum.ibm.com/account
   ```

## Quick Start

### Running the Server

```bash
uv run mcp-qiskit-code-assistant
```

The server will start and listen for MCP connections.


### Testing and debugging the server

> _**Note**: to launch the MCP inspector you will need to have [`node` and `npm`](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm)_

1. From a terminal, go into the cloned repo directory

1. Switch to the virtual environment

    ```sh
    source .venv/bin/activate
    ```

1. Run the MCP Inspector:

    ```sh
    npx @modelcontextprotocol/inspector uv run mcp-qiskit-code-assistant
    ```

1. Open your browser to the URL shown in the console message e.g.,

    ```
    MCP Inspector is up and running at http://localhost:5173
    ```

## Testing

This project includes comprehensive unit and integration tests.

### Running Tests

**Quick test run:**
```bash
./run_tests.sh
```

**Manual test commands:**
```bash
# Install test dependencies
uv sync --group dev --group test

# Run all tests
uv run pytest

# Run only unit tests
uv run pytest -m "not integration"

# Run only integration tests
uv run pytest -m "integration"

# Run tests with coverage
uv run pytest --cov=src --cov-report=html

# Run specific test file
uv run pytest tests/test_qca.py -v
```

### Test Structure

- `tests/test_qca.py` - Unit tests for QCA functions
- `tests/test_utils.py` - Unit tests for utility functions  
- `tests/test_constants.py` - Unit tests for configuration
- `tests/test_integration.py` - Integration tests
- `tests/conftest.py` - Test fixtures and configuration

### Test Coverage

The test suite covers:
- ✅ All QCA API interactions
- ✅ Error handling and validation
- ✅ HTTP client management
- ✅ Configuration validation
- ✅ Integration scenarios
- ✅ Resource and tool handlers

## Resources

- [Qiskit Code Assistant](https://docs.quantum.ibm.com/guides/qiskit-code-assistant)
- [Model Context Protocol (MCP)](https://modelcontextprotocol.io/introduction)
- [MCP Inspector](https://github.com/modelcontextprotocol/inspector)
- [BeeAI Framework](https://i-am-bee.github.io/beeai-framework/#/)
