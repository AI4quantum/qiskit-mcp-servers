# Qiskit IBM Runtime MCP Server

A comprehensive Model Context Protocol (MCP) server that provides AI assistants with access to IBM Quantum computing services through Qiskit IBM Runtime. This server enables quantum circuit creation, execution, and management directly from AI conversations.

## Features

- **Quantum Backend Management**: List and inspect available quantum backends
- **Job Management**: Monitor, cancel, and retrieve job results
- **Account Management**: Easy setup and configuration of IBM Quantum accounts

## Prerequisites

- Python 3.10 or higher
- IBM Quantum account (free at [quantum.ibm.com](https://quantum.ibm.com))
- IBM Quantum API token

## Installation

This project recommends using [uv](https://astral.sh/uv) for virtual environments and dependencies management. If you don't have `uv` installed, check out the instructions in <https://docs.astral.sh/uv/getting-started/installation/>

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
   ```

3. **Get your IBM Quantum token**:
   - Visit [IBM Quantum](https://quantum.ibm.com)
   - Sign up or log in to your account
   - Go to Account Settings
   - Copy your API token
   - Add it to your .env file

## Quick Start

### Running the Server

```bash
uv run qiskit-ibm-runtime-mcp-server
```

The server will start and listen for MCP connections.

### Basic Usage Examples

#### Async Usage (MCP Server)

```python
# 1. Setup IBM Quantum Account
await setup_ibm_quantum_account(token="your_token_here")

# 2. List Available Backends
backends = await list_backends()
print(f"Available backends: {len(backends['backends'])}")

# 3. Get the least busy backend
backend = await least_busy_backend()
print(f"Least busy backend: {backend}")

# 4. Get backend's properties
backend_props = await get_backend_properties("backend_name")
print(f"Backend_name properties: {backend_props}")

# 5. List recent jobs
jobs = await list_my_jobs(10)
print(f"Last 10 jobs: {jobs}")

# 6. Get job status
job_status = await get_job_status("job_id")
print(f"Job status: {job_status}")

# 7. Cancel job
cancelled_job = await cancel_job("job_id")
print(f"Cancelled job: {cancelled_job}")
```

#### Sync Usage (DSPy, Scripts, Jupyter)

For frameworks that don't support async operations:

```python
from qiskit_ibm_runtime_mcp_server.sync import (
    setup_ibm_quantum_account_sync,
    list_backends_sync,
    least_busy_backend_sync,
    get_backend_properties_sync,
    list_my_jobs_sync,
    get_job_status_sync,
    cancel_job_sync
)

# Use synchronously without async/await
backends = list_backends_sync()
print(f"Available backends: {backends['total_backends']}")

# Get least busy backend
backend = least_busy_backend_sync()
print(f"Least busy: {backend['backend_name']}")

# Works in Jupyter notebooks and DSPy agents
jobs = list_my_jobs_sync(limit=5)
print(f"Recent jobs: {len(jobs['jobs'])}")
```

**DSPy Integration Example:**

```python
import dspy
from qiskit_ibm_runtime_mcp_server.sync import (
    list_backends_sync,
    least_busy_backend_sync,
    get_backend_properties_sync
)

agent = dspy.ReAct(
    YourSignature,
    tools=[
        list_backends_sync,
        least_busy_backend_sync,
        get_backend_properties_sync
    ]
)

result = agent(user_request="What QPUs are available?")
```


## API Reference

### Tools

#### `setup_ibm_quantum_account(token: str, channel: str = "ibm_quantum_platform")`
Configure IBM Quantum account with API token.

**Parameters:**
- `token`: IBM Quantum API token
- `channel`: Service channel ("ibm_quantum_platform")

**Returns:** Setup status and account information

#### `list_backends()`
Get list of available quantum backends.

**Returns:** Array of backend information including:
- Name, status, queue length
- Number of qubits, coupling map
- Simulator vs. hardware designation

### `least_busy_backend()`
Get the current least busy IBM Quantum backend available
**Returns:** The backend with the fewest number of pending jobs

#### `get_backend_properties(backend_name: str)`
Get detailed properties of specific backend.

**Returns:** Complete backend configuration including:
- Hardware specifications
- Gate set and coupling map
- Current operational status
- Queue information

#### `list_my_jobs(limit: int = 10)`
Get list of recent jobs from your account.

**Parameters:**
- `limit`: The N of jobs to retrieve

#### `get_job_status(job_id: str)`
Check status of submitted job.

**Parameters:**
- `job_id`: The ID of the job to get its status

**Returns:** Current job status, creation date, backend info

#### `cancel_job(job_id: str)`
Cancel a running or queued job.

**Parameters:**
- `job_id`: The ID of the job to cancel


### Resources

#### `ibm_quantum://status`
Get current service status and connection info.


## Security Considerations

- Store IBM Quantum tokens securely
- Use environment variables for production deployments
- Implement rate limiting for production use
- Monitor quantum resource consumption

## Contributing

Contributions are welcome! Areas for improvement:

- Support for Primitives
- Support for error mitigation/correction/cancellation techniques
- Other qiskit-ibm-runtime features


### Other ways of testing and debugging the server

> _**Note**: to launch the MCP inspector you will need to have [`node` and `npm`](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm)_

1. From a terminal, go into the cloned repo directory

1. Switch to the virtual environment

    ```sh
    source .venv/bin/activate
    ```

1. Run the MCP Inspector:

    ```sh
    npx @modelcontextprotocol/inspector uv run qiskit-ibm-runtime-mcp-server
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
uv run pytest tests/test_server.py -v
```

### Test Structure

- `tests/test_server.py` - Unit tests for server functions
- `tests/test_integration.py` - Integration tests
- `tests/conftest.py` - Test fixtures and configuration

### Test Coverage

The test suite covers:
- ✅ Service initialization and account setup
- ✅ Backend listing and analysis
- ✅ Job management and monitoring
- ✅ Error handling and validation
- ✅ Integration scenarios
- ✅ Resource and tool handlers

