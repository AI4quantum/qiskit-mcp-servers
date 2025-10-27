# Qiskit MCP Server

A comprehensive Model Context Protocol (MCP) server that provides AI assistants with access to Qiskit SDK functions for quantum circuit creation, manipulation, transpilation, simulation, and visualization. This server enables quantum computing development directly from AI conversations without requiring cloud credentials.

## Features

### Core Circuit Operations
- **Quantum Circuit Creation**: Create circuits with custom qubits and classical bits
- **Gate Operations**: Add quantum gates including parameterized rotation gates (RX, RY, RZ, U)
- **Circuit Transpilation**: Optimize circuits with different optimization levels and basis gates
- **Random Circuits**: Generate random circuits for testing

### Advanced Circuit Features
- **Circuit Composition**: Compose and tensor circuits together
- **Parametric Circuits**: Create circuits with named parameters and bind values
- **Circuit Decomposition**: Break down complex gates into basis gates
- **Controlled Gates**: Add controlled versions of any gate (CX, CCX, MCX, CZ, etc.)
- **Power Gates**: Apply gates raised to powers (e.g., sqrt(X))

### Circuit Library
- **QFT**: Quantum Fourier Transform and inverse QFT circuits
- **Grover Operator**: Grover's search algorithm operators
- **Variational Ansätze**: EfficientSU2, RealAmplitudes circuits for VQE/QAOA

### Simulation & Analysis
- **Statevector Simulation**: Get quantum state vectors
- **Aer Simulation**: High-performance sampling with Qiskit Aer
- **Unitary Matrices**: Extract unitary matrix representations
- **Circuit Analysis**: Comprehensive metrics (depth, gate counts, connectivity)
- **Instruction Details**: Get detailed gate-by-gate breakdown

### Quantum Information
- **SparsePauliOp**: Create and manipulate quantum observables
- **Density Matrices**: Mixed state representations
- **Fidelity Calculations**: State and gate fidelity measurements
- **Entropy & Entanglement**: von Neumann entropy, entanglement of formation
- **Partial Trace**: Reduced density matrices for subsystems
- **Expectation Values**: Calculate observable measurements

### Qiskit Primitives
- **Sampler**: Measurement sampling from quantum circuits
- **Estimator**: Expectation value calculations
- **Variational Algorithms**: Full support for VQE, QAOA, and custom workflows

### Advanced Features
- **OpenQASM 3 Support**: Convert between QASM 2.0 and QASM 3.0
- **Circuit Visualization**: Text/ASCII art and matplotlib-based drawing
- **Multiple Drawing Styles**: Default, IQP, Clifford, and textbook styles

## Prerequisites

- Python 3.10 or higher
- No API tokens or cloud credentials required

## Installation

### Install from PyPI

The easiest way to install is via pip:

```bash
pip install qiskit-mcp-server
```

### Install from Source

This project recommends using [uv](https://astral.sh/uv) for virtual environments and dependencies management. If you don't have `uv` installed, check out the instructions in <https://docs.astral.sh/uv/getting-started/installation/>

### Setting up the Project with uv

1. **Initialize or sync the project**:
   ```bash
   # This will create a virtual environment and install dependencies
   uv sync
   ```

## Quick Start

### Running the Server

```bash
uv run qiskit-mcp-server
```

The server will start and listen for MCP connections.

### Basic Usage Examples

#### Async Usage (MCP Server)

```python
# 1. Create a quantum circuit
circuit_result = await create_quantum_circuit(num_qubits=2, num_classical_bits=2, name="bell_state")
circuit_qasm = circuit_result["circuit"]["qasm"]

# 2. Add gates to create a Bell state
bell_result = await add_gates_to_circuit(circuit_qasm, "h 0; cx 0 1")
bell_qasm = bell_result["circuit"]["qasm"]

# 3. Visualize the circuit
visualization = await visualize_circuit(bell_qasm, output_format="text")
print(visualization["visualization"])

# 4. Get statevector simulation
statevector = await get_statevector(bell_qasm)
print(f"Statevector: {statevector['statevector']}")
print(f"Probabilities: {statevector['probabilities']}")

# 5. Transpile the circuit
transpiled = await transpile_circuit(bell_qasm, optimization_level=3, basis_gates="cx,id,rz,sx,x")
print(f"Original depth: {transpiled['original_depth']}")
print(f"Transpiled depth: {transpiled['transpiled_depth']}")

# 6. Get circuit depth and size
depth_info = await get_circuit_depth(bell_qasm)
print(f"Depth: {depth_info['depth']}, Size: {depth_info['size']}")

# 7. Create random circuit for testing
random_circuit = await create_random_circuit(num_qubits=3, depth=5, measure=True, seed=42)
print(f"Random circuit created: {random_circuit['circuit']['name']}")
```

#### Sync Usage (DSPy, Scripts, Jupyter)

For frameworks that don't support async operations:

```python
from qiskit_mcp_server.sync import (
    create_quantum_circuit_sync,
    add_gates_to_circuit_sync,
    transpile_circuit_sync,
    get_circuit_depth_sync,
    get_statevector_sync,
    visualize_circuit_sync,
    create_random_circuit_sync,
    get_qiskit_version_sync
)

# Create and manipulate circuits synchronously
circuit = create_quantum_circuit_sync(2, 2, "my_circuit")
circuit_qasm = circuit["circuit"]["qasm"]

# Add gates
bell_circuit = add_gates_to_circuit_sync(circuit_qasm, "h 0; cx 0 1")
bell_qasm = bell_circuit["circuit"]["qasm"]

# Get statevector
result = get_statevector_sync(bell_qasm)
print(f"Statevector: {result['statevector']}")

# Visualize
viz = visualize_circuit_sync(bell_qasm, "text")
print(viz["visualization"])

# Works in Jupyter notebooks and DSPy agents
depth = get_circuit_depth_sync(bell_qasm)
print(f"Circuit depth: {depth['depth']}")
```

**DSPy Integration Example:**

```python
import dspy
from qiskit_mcp_server.sync import (
    create_quantum_circuit_sync,
    add_gates_to_circuit_sync,
    get_statevector_sync,
    visualize_circuit_sync
)

agent = dspy.ReAct(
    YourSignature,
    tools=[
        create_quantum_circuit_sync,
        add_gates_to_circuit_sync,
        get_statevector_sync,
        visualize_circuit_sync
    ]
)

result = agent(user_request="Create a Bell state and show me the statevector")
```


## API Reference

### Tools

#### `create_quantum_circuit(num_qubits: int, num_classical_bits: int = 0, name: str = "circuit")`
Create a new quantum circuit with specified number of qubits and classical bits.

**Parameters:**
- `num_qubits`: Number of quantum bits in the circuit (must be positive)
- `num_classical_bits`: Number of classical bits for measurement (default: 0)
- `name`: Name for the circuit (default: "circuit")

**Returns:** Circuit information including QASM representation

**Example:**
```python
result = await create_quantum_circuit(3, 3, "my_circuit")
# Returns: {"status": "success", "circuit": {...}}
```

#### `add_gates_to_circuit(circuit_qasm: str, gates: str)`
Add quantum gates to an existing circuit specified in QASM format.

**Parameters:**
- `circuit_qasm`: QASM representation of the circuit
- `gates`: Gates to add in natural language format. Multiple gates separated by semicolons.

**Supported Gates:**

*Standard Single-Qubit Gates:*
- `h` (Hadamard), `x`, `y`, `z`, `s`, `t`
- `sdg` (S-dagger), `tdg` (T-dagger)
- `sx` (√X), `sxdg` (√X-dagger)

*Rotation Gates (Parameterized):*
- `rx <angle> <qubit>` - Rotation around X-axis
- `ry <angle> <qubit>` - Rotation around Y-axis
- `rz <angle> <qubit>` - Rotation around Z-axis
- `p <angle> <qubit>` - Phase gate
- `u <theta> <phi> <lambda> <qubit>` - General single-qubit rotation

*Two-Qubit Gates:*
- `cx`/`cnot`, `cz`, `cy`, `swap`, `iswap`, `ecr`

*Circuit Control:*
- `barrier` - Prevent optimization across barrier
- `reset` - Reset qubit to |0⟩
- `measure` - Measurement

**Examples:**
```python
# Standard gates
result = await add_gates_to_circuit(qasm, "h 0")

# Rotation gates with angles (in radians)
import math
result = await add_gates_to_circuit(qasm, f"rx {math.pi/2} 0")
result = await add_gates_to_circuit(qasm, f"ry {math.pi/4} 1")
result = await add_gates_to_circuit(qasm, f"rz {math.pi} 0")

# Phase gate
result = await add_gates_to_circuit(qasm, f"p {math.pi/2} 0")

# General U gate (theta, phi, lambda, qubit)
result = await add_gates_to_circuit(qasm, f"u {math.pi/2} {math.pi/4} {math.pi/3} 0")

# Multiple gates with rotations
result = await add_gates_to_circuit(
    qasm,
    f"h 0; rx {math.pi/2} 0; cx 0 1; ry {math.pi/4} 1; measure 0 0; measure 1 1"
)

# Barriers and resets
result = await add_gates_to_circuit(qasm, "h 0; barrier 0 1; cx 0 1; reset 0")

# Advanced gates
result = await add_gates_to_circuit(qasm, "sdg 0; sx 1; iswap 0 1")
```

#### `transpile_circuit(circuit_qasm: str, optimization_level: int = 1, basis_gates: str = "")`
Transpile a quantum circuit to optimize it and map to specific basis gates.

**Parameters:**
- `circuit_qasm`: QASM representation of the circuit to transpile
- `optimization_level`: Optimization level (0-3, default: 1)
  - 0: No optimization
  - 1: Light optimization
  - 2: Medium optimization
  - 3: Heavy optimization
- `basis_gates`: Comma-separated list of basis gates (e.g., "cx,id,rz,sx,x")
                If empty, uses default basis gates

**Returns:** Transpiled circuit with original and new depths

**Example:**
```python
result = await transpile_circuit(qasm, optimization_level=3, basis_gates="cx,id,rz,sx,x")
# Returns: {"status": "success", "original_depth": 5, "transpiled_depth": 3, ...}
```

#### `get_circuit_depth(circuit_qasm: str)`
Get the depth of a quantum circuit.

**Parameters:**
- `circuit_qasm`: QASM representation of the circuit

**Returns:** Circuit depth, number of qubits, and size

#### `get_circuit_qasm(circuit_qasm: str)`
Get the OpenQASM representation of a circuit (for validation/formatting).

**Parameters:**
- `circuit_qasm`: QASM representation of the circuit

**Returns:** Formatted QASM string

#### `get_statevector(circuit_qasm: str)`
Get the statevector result from simulating a quantum circuit.

**Parameters:**
- `circuit_qasm`: QASM representation of the circuit to simulate

**Returns:** Statevector and probability distribution

**Example:**
```python
result = await get_statevector(bell_state_qasm)
# Returns: {
#   "status": "success",
#   "statevector": "Statevector([0.70710678+0.j, 0.+0.j, 0.+0.j, 0.70710678+0.j])",
#   "probabilities": [0.5, 0.0, 0.0, 0.5],
#   "num_qubits": 2
# }
```

#### `visualize_circuit(circuit_qasm: str, output_format: str = "text")`
Visualize a quantum circuit.

**Parameters:**
- `circuit_qasm`: QASM representation of the circuit
- `output_format`: Output format - "text" for ASCII art, "mpl" for matplotlib description

**Returns:** Circuit visualization as text

**Example:**
```python
result = await visualize_circuit(qasm, "text")
# Returns ASCII art like:
#      ┌───┐
# q_0: ┤ H ├──■──
#      └───┘┌─┴─┐
# q_1: ─────┤ X ├
#           └───┘
```

#### `create_random_circuit(num_qubits: int, depth: int, measure: bool = False, seed: int = None)`
Create a random quantum circuit for testing purposes.

**Parameters:**
- `num_qubits`: Number of qubits
- `depth`: Depth of the circuit
- `measure`: Whether to add measurements (default: False)
- `seed`: Random seed for reproducibility (default: None)

**Returns:** Random circuit information

**Example:**
```python
result = await create_random_circuit(num_qubits=3, depth=5, measure=True, seed=42)
# Creates reproducible random circuit
```

### Quantum Information Tools

#### `create_pauli_operator(pauli_strings: str, coeffs: str = "")`
Create a SparsePauliOp (quantum observable) from Pauli strings and coefficients.

**Parameters:**
- `pauli_strings`: Comma-separated Pauli strings (e.g., "XX,YZ,ZZ")
- `coeffs`: Optional comma-separated coefficients (default: all 1.0)

**Returns:** SparsePauliOp information including Pauli terms and coefficients

**Example:**
```python
# Create Hamiltonian: H = XX + 0.5*YZ + 0.25*ZZ
result = await create_pauli_operator("XX,YZ,ZZ", "1.0,0.5,0.25")
```

#### `create_operator_from_circuit(circuit_qasm: str)`
Create a unitary Operator from a quantum circuit.

**Parameters:**
- `circuit_qasm`: QASM representation of the circuit

**Returns:** Operator information including matrix representation and properties

#### `create_density_matrix(circuit_qasm: str)`
Create a DensityMatrix (mixed quantum state) from a circuit.

**Parameters:**
- `circuit_qasm`: QASM representation of the circuit

**Returns:** Density matrix with purity and validity information

**Example:**
```python
result = await create_density_matrix(circuit_qasm)
print(f"Purity: {result['density_matrix']['purity']}")
```

#### `calculate_state_fidelity(circuit_qasm1: str, circuit_qasm2: str)`
Calculate fidelity between two quantum states (0 = orthogonal, 1 = identical).

**Parameters:**
- `circuit_qasm1`: QASM representation of first circuit
- `circuit_qasm2`: QASM representation of second circuit

**Returns:** Fidelity value between 0 and 1

**Example:**
```python
fidelity = await calculate_state_fidelity(bell_state_qasm, target_qasm)
# Returns: {"status": "success", "fidelity": 0.999}
```

#### `calculate_gate_fidelity(circuit_qasm: str, target_qasm: str)`
Calculate average gate fidelity between two quantum operations.

**Parameters:**
- `circuit_qasm`: QASM of actual operation
- `target_qasm`: QASM of target operation

**Returns:** Average gate fidelity and process fidelity

#### `calculate_entropy(circuit_qasm: str, subsystem_qubits: str = "")`
Calculate von Neumann entropy of a quantum state.

**Parameters:**
- `circuit_qasm`: QASM representation of the circuit
- `subsystem_qubits`: Optional comma-separated qubit indices for partial trace

**Returns:** Entropy value (0 for pure states, > 0 for mixed states)

**Example:**
```python
# Calculate total entropy
result = await calculate_entropy(circuit_qasm)

# Calculate entropy of subsystem (trace out qubits 0 and 2)
result = await calculate_entropy(circuit_qasm, "0,2")
```

#### `calculate_entanglement(circuit_qasm: str)`
Calculate entanglement of formation for a 2-qubit state.

**Parameters:**
- `circuit_qasm`: QASM of a 2-qubit circuit

**Returns:** Entanglement of formation value

**Example:**
```python
# For Bell state
result = await calculate_entanglement(bell_state_qasm)
# Returns high entanglement value
```

#### `partial_trace_state(circuit_qasm: str, trace_qubits: str)`
Compute partial trace (reduced density matrix) over specified qubits.

**Parameters:**
- `circuit_qasm`: QASM representation
- `trace_qubits`: Comma-separated qubit indices to trace out (e.g., "0,2")

**Returns:** Reduced density matrix information

**Example:**
```python
# Trace out qubits 0 and 2 from a 3-qubit state
result = await partial_trace_state(circuit_qasm, "0,2")
# Returns reduced state for qubit 1
```

#### `expectation_value(circuit_qasm: str, pauli_strings: str, coeffs: str = "")`
Calculate expectation value ⟨ψ|H|ψ⟩ of an observable for a quantum state.

**Parameters:**
- `circuit_qasm`: QASM representation
- `pauli_strings`: Comma-separated Pauli strings defining the observable
- `coeffs`: Optional coefficients

**Returns:** Expectation value (real number)

**Example:**
```python
# Calculate ⟨ZZ⟩ for Bell state
result = await expectation_value(bell_state_qasm, "ZZ")

# Calculate ⟨H⟩ = 0.5*XX + 0.5*ZZ
result = await expectation_value(circuit_qasm, "XX,ZZ", "0.5,0.5")
```

#### `random_quantum_state(num_qubits: int, seed: int = None)`
Generate a random quantum statevector (pure state) for testing.

**Parameters:**
- `num_qubits`: Number of qubits
- `seed`: Random seed for reproducibility

**Returns:** Random statevector information

#### `random_density_matrix_state(num_qubits: int, rank: int = None, seed: int = None)`
Generate a random density matrix (mixed state) for testing.

**Parameters:**
- `num_qubits`: Number of qubits
- `rank`: Rank of density matrix (None for full rank)
- `seed`: Random seed

**Returns:** Random density matrix information

### Qiskit Primitives (Sampler & Estimator)

The Primitives are the **modern, recommended way** to execute quantum circuits in Qiskit 1.0+.

#### `sample_circuit(circuit_qasm: str, shots: int = 1024, seed: int = None)`
Sample from a quantum circuit using the **Sampler primitive**.

**Parameters:**
- `circuit_qasm`: QASM representation (must include measurements)
- `shots`: Number of measurement shots (default: 1024)
- `seed`: Random seed for reproducibility

**Returns:** Measurement counts and probability distribution

**Example:**
```python
# Create Bell state with measurements
circuit = await create_quantum_circuit(2, 2, "bell")
qasm = circuit["circuit"]["qasm"]
bell = await add_gates_to_circuit(qasm, "h 0; cx 0 1; measure 0 0; measure 1 1")

# Sample using the Sampler primitive
result = await sample_circuit(bell["circuit"]["qasm"], shots=1000)
print(result["counts"])  # {'00': 503, '11': 497}
```

#### `estimate_expectation_values(circuit_qasm: str, observables: str, coeffs: str = "", precision: float = 0.0)`
Estimate expectation values using the **Estimator primitive**.

**Parameters:**
- `circuit_qasm`: QASM representation of the circuit
- `observables`: Comma-separated Pauli strings (e.g., "ZZ,XX,YY")
- `coeffs`: Optional comma-separated coefficients
- `precision`: Target precision (0.0 = exact statevector simulation)

**Returns:** Expectation value and standard deviation

**Example:**
```python
# Create Bell state
bell_qasm = ... # Bell state circuit

# Estimate ⟨ZZ⟩ using Estimator primitive
result = await estimate_expectation_values(bell_qasm, "ZZ")
print(f"⟨ZZ⟩ = {result['expectation_value']}")  # 1.0 for Bell state

# Estimate Hamiltonian H = 0.5*XX + 0.3*ZZ
result = await estimate_expectation_values(bell_qasm, "XX,ZZ", "0.5,0.3")
print(f"⟨H⟩ = {result['expectation_value']}")
```

#### `run_variational_estimation(circuit_qasms_json: str, observables: str, coeffs: str = "")`
Run Estimator on multiple circuit variations for **VQE/QAOA** workflows.

**Parameters:**
- `circuit_qasms_json`: JSON array of QASM strings (circuit variations)
- `observables`: Comma-separated Pauli strings for the Hamiltonian
- `coeffs`: Optional coefficients

**Returns:** Expectation values for all variations, minimum energy, optimal circuit index

**Example:**
```python
import json
import math

# Create circuit variations with different parameters
circuits = []
for theta in [0, math.pi/4, math.pi/2, 3*math.pi/4]:
    circuit = await create_quantum_circuit(2, 0, f"ansatz_{theta}")
    qasm = circuit["circuit"]["qasm"]
    result = await add_gates_to_circuit(qasm, f"ry {theta} 0; ry {theta} 1")
    circuits.append(result["circuit"]["qasm"])

# Run variational estimation
result = await run_variational_estimation_tool(
    json.dumps(circuits),
    "ZZ",
    "1.0"
)
print(f"Minimum energy: {result['minimum_energy']}")
print(f"Optimal circuit: {result['minimum_index']}")
```

**Use Cases:**
- **VQE (Variational Quantum Eigensolver)**: Find ground state energies
- **QAOA (Quantum Approximate Optimization Algorithm)**: Solve combinatorial problems
- **Parameter optimization**: Find optimal circuit parameters
- **Quantum machine learning**: Train quantum neural networks

### Advanced Circuit Tools

#### `compose_circuits(circuit1_qasm: str, circuit2_qasm: str, inplace: bool = False)`
Compose two quantum circuits sequentially (append circuit2 to circuit1).

**Parameters:**
- `circuit1_qasm`: QASM representation of first circuit
- `circuit2_qasm`: QASM representation of second circuit to append
- `inplace`: If True, modify first circuit; if False, return new circuit

**Example:**
```python
# Create two circuits
result1 = await create_quantum_circuit(2, 0)
qasm1 = await add_gates_to_circuit(result1["circuit"]["qasm"], "h 0; h 1")
qasm1 = qasm1["circuit"]["qasm"]

result2 = await create_quantum_circuit(2, 0)
qasm2 = await add_gates_to_circuit(result2["circuit"]["qasm"], "cx 0 1")
qasm2 = qasm2["circuit"]["qasm"]

# Compose them
result = await compose_circuits(qasm1, qasm2)
# Result: H gates followed by CX gate
```

#### `tensor_circuits(circuit1_qasm: str, circuit2_qasm: str)`
Tensor product of two circuits (place side by side on different qubits).

**Example:**
```python
# 2-qubit circuit with H gate
circuit1 = await create_quantum_circuit(2, 0)
qasm1 = await add_gates_to_circuit(circuit1["circuit"]["qasm"], "h 0")

# 1-qubit circuit with X gate
circuit2 = await create_quantum_circuit(1, 0)
qasm2 = await add_gates_to_circuit(circuit2["circuit"]["qasm"], "x 0")

# Tensor: results in 3-qubit circuit
result = await tensor_circuits(qasm1["circuit"]["qasm"], qasm2["circuit"]["qasm"])
```

#### `create_parametric_circuit(num_qubits: int, parameter_names: str, num_classical_bits: int = 0)`
Create a circuit with named parameters for variational algorithms.

**Parameters:**
- `num_qubits`: Number of qubits
- `parameter_names`: Comma-separated parameter names (e.g., "theta,phi,lambda")
- `num_classical_bits`: Number of classical bits

**Example:**
```python
result = await create_parametric_circuit(2, "theta,phi")
# Returns circuit with parameters that can be bound later
```

#### `bind_parameters(circuit_qasm: str, parameter_values: str)`
Bind parameter values to a parametric circuit.

**Parameters:**
- `circuit_qasm`: QASM representation of parametric circuit
- `parameter_values`: JSON dict of parameter names to values

**Example:**
```python
import json
param_values = json.dumps({"theta": 1.57, "phi": 3.14})
result = await bind_parameters(circuit_qasm, param_values)
```

#### `decompose_circuit(circuit_qasm: str, reps: int = 1)`
Decompose circuit gates into basis gates.

**Parameters:**
- `circuit_qasm`: QASM representation of circuit
- `reps`: Number of decomposition repetitions

**Returns:** Original and decomposed depths and sizes

#### `add_controlled_gate(circuit_qasm: str, gate_name: str, control_qubits: str, target_qubits: str)`
Add a controlled version of a gate to the circuit.

**Parameters:**
- `circuit_qasm`: QASM representation of circuit
- `gate_name`: Name of gate to control (x, z, h, y, swap)
- `control_qubits`: Comma-separated control qubit indices (e.g., "0" or "0,1")
- `target_qubits`: Comma-separated target qubit indices

**Example:**
```python
# Add CX (controlled-X)
result = await add_controlled_gate(qasm, "x", "0", "1")

# Add CCX (Toffoli)
result = await add_controlled_gate(qasm, "x", "0,1", "2")

# Add multi-controlled X
result = await add_controlled_gate(qasm, "x", "0,1,2", "3")
```

#### `add_power_gate(circuit_qasm: str, gate_name: str, power: float, qubit: int)`
Add a gate raised to a power (e.g., sqrt(X) is X^0.5).

**Example:**
```python
# Add sqrt(X) gate
result = await add_power_gate(qasm, "x", 0.5, 0)

# Add X^(1/3)
result = await add_power_gate(qasm, "x", 0.333, 0)
```

### Circuit Library Tools

#### `create_qft_circuit(num_qubits: int, inverse: bool = False, do_swaps: bool = True)`
Create a Quantum Fourier Transform circuit.

**Parameters:**
- `num_qubits`: Number of qubits
- `inverse`: If True, create inverse QFT
- `do_swaps`: If True, include swap gates

**Example:**
```python
# Create QFT circuit
qft = await create_qft_circuit(3, inverse=False, do_swaps=True)

# Create inverse QFT
iqft = await create_qft_circuit(3, inverse=True)
```

#### `create_grover_operator(num_qubits: int, oracle_qasm: str)`
Create a Grover operator circuit for quantum search.

**Parameters:**
- `num_qubits`: Number of qubits
- `oracle_qasm`: QASM representation of oracle circuit

#### `create_efficient_su2(num_qubits: int, reps: int = 3, entanglement: str = "full")`
Create an EfficientSU2 variational circuit (hardware-efficient ansatz).

**Parameters:**
- `num_qubits`: Number of qubits
- `reps`: Number of repetitions
- `entanglement`: Entanglement strategy (full, linear, circular)

**Example:**
```python
# Create EfficientSU2 ansatz for VQE
result = await create_efficient_su2(4, reps=3, entanglement="linear")
print(f"Parameters: {result['num_parameters']}")
```

#### `create_real_amplitudes(num_qubits: int, reps: int = 3, entanglement: str = "full")`
Create a RealAmplitudes variational circuit (for quantum chemistry).

**Example:**
```python
# Create RealAmplitudes ansatz
result = await create_real_amplitudes(4, reps=2, entanglement="full")
```

### Simulation & Analysis Tools

#### `simulate_with_aer(circuit_qasm: str, shots: int = 1024, backend_name: str = "aer_simulator")`
Simulate circuit using Qiskit Aer high-performance simulator.

**Parameters:**
- `circuit_qasm`: QASM representation of circuit
- `shots`: Number of shots
- `backend_name`: Aer backend name

**Example:**
```python
result = await simulate_with_aer(qasm, shots=2048)
print(f"Counts: {result['counts']}")
```

#### `get_unitary_matrix(circuit_qasm: str)`
Get the unitary matrix representation of a circuit.

**Returns:** Unitary matrix, dimension, and whether it's unitary

**Example:**
```python
result = await get_unitary_matrix(bell_qasm)
print(f"Dimension: {result['dimension']}")
print(f"Is unitary: {result['is_unitary']}")
```

#### `analyze_circuit(circuit_qasm: str)`
Perform comprehensive circuit analysis including gate counts and metrics.

**Returns:**
- `num_qubits`, `num_clbits`, `depth`, `size`, `width`
- `gate_counts`: Dict of gate types and counts
- `two_qubit_gate_count`: Number of two-qubit gates
- `num_nonlocal_gates`: Number of multi-qubit gates
- `num_connected_components`: Circuit connectivity

**Example:**
```python
result = await analyze_circuit(qasm)
print(f"Depth: {result['depth']}")
print(f"Gate counts: {result['gate_counts']}")
print(f"Two-qubit gates: {result['two_qubit_gate_count']}")
```

#### `get_circuit_instructions(circuit_qasm: str)`
Get detailed list of all circuit instructions with parameters.

**Returns:** List of instructions with index, gate name, qubits, clbits, and params

**Example:**
```python
result = await get_circuit_instructions(qasm)
for inst in result['instructions']:
    print(f"{inst['index']}: {inst['gate']} on qubits {inst['qubits']}")
```

### OpenQASM 3 Tools

#### `convert_to_qasm3(circuit_qasm: str)`
Convert circuit from OpenQASM 2.0 to OpenQASM 3.0 format.

**Example:**
```python
result = await convert_to_qasm3(qasm2_str)
print(result['qasm3'])
```

#### `load_qasm3_circuit(qasm3_str: str)`
Load circuit from OpenQASM 3.0 format.

**Returns:** Circuit information with both QASM 2.0 and 3.0 representations

### Circuit Drawing Tools

#### `draw_circuit_text(circuit_qasm: str, fold: int = -1)`
Draw circuit as text/ASCII art.

**Parameters:**
- `circuit_qasm`: QASM representation of circuit
- `fold`: Column to fold the circuit at (-1 for no folding)

**Example:**
```python
result = await draw_circuit_text(bell_qasm)
print(result['drawing'])
```

#### `draw_circuit_matplotlib(circuit_qasm: str, style: str = "default")`
Draw circuit using matplotlib (returns base64 encoded PNG image).

**Parameters:**
- `style`: Drawing style (default, iqp, clifford, textbook)

**Returns:** Base64 encoded PNG image

**Example:**
```python
result = await draw_circuit_matplotlib(qasm, style="textbook")
# result['image_base64'] contains the base64 PNG data
```

### Resources

#### `qiskit://version`
Get current Qiskit version information.

**Returns:** Qiskit SDK version and package info

## Common Patterns

### Creating a Bell State

```python
# Create circuit
circuit = await create_quantum_circuit(2, 2, "bell_state")
qasm = circuit["circuit"]["qasm"]

# Add Bell state gates
bell = await add_gates_to_circuit(qasm, "h 0; cx 0 1")
bell_qasm = bell["circuit"]["qasm"]

# Measure
measured = await add_gates_to_circuit(bell_qasm, "measure 0 0; measure 1 1")
```

### Creating a GHZ State

```python
# Create circuit
circuit = await create_quantum_circuit(3, 3, "ghz_state")
qasm = circuit["circuit"]["qasm"]

# Add GHZ gates
ghz = await add_gates_to_circuit(qasm, "h 0; cx 0 1; cx 0 2")
```

### Quantum Teleportation Circuit

```python
# Create circuit with 3 qubits and 2 classical bits
circuit = await create_quantum_circuit(3, 2, "teleportation")
qasm = circuit["circuit"]["qasm"]

# Build teleportation circuit
result = await add_gates_to_circuit(
    qasm,
    "h 1; cx 1 2; cx 0 1; h 0; measure 0 0; measure 1 1"
)
```

### Using Rotation Gates for Quantum Algorithms

```python
import math

# Create a single-qubit rotation circuit
circuit = await create_quantum_circuit(1, 1, "rotation_demo")
qasm = circuit["circuit"]["qasm"]

# Apply rotation sequence (Euler angles)
result = await add_gates_to_circuit(
    qasm,
    f"rz {math.pi/4} 0; ry {math.pi/2} 0; rz {math.pi/4} 0"
)

# Get statevector to verify rotation
state = await get_statevector(result["circuit"]["qasm"])
print(state["statevector"])
```

### Variational Quantum Eigensolver (VQE) Ansatz

```python
import math

# Create parameterized circuit for VQE
circuit = await create_quantum_circuit(2, 0, "vqe_ansatz")
qasm = circuit["circuit"]["qasm"]

# Variational form with rotation gates
theta1, theta2, theta3 = 0.5, 1.2, 0.8

result = await add_gates_to_circuit(
    qasm,
    f"ry {theta1} 0; ry {theta2} 1; cx 0 1; ry {theta3} 0"
)
```

### Quantum Approximate Optimization Algorithm (QAOA) Layer

```python
import math

# Create circuit for QAOA
circuit = await create_quantum_circuit(3, 0, "qaoa")
qasm = circuit["circuit"]["qasm"]

# Problem Hamiltonian layer (mixing)
beta = math.pi / 4
result = await add_gates_to_circuit(
    qasm,
    f"rx {2*beta} 0; rx {2*beta} 1; rx {2*beta} 2"
)

# Cost Hamiltonian layer (problem-specific)
gamma = math.pi / 8
qasm = result["circuit"]["qasm"]
result = await add_gates_to_circuit(
    qasm,
    f"rz {2*gamma} 0; rz {2*gamma} 1; cx 0 1; rz {2*gamma} 1; cx 0 1"
)
```

### Arbitrary State Preparation with U gate

```python
import math

# Prepare arbitrary single-qubit state using U gate
circuit = await create_quantum_circuit(1, 0, "arbitrary_state")
qasm = circuit["circuit"]["qasm"]

# U(θ, φ, λ) creates any single-qubit state
theta = math.pi / 3
phi = math.pi / 4
lambda_param = math.pi / 6

result = await add_gates_to_circuit(
    qasm,
    f"u {theta} {phi} {lambda_param} 0"
)

# Verify the state
state = await get_statevector(result["circuit"]["qasm"])
```

### Quantum Information and Observables

```python
# Create a Bell state
circuit = await create_quantum_circuit(2, 0, "bell")
qasm = circuit["circuit"]["qasm"]
bell = await add_gates_to_circuit(qasm, "h 0; cx 0 1")
bell_qasm = bell["circuit"]["qasm"]

# Create Pauli observable (Hamiltonian)
# H = ZZ + 0.5*XX
observable = await create_pauli_operator("ZZ,XX", "1.0,0.5")

# Calculate expectation value ⟨ψ|H|ψ⟩
exp_val = await expectation_value(bell_qasm, "ZZ,XX", "1.0,0.5")
print(f"Expectation value: {exp_val['expectation_value']}")

# Check entanglement
entanglement = await calculate_entanglement(bell_qasm)
print(f"Entanglement: {entanglement['entanglement_of_formation']}")

# Calculate state fidelity with target
target_bell = await create_quantum_circuit(2, 0, "target")
# ... create target state ...
fidelity = await calculate_state_fidelity(bell_qasm, target_bell_qasm)
print(f"Fidelity: {fidelity['fidelity']}")
```

### Analyzing Mixed States

```python
# Create a 3-qubit entangled state
circuit = await create_quantum_circuit(3, 0, "ghz")
qasm = circuit["circuit"]["qasm"]
ghz = await add_gates_to_circuit(qasm, "h 0; cx 0 1; cx 0 2")
ghz_qasm = ghz["circuit"]["qasm"]

# Create density matrix
rho = await create_density_matrix(ghz_qasm)
print(f"Purity: {rho['density_matrix']['purity']}")

# Calculate von Neumann entropy
entropy_result = await calculate_entropy(ghz_qasm)
print(f"Total entropy: {entropy_result['entropy']}")

# Trace out qubit 0 to get reduced state of qubits 1 and 2
reduced = await partial_trace_state(ghz_qasm, "0")
print(f"Reduced state purity: {reduced['reduced_state']['purity']}")

# Calculate entropy of subsystem
subsystem_entropy = await calculate_entropy(ghz_qasm, "0")
print(f"Subsystem entropy: {subsystem_entropy['entropy']}")
```

### Quantum Chemistry Example

```python
import math

# Create a simple molecular Hamiltonian
# H = -0.8*II + 0.17*ZZ - 0.22*XX + 0.17*YY
circuit = await create_quantum_circuit(2, 0, "h2_molecule")
qasm = circuit["circuit"]["qasm"]

# Prepare initial state with rotation gates
theta = math.pi / 4
prepared = await add_gates_to_circuit(
    qasm,
    f"ry {theta} 0; ry {theta} 1"
)

# Define molecular Hamiltonian as observable
hamiltonian = await create_pauli_operator(
    "II,ZZ,XX,YY",
    "-0.8,0.17,-0.22,0.17"
)

# Calculate ground state energy estimate
energy = await expectation_value(
    prepared["circuit"]["qasm"],
    "II,ZZ,XX,YY",
    "-0.8,0.17,-0.22,0.17"
)
print(f"Energy: {energy['expectation_value']}")
```

### Using Primitives for VQE (Variational Quantum Eigensolver)

```python
import json
import math

# Define molecular Hamiltonian (H2 molecule example)
# H = -1.05*II + 0.39*ZZ - 0.39*XX - 0.01*YY
hamiltonian_paulis = "II,ZZ,XX,YY"
hamiltonian_coeffs = "-1.05,0.39,-0.39,-0.01"

# Create variational ansatz with different theta values
circuits = []
theta_values = [i * math.pi / 8 for i in range(9)]  # 0 to π in steps of π/8

for theta in theta_values:
    circuit = await create_quantum_circuit(2, 0, f"vqe_ansatz_{theta:.3f}")
    qasm = circuit["circuit"]["qasm"]

    # Hardware-efficient ansatz
    ansatz = await add_gates_to_circuit(
        qasm,
        f"ry {theta} 0; ry {theta} 1; cx 0 1; ry {theta} 0"
    )
    circuits.append(ansatz["circuit"]["qasm"])

# Run variational estimation to find ground state
result = await run_variational_estimation_tool(
    json.dumps(circuits),
    hamiltonian_paulis,
    hamiltonian_coeffs
)

print(f"Ground state energy: {result['minimum_energy']}")
print(f"Optimal theta index: {result['minimum_index']}")
optimal_theta = theta_values[result['minimum_index']]
print(f"Optimal theta: {optimal_theta:.3f}")

# Sample the optimal circuit
optimal_qasm = circuits[result['minimum_index']]
# Add measurements
measured = await add_gates_to_circuit(optimal_qasm, "measure 0 0; measure 1 1")

# Get measurement statistics using Sampler
samples = await sample_circuit(measured["circuit"]["qasm"], shots=1000)
print(f"Measurement counts: {samples['counts']}")
```

### QAOA for MaxCut Problem

```python
import json
import math

# MaxCut on a 2-node graph with Hamiltonian H = 0.5*(I-ZZ)
# This encourages |01⟩ or |10⟩ states

# QAOA parameters
gamma = math.pi / 4  # Problem Hamiltonian parameter
beta = math.pi / 8   # Mixing Hamiltonian parameter

# Create circuit
circuit = await create_quantum_circuit(2, 2, "qaoa")
qasm = circuit["circuit"]["qasm"]

# Initial state: equal superposition
prepared = await add_gates_to_circuit(qasm, "h 0; h 1")

# Problem Hamiltonian layer (cost function)
cost_layer = await add_gates_to_circuit(
    prepared["circuit"]["qasm"],
    f"cx 0 1; rz {2*gamma} 1; cx 0 1"
)

# Mixing Hamiltonian layer
mixing_layer = await add_gates_to_circuit(
    cost_layer["circuit"]["qasm"],
    f"rx {2*beta} 0; rx {2*beta} 1"
)

# Estimate cost function value using Estimator
cost = await estimate_expectation_values(
    mixing_layer["circuit"]["qasm"],
    "II,ZZ",
    "0.5,-0.5"  # H = 0.5*(I-ZZ)
)
print(f"Cost function value: {-cost['expectation_value']}")

# Sample to get solution
measured = await add_gates_to_circuit(
    mixing_layer["circuit"]["qasm"],
    "measure 0 0; measure 1 1"
)
result = await sample_circuit(measured["circuit"]["qasm"], shots=1000)
print(f"Solution distribution: {result['counts']}")
```

### Comparing Sampler and Estimator

```python
# Create a simple state
circuit = await create_quantum_circuit(1, 1, "compare")
qasm = circuit["circuit"]["qasm"]
state = await add_gates_to_circuit(qasm, "h 0")

# METHOD 1: Using Estimator to get ⟨Z⟩
estimator_result = await estimate_expectation_values(
    state["circuit"]["qasm"],
    "Z"
)
print(f"⟨Z⟩ from Estimator: {estimator_result['expectation_value']}")

# METHOD 2: Using Sampler to estimate ⟨Z⟩ from counts
measured = await add_gates_to_circuit(state["circuit"]["qasm"], "measure 0 0")
sampler_result = await sample_circuit(measured["circuit"]["qasm"], shots=10000)

# Calculate ⟨Z⟩ from measurement statistics
# ⟨Z⟩ = P(0) - P(1)
counts = sampler_result["counts"]
total = sum(counts.values())
prob_0 = counts.get('0', 0) / total
prob_1 = counts.get('1', 0) / total
estimated_z = prob_0 - prob_1
print(f"⟨Z⟩ from Sampler: {estimated_z}")

# For |+⟩ = H|0⟩, both should give ⟨Z⟩ ≈ 0
```

### Using Circuit Composition

```python
# Build modular circuits and compose them
# Create initialization circuit
init_circuit = await create_quantum_circuit(2, 0, "init")
init_qasm = init_circuit["circuit"]["qasm"]
init = await add_gates_to_circuit(init_qasm, "h 0; h 1")

# Create entanglement circuit
entangle_circuit = await create_quantum_circuit(2, 0, "entangle")
entangle_qasm = entangle_circuit["circuit"]["qasm"]
entangle = await add_gates_to_circuit(entangle_qasm, "cx 0 1")

# Compose them
final = await compose_circuits(
    init["circuit"]["qasm"],
    entangle["circuit"]["qasm"]
)
# Result: H gates on both qubits followed by CX
```

### Building Quantum Fourier Transform Circuits

```python
# Create QFT circuit for phase estimation
qft = await create_qft_circuit(num_qubits=4, inverse=False)
qft_qasm = qft["circuit"]["qasm"]

# Use inverse QFT for the final step
iqft = await create_qft_circuit(num_qubits=4, inverse=True)

# Compose with other operations
phase_estimation = await compose_circuits(qft_qasm, my_unitary_qasm)
phase_estimation = await compose_circuits(
    phase_estimation["circuit"]["qasm"],
    iqft["circuit"]["qasm"]
)
```

### Multi-Controlled Gates

```python
# Create circuit with multi-controlled gates
circuit = await create_quantum_circuit(4, 0)
qasm = circuit["circuit"]["qasm"]

# Add CCX (Toffoli) gate
result = await add_controlled_gate(qasm, "x", "0,1", "2")
qasm = result["circuit"]["qasm"]

# Add CCCX (4-qubit Toffoli)
result = await add_controlled_gate(qasm, "x", "0,1,2", "3")
```

### Circuit Analysis and Optimization

```python
# Create and analyze a circuit
circuit = await create_quantum_circuit(3, 0)
qasm = circuit["circuit"]["qasm"]
qasm = await add_gates_to_circuit(qasm, "h 0; h 1; h 2; cx 0 1; cx 1 2; cx 0 2")
qasm = qasm["circuit"]["qasm"]

# Analyze the circuit
analysis = await analyze_circuit(qasm)
print(f"Depth: {analysis['depth']}")
print(f"Gate counts: {analysis['gate_counts']}")
print(f"Two-qubit gates: {analysis['two_qubit_gate_count']}")

# Transpile and optimize
optimized = await transpile_circuit(qasm, optimization_level=3)
print(f"Original depth: {optimized['original_depth']}")
print(f"Optimized depth: {optimized['transpiled_depth']}")

# Get detailed instructions
instructions = await get_circuit_instructions(optimized["circuit"]["qasm"])
for inst in instructions["instructions"]:
    print(f"{inst['gate']} on qubits {inst['qubits']}")
```

### Using Variational Ansätze

```python
# Create EfficientSU2 ansatz for VQE
ansatz = await create_efficient_su2(num_qubits=4, reps=3, entanglement="linear")
print(f"Number of parameters: {ansatz['circuit']['num_parameters']}")

# For quantum chemistry, use RealAmplitudes
chem_ansatz = await create_real_amplitudes(num_qubits=4, reps=2, entanglement="full")

# These circuits have parameters that can be optimized during VQE
```

### High-Performance Simulation with Aer

```python
# Create a larger circuit
circuit = await create_random_circuit(num_qubits=10, depth=20, measure=True, seed=42)
qasm = circuit["circuit"]["qasm"]

# Simulate with Aer for better performance on larger circuits
result = await simulate_with_aer(qasm, shots=10000)
print(f"Measurement counts: {result['counts']}")
print(f"Success: {result['success']}")
```

### Circuit Visualization

```python
# Create a circuit
circuit = await create_quantum_circuit(3, 0)
qasm = circuit["circuit"]["qasm"]
result = await add_gates_to_circuit(qasm, "h 0; cx 0 1; cx 1 2; h 2")
qasm = result["circuit"]["qasm"]

# Draw as ASCII art
drawing = await draw_circuit_text(qasm)
print(drawing['drawing'])

# Draw with matplotlib (returns base64 PNG)
img_result = await draw_circuit_matplotlib(qasm, style="textbook")
# img_result['image_base64'] contains the PNG data

# For long circuits, use folding
long_drawing = await draw_circuit_text(qasm, fold=40)
```

### OpenQASM 3.0 Conversion

```python
# Convert existing QASM 2.0 to QASM 3.0
result = await convert_to_qasm3(qasm2_string)
qasm3 = result['qasm3']
print(qasm3)

# Load QASM 3.0 circuit
loaded = await load_qasm3_circuit(qasm3)
# Returns circuit info with both QASM 2 and QASM 3 representations
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

# Run tests with coverage
uv run pytest --cov=src --cov-report=html

# Run specific test file
uv run pytest tests/test_qiskit_sdk.py -v
```

### Test Structure

- `tests/test_qiskit_sdk.py` - Unit tests for SDK functions
- `tests/test_integration.py` - Integration tests
- `tests/test_sync.py` - Synchronous wrapper tests
- `tests/conftest.py` - Test fixtures and configuration

### Test Coverage

The test suite covers:
- ✅ Circuit creation and manipulation
- ✅ Gate operations (single and multi-qubit)
- ✅ Transpilation with various optimization levels
- ✅ Statevector simulation
- ✅ Circuit visualization
- ✅ Random circuit generation
- ✅ Error handling and validation
- ✅ Synchronous wrappers

### Other ways of testing and debugging the server

> _**Note**: to launch the MCP inspector you will need to have [`node` and `npm`](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm)_

1. From a terminal, go into the cloned repo directory

1. Switch to the virtual environment

    ```sh
    source .venv/bin/activate
    ```

1. Run the MCP Inspector:

    ```sh
    npx @modelcontextprotocol/inspector uv run qiskit-mcp-server
    ```

1. Open your browser to the URL shown in the console message e.g.,

    ```
    MCP Inspector is up and running at http://localhost:5173
    ```

## Contributing

Contributions are welcome! Areas for improvement:

- Support for additional quantum gates (U, RX, RY, RZ, etc.)
- Circuit composition and decomposition
- More visualization options
- Quantum state tomography
- Circuit optimization strategies
- Integration with Qiskit Runtime for hardware execution

## License

This project is licensed under the Apache License 2.0.

## Resources

- [Qiskit Documentation](https://docs.quantum.ibm.com/api/qiskit)
- [Qiskit Tutorials](https://quantum.cloud.ibm.com/docs/en/guides/intro-to-qiskit)
- [Model Context Protocol](https://modelcontextprotocol.io/introduction)
- [MCP Inspector](https://github.com/modelcontextprotocol/inspector)

# Assisted by watsonx Code Assistant
