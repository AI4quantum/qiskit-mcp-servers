# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

#!/usr/bin/env python3
"""
Qiskit MCP Server

A Model Context Protocol server that provides access to Qiskit SDK functions
for quantum circuit creation, manipulation, transpilation, and visualization.

Dependencies:
- fastmcp
- qiskit
- python-dotenv
"""

import logging

from fastmcp import FastMCP

from qiskit_mcp_server.qiskit_sdk import (
    create_quantum_circuit,
    add_gates_to_circuit,
    transpile_circuit,
    get_circuit_depth,
    get_circuit_qasm,
    get_statevector,
    visualize_circuit,
    create_random_circuit,
    get_qiskit_version,
)
from qiskit_mcp_server.quantum_info import (
    create_pauli_operator,
    create_operator_from_circuit,
    create_density_matrix,
    calculate_state_fidelity,
    calculate_gate_fidelity,
    calculate_entropy,
    calculate_entanglement,
    partial_trace_state,
    expectation_value,
    random_quantum_state,
    random_density_matrix_state,
)
from qiskit_mcp_server.primitives import (
    sample_circuit,
    estimate_expectation_values,
    run_variational_estimation,
)
from qiskit_mcp_server.advanced_circuits import (
    compose_circuits,
    tensor_circuits,
    create_parametric_circuit,
    bind_parameters,
    decompose_circuit,
    add_controlled_gate,
    add_power_gate,
    create_qft_circuit,
    create_grover_operator,
    create_efficient_su2,
    create_real_amplitudes,
    create_pauli_evolution_circuit,
    create_phase_oracle_circuit,
    create_general_two_local_circuit,
    create_parametric_circuit_with_vector,
    simulate_with_aer,
    get_unitary_matrix,
    analyze_circuit,
    get_circuit_instructions,
    convert_to_qasm3,
    load_qasm3_circuit,
    draw_circuit_text,
    draw_circuit_matplotlib,
)
from qiskit_mcp_server.noise_and_mitigation import (
    create_depolarizing_noise_model,
    create_thermal_noise_model,
    simulate_with_noise,
    compare_ideal_vs_noisy,
)
from qiskit_mcp_server.state_prep_and_tomography import (
    prepare_uniform_superposition,
    prepare_w_state,
    prepare_ghz_state,
    prepare_dicke_state,
    prepare_product_state,
    generate_tomography_circuits,
    verify_state_preparation,
)
from qiskit_mcp_server.circuit_equivalence import (
    check_circuit_equivalence,
    check_unitary_equivalence,
    compare_circuit_resources,
    find_circuit_optimizations,
    verify_optimization,
)
from qiskit_mcp_server.circuit_utilities import (
    circuit_inverse,
    circuit_copy,
    circuit_reverse_bits,
    circuit_to_gate,
    circuit_to_instruction,
    load_qasm2_file,
    save_qasm2_file,
    load_qasm3_file,
    save_qasm3_file,
    convert_circuit_to_dag,
    convert_dag_to_circuit_wrapper,
    decompose_circuit as decompose_circuit_utility,
)
from qiskit_mcp_server.transpilation import (
    transpile_with_backend,
    transpile_with_coupling_map,
    transpile_with_layout_strategy,
    compare_transpilation_strategies,
    transpile_for_basis_gates,
    get_available_backends,
)
from qiskit_mcp_server.pass_manager import (
    run_preset_pass_manager,
    run_optimization_passes,
    run_analysis_passes,
    run_unroll_passes,
    run_combined_passes,
)
from qiskit_mcp_server.result_processing import (
    marginal_counts,
    marginal_distribution,
    counts_to_probabilities,
    filter_counts,
    combine_counts,
    expectation_from_counts,
    analyze_measurement_results,
)
from qiskit_mcp_server.visualization_extended import (
    plot_bloch_multivector,
    plot_state_qsphere,
    plot_state_hinton,
    plot_state_city,
    plot_state_paulivec,
    plot_histogram,
    plot_distribution,
)
from qiskit_mcp_server.circuit_library_extended import (
    create_two_local_circuit,
    create_n_local_circuit,
    create_pauli_feature_map,
    create_z_feature_map,
    create_zz_feature_map,
    create_qaoa_ansatz,
    create_and_gate,
    create_or_gate,
    create_xor_gate,
    create_hidden_linear_function,
    create_iqp_circuit,
    create_phase_estimation_circuit,
)
from qiskit_mcp_server.backend_execution import (
    list_available_backends,
    get_backend_properties,
    execute_circuit_local,
    submit_job_to_ibm,
    retrieve_job_result,
    cancel_job,
    get_job_status,
    estimate_circuit_cost,
)
from qiskit_mcp_server.algorithm_solvers import (
    run_vqe,
    evaluate_expectation_value,
    run_qaoa,
    optimize_parameters,
)
from qiskit_mcp_server.error_mitigation import (
    create_measurement_calibration,
    apply_measurement_mitigation,
    zero_noise_extrapolation,
    create_readout_error_model,
    probabilistic_error_cancellation,
    apply_dynamical_decoupling,
)
from qiskit_mcp_server.clifford_operations import (
    create_random_clifford,
    circuit_to_clifford,
    compose_cliffords,
    create_stabilizer_state,
    measure_stabilizer_state,
    check_if_clifford,
    pauli_tracking,
)
from qiskit_mcp_server.dynamic_circuits import (
    add_mid_circuit_measurement,
    add_conditional_gate,
    add_conditional_reset,
    create_teleportation_circuit,
    create_repeat_until_success_circuit,
    analyze_dynamic_circuit,
)
from qiskit_mcp_server.advanced_synthesis import (
    decompose_single_qubit_unitary,
    decompose_two_qubit_unitary,
    synthesize_clifford,
    synthesize_unitary,
    optimize_circuit_synthesis,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("Qiskit SDK")


# Tools
@mcp.tool()
async def create_quantum_circuit_tool(
    num_qubits: int, num_classical_bits: int = 0, name: str = "circuit"
):
    """Create a new quantum circuit with specified number of qubits and classical bits.

    Args:
        num_qubits: Number of quantum bits in the circuit
        num_classical_bits: Number of classical bits for measurement (default: 0)
        name: Name for the circuit (default: "circuit")
    """
    return await create_quantum_circuit(num_qubits, num_classical_bits, name)


@mcp.tool()
async def add_gates_to_circuit_tool(circuit_qasm: str, gates: str):
    """Add quantum gates to an existing circuit specified in QASM format.

    Args:
        circuit_qasm: QASM representation of the circuit
        gates: Gates to add in natural language or QASM-like format

               Standard gates: "h 0", "x 0", "cx 0 1"
               Rotation gates: "rx 1.57 0", "ry 0.785 1", "rz 3.14 0"
               Phase gate: "p 1.57 0"
               U gate: "u 1.57 0.785 3.14 0"
               Advanced: "sdg 0", "tdg 1", "sx 0", "barrier 0 1", "reset 0"

               Multiple gates: "h 0; rx 1.57 0; cx 0 1"
    """
    return await add_gates_to_circuit(circuit_qasm, gates)


@mcp.tool()
async def transpile_circuit_tool(
    circuit_qasm: str, optimization_level: int = 1, basis_gates: str = ""
):
    """Transpile a quantum circuit to optimize it and map to specific basis gates.

    Args:
        circuit_qasm: QASM representation of the circuit to transpile
        optimization_level: Optimization level (0-3, default: 1)
        basis_gates: Comma-separated list of basis gates (e.g., "cx,id,rz,sx,x")
                    If empty, uses default basis gates
    """
    return await transpile_circuit(circuit_qasm, optimization_level, basis_gates)


@mcp.tool()
async def get_circuit_depth_tool(circuit_qasm: str):
    """Get the depth of a quantum circuit.

    Args:
        circuit_qasm: QASM representation of the circuit
    """
    return await get_circuit_depth(circuit_qasm)


@mcp.tool()
async def get_circuit_qasm_tool(circuit_qasm: str):
    """Get the OpenQASM representation of a circuit (for validation/formatting).

    Args:
        circuit_qasm: QASM representation of the circuit
    """
    return await get_circuit_qasm(circuit_qasm)


@mcp.tool()
async def get_statevector_tool(circuit_qasm: str):
    """Get the statevector result from simulating a quantum circuit.

    Args:
        circuit_qasm: QASM representation of the circuit to simulate
    """
    return await get_statevector(circuit_qasm)


@mcp.tool()
async def visualize_circuit_tool(circuit_qasm: str, output_format: str = "text"):
    """Visualize a quantum circuit.

    Args:
        circuit_qasm: QASM representation of the circuit
        output_format: Output format - "text" for ASCII art, "mpl" for matplotlib description
    """
    return await visualize_circuit(circuit_qasm, output_format)


@mcp.tool()
async def create_random_circuit_tool(
    num_qubits: int, depth: int, measure: bool = False, seed: int = None
):
    """Create a random quantum circuit for testing purposes.

    Args:
        num_qubits: Number of qubits
        depth: Depth of the circuit
        measure: Whether to add measurements (default: False)
        seed: Random seed for reproducibility (default: None)
    """
    return await create_random_circuit(num_qubits, depth, measure, seed)


# Quantum Information Tools
@mcp.tool()
async def create_pauli_operator_tool(pauli_strings: str, coeffs: str = ""):
    """Create a SparsePauliOp (quantum observable) from Pauli strings.

    Args:
        pauli_strings: Comma-separated Pauli strings (e.g., "XX,YZ,ZZ")
        coeffs: Optional comma-separated coefficients (e.g., "1.0,0.5,0.25")
    """
    return await create_pauli_operator(pauli_strings, coeffs)


@mcp.tool()
async def create_operator_from_circuit_tool(circuit_qasm: str):
    """Create a unitary Operator from a quantum circuit."""
    return await create_operator_from_circuit(circuit_qasm)


@mcp.tool()
async def create_density_matrix_tool(circuit_qasm: str):
    """Create a DensityMatrix (mixed state) from a quantum circuit."""
    return await create_density_matrix(circuit_qasm)


@mcp.tool()
async def calculate_state_fidelity_tool(circuit_qasm1: str, circuit_qasm2: str):
    """Calculate fidelity between two quantum states (0=orthogonal, 1=identical)."""
    return await calculate_state_fidelity(circuit_qasm1, circuit_qasm2)


@mcp.tool()
async def calculate_gate_fidelity_tool(circuit_qasm: str, target_qasm: str):
    """Calculate average gate fidelity between two quantum operations."""
    return await calculate_gate_fidelity(circuit_qasm, target_qasm)


@mcp.tool()
async def calculate_entropy_tool(circuit_qasm: str, subsystem_qubits: str = ""):
    """Calculate von Neumann entropy of a quantum state.

    Args:
        circuit_qasm: QASM representation of the circuit
        subsystem_qubits: Optional comma-separated qubit indices for partial trace
    """
    return await calculate_entropy(circuit_qasm, subsystem_qubits)


@mcp.tool()
async def calculate_entanglement_tool(circuit_qasm: str):
    """Calculate entanglement of formation for a 2-qubit state."""
    return await calculate_entanglement(circuit_qasm)


@mcp.tool()
async def partial_trace_state_tool(circuit_qasm: str, trace_qubits: str):
    """Compute partial trace (reduced density matrix) over specified qubits.

    Args:
        circuit_qasm: QASM representation of the circuit
        trace_qubits: Comma-separated qubit indices to trace out (e.g., "0,2")
    """
    return await partial_trace_state(circuit_qasm, trace_qubits)


@mcp.tool()
async def expectation_value_tool(
    circuit_qasm: str, pauli_strings: str, coeffs: str = ""
):
    """Calculate expectation value of an observable for a quantum state.

    Args:
        circuit_qasm: QASM representation of the circuit
        pauli_strings: Comma-separated Pauli strings (e.g., "XX,YZ,ZZ")
        coeffs: Optional comma-separated coefficients
    """
    return await expectation_value(circuit_qasm, pauli_strings, coeffs)


@mcp.tool()
async def random_quantum_state_tool(num_qubits: int, seed: int = None):
    """Generate a random quantum statevector for testing."""
    return await random_quantum_state(num_qubits, seed)


@mcp.tool()
async def random_density_matrix_state_tool(
    num_qubits: int, rank: int = None, seed: int = None
):
    """Generate a random density matrix for testing.

    Args:
        num_qubits: Number of qubits
        rank: Rank of the density matrix (None for full rank)
        seed: Random seed for reproducibility
    """
    return await random_density_matrix_state(num_qubits, rank, seed)


# Primitives Tools (Sampler and Estimator)
@mcp.tool()
async def sample_circuit_tool(circuit_qasm: str, shots: int = 1024, seed: int = None):
    """Sample from a quantum circuit using the Sampler primitive.

    Args:
        circuit_qasm: QASM representation of the circuit (must have measurements)
        shots: Number of measurement shots (default: 1024)
        seed: Random seed for reproducibility
    """
    return await sample_circuit(circuit_qasm, shots, seed)


@mcp.tool()
async def estimate_expectation_values_tool(
    circuit_qasm: str, observables: str, coeffs: str = "", precision: float = 0.0
):
    """Estimate expectation values using the Estimator primitive.

    Args:
        circuit_qasm: QASM representation of the circuit
        observables: Comma-separated Pauli strings (e.g., "ZZ,XX,YY")
        coeffs: Optional comma-separated coefficients
        precision: Target precision (default: 0.0 = exact simulation)
    """
    return await estimate_expectation_values(
        circuit_qasm, observables, coeffs, precision
    )


@mcp.tool()
async def run_variational_estimation_tool(
    circuit_qasms_json: str, observables: str, coeffs: str = ""
):
    """Run Estimator on multiple circuit variations (for VQE/QAOA).

    Args:
        circuit_qasms_json: JSON list of QASM strings (circuit variations)
        observables: Comma-separated Pauli strings for the Hamiltonian
        coeffs: Optional comma-separated coefficients

    Returns:
        Expectation values for all variations with minimum energy
    """
    import json

    try:
        circuit_qasms = json.loads(circuit_qasms_json)
        return await run_variational_estimation(circuit_qasms, observables, coeffs)
    except json.JSONDecodeError as e:
        return {"status": "error", "message": f"Invalid JSON format: {str(e)}"}


# Advanced Circuit Composition Tools
@mcp.tool()
async def compose_circuits_tool(
    circuit1_qasm: str, circuit2_qasm: str, inplace: bool = False
):
    """Compose two quantum circuits sequentially.

    Args:
        circuit1_qasm: QASM representation of first circuit
        circuit2_qasm: QASM representation of second circuit to append
        inplace: If True, modify first circuit; if False, return new circuit
    """
    return await compose_circuits(circuit1_qasm, circuit2_qasm, inplace)


@mcp.tool()
async def tensor_circuits_tool(circuit1_qasm: str, circuit2_qasm: str):
    """Tensor product of two circuits (place side by side on different qubits).

    Args:
        circuit1_qasm: QASM representation of first circuit
        circuit2_qasm: QASM representation of second circuit
    """
    return await tensor_circuits(circuit1_qasm, circuit2_qasm)


@mcp.tool()
async def create_parametric_circuit_tool(
    num_qubits: int, parameter_names: str, num_classical_bits: int = 0
):
    """Create a circuit with named parameters for variational algorithms.

    Args:
        num_qubits: Number of qubits
        parameter_names: Comma-separated parameter names (e.g., "theta,phi,lambda")
        num_classical_bits: Number of classical bits
    """
    return await create_parametric_circuit(
        num_qubits, parameter_names, num_classical_bits
    )


@mcp.tool()
async def bind_parameters_tool(circuit_qasm: str, parameter_values: str):
    """Bind parameter values to a parametric circuit.

    Args:
        circuit_qasm: QASM representation of parametric circuit
        parameter_values: JSON dict of parameter names to values (e.g., '{"theta": 1.57, "phi": 3.14}')
    """
    return await bind_parameters(circuit_qasm, parameter_values)


@mcp.tool()
async def decompose_circuit_tool(circuit_qasm: str, reps: int = 1):
    """Decompose circuit gates into basis gates.

    Args:
        circuit_qasm: QASM representation of circuit
        reps: Number of decomposition repetitions (default: 1)
    """
    return await decompose_circuit(circuit_qasm, reps)


# Advanced Gate Operations Tools
@mcp.tool()
async def add_controlled_gate_tool(
    circuit_qasm: str, gate_name: str, control_qubits: str, target_qubits: str
):
    """Add a controlled version of a gate to the circuit.

    Args:
        circuit_qasm: QASM representation of circuit
        gate_name: Name of gate to control (x, z, h, y, swap)
        control_qubits: Comma-separated control qubit indices (e.g., "0" or "0,1")
        target_qubits: Comma-separated target qubit indices
    """
    return await add_controlled_gate(
        circuit_qasm, gate_name, control_qubits, target_qubits
    )


@mcp.tool()
async def add_power_gate_tool(
    circuit_qasm: str, gate_name: str, power: float, qubit: int
):
    """Add a gate raised to a power (e.g., sqrt(X) is X^0.5).

    Args:
        circuit_qasm: QASM representation of circuit
        gate_name: Name of gate (x, y, z, h)
        power: Power to raise gate to (e.g., 0.5 for square root)
        qubit: Target qubit index
    """
    return await add_power_gate(circuit_qasm, gate_name, power, qubit)


# Circuit Library Tools
@mcp.tool()
async def create_qft_circuit_tool(
    num_qubits: int, inverse: bool = False, do_swaps: bool = True
):
    """Create a Quantum Fourier Transform circuit.

    Args:
        num_qubits: Number of qubits
        inverse: If True, create inverse QFT (default: False)
        do_swaps: If True, include swap gates (default: True)
    """
    return await create_qft_circuit(num_qubits, inverse, do_swaps)


@mcp.tool()
async def create_grover_operator_tool(num_qubits: int, oracle_qasm: str):
    """Create a Grover operator circuit for quantum search.

    Args:
        num_qubits: Number of qubits
        oracle_qasm: QASM representation of oracle circuit
    """
    return await create_grover_operator(num_qubits, oracle_qasm)


@mcp.tool()
async def create_efficient_su2_tool(
    num_qubits: int, reps: int = 3, entanglement: str = "full"
):
    """Create an EfficientSU2 variational circuit (hardware-efficient ansatz).

    Args:
        num_qubits: Number of qubits
        reps: Number of repetitions (default: 3)
        entanglement: Entanglement strategy - full, linear, or circular (default: full)
    """
    return await create_efficient_su2(num_qubits, reps, entanglement)


@mcp.tool()
async def create_real_amplitudes_tool(
    num_qubits: int, reps: int = 3, entanglement: str = "full"
):
    """Create a RealAmplitudes variational circuit (for quantum chemistry).

    Args:
        num_qubits: Number of qubits
        reps: Number of repetitions (default: 3)
        entanglement: Entanglement strategy - full, linear, or circular (default: full)
    """
    return await create_real_amplitudes(num_qubits, reps, entanglement)


@mcp.tool()
async def create_pauli_evolution_circuit_tool(
    pauli_string: str, time: float, num_qubits: int = None
):
    """Create a circuit for Pauli operator time evolution (Hamiltonian simulation).

    Implements exp(-i * time * PauliOp) for applications like VQE, Trotter evolution,
    and quantum dynamics simulation.

    Args:
        pauli_string: Pauli string (e.g., 'XYZI', 'XY', 'ZZ')
        time: Evolution time parameter
        num_qubits: Number of qubits (optional, inferred from pauli_string if not provided)
    """
    return await create_pauli_evolution_circuit(pauli_string, time, num_qubits)


@mcp.tool()
async def create_phase_oracle_circuit_tool(expression: str, num_qubits: int):
    """Create a phase oracle circuit from a boolean expression for Grover's algorithm.

    The oracle marks solutions to the boolean expression by applying a phase flip.
    Useful for quantum search and constraint satisfaction problems.

    Args:
        expression: Boolean expression (e.g., '(a & b) | (~c)', 'a ^ b')
        num_qubits: Number of qubits
    """
    return await create_phase_oracle_circuit(expression, num_qubits)


@mcp.tool()
async def create_general_two_local_circuit_tool(
    num_qubits: int,
    rotation_blocks: str = "ry",
    entanglement_blocks: str = "cx",
    entanglement: str = "full",
    reps: int = 3,
    insert_barriers: bool = False,
):
    """Create a general TwoLocal variational circuit with customizable structure.

    More flexible than EfficientSU2/RealAmplitudes - allows custom rotation and
    entanglement blocks with various entanglement patterns.

    Args:
        num_qubits: Number of qubits
        rotation_blocks: Single-qubit rotation gates (e.g., 'ry', 'rz', 'ry,rz') (default: ry)
        entanglement_blocks: Two-qubit entanglement gates (e.g., 'cx', 'cz') (default: cx)
        entanglement: Entanglement pattern - full, linear, circular, sca, pairwise (default: full)
        reps: Number of repetitions (default: 3)
        insert_barriers: Insert barriers between layers (default: False)
    """
    return await create_general_two_local_circuit(
        num_qubits,
        rotation_blocks,
        entanglement_blocks,
        entanglement,
        reps,
        insert_barriers,
    )


@mcp.tool()
async def create_parametric_circuit_with_vector_tool(
    num_qubits: int, num_parameters: int, structure: str = "ry_cx"
):
    """Create a parametric circuit using ParameterVector for convenient parameter management.

    ParameterVector provides named parameter arrays ideal for variational algorithms.
    Returns both the circuit and parameter binding information.

    Args:
        num_qubits: Number of qubits
        num_parameters: Number of parameters to create
        structure: Circuit structure - ry_cx, rz_cz, full_rotation, hardware_efficient (default: ry_cx)
    """
    return await create_parametric_circuit_with_vector(
        num_qubits, num_parameters, structure
    )


# Simulation Backend Tools
@mcp.tool()
async def simulate_with_aer_tool(
    circuit_qasm: str, shots: int = 1024, backend_name: str = "aer_simulator"
):
    """Simulate circuit using Qiskit Aer high-performance simulator.

    Args:
        circuit_qasm: QASM representation of circuit
        shots: Number of shots (default: 1024)
        backend_name: Aer backend name (default: aer_simulator)
    """
    return await simulate_with_aer(circuit_qasm, shots, backend_name)


@mcp.tool()
async def get_unitary_matrix_tool(circuit_qasm: str):
    """Get the unitary matrix representation of a circuit.

    Args:
        circuit_qasm: QASM representation of circuit
    """
    return await get_unitary_matrix(circuit_qasm)


# Circuit Analysis Tools
@mcp.tool()
async def analyze_circuit_tool(circuit_qasm: str):
    """Perform comprehensive circuit analysis including gate counts and metrics.

    Args:
        circuit_qasm: QASM representation of circuit
    """
    return await analyze_circuit(circuit_qasm)


@mcp.tool()
async def get_circuit_instructions_tool(circuit_qasm: str):
    """Get detailed list of all circuit instructions with parameters.

    Args:
        circuit_qasm: QASM representation of circuit
    """
    return await get_circuit_instructions(circuit_qasm)


# OpenQASM 3 Tools
@mcp.tool()
async def convert_to_qasm3_tool(circuit_qasm: str):
    """Convert circuit from OpenQASM 2.0 to OpenQASM 3.0 format.

    Args:
        circuit_qasm: QASM 2.0 representation of circuit
    """
    return await convert_to_qasm3(circuit_qasm)


@mcp.tool()
async def load_qasm3_circuit_tool(qasm3_str: str):
    """Load circuit from OpenQASM 3.0 format.

    Args:
        qasm3_str: QASM 3.0 representation of circuit
    """
    return await load_qasm3_circuit(qasm3_str)


# Circuit Drawing Tools
@mcp.tool()
async def draw_circuit_text_tool(circuit_qasm: str, fold: int = -1):
    """Draw circuit as text/ASCII art.

    Args:
        circuit_qasm: QASM representation of circuit
        fold: Column to fold the circuit at (-1 for no folding)
    """
    return await draw_circuit_text(circuit_qasm, fold)


@mcp.tool()
async def draw_circuit_matplotlib_tool(circuit_qasm: str, style: str = "default"):
    """Draw circuit using matplotlib (returns base64 encoded PNG image).

    Args:
        circuit_qasm: QASM representation of circuit
        style: Drawing style - default, iqp, clifford, or textbook (default: default)
    """
    return await draw_circuit_matplotlib(circuit_qasm, style)


# Noise and Error Mitigation Tools
@mcp.tool()
async def create_depolarizing_noise_model_tool(
    single_qubit_error: float = 0.001, two_qubit_error: float = 0.01
):
    """Create a depolarizing noise model for noisy simulations.

    Args:
        single_qubit_error: Error probability for single-qubit gates (default: 0.001)
        two_qubit_error: Error probability for two-qubit gates (default: 0.01)
    """
    return await create_depolarizing_noise_model(single_qubit_error, two_qubit_error)


@mcp.tool()
async def create_thermal_noise_model_tool(
    t1: float = 50000.0,
    t2: float = 70000.0,
    gate_time_1q: float = 50.0,
    gate_time_2q: float = 300.0,
):
    """Create a thermal relaxation noise model.

    Args:
        t1: T1 relaxation time in nanoseconds (default: 50µs)
        t2: T2 dephasing time in nanoseconds (default: 70µs)
        gate_time_1q: Single-qubit gate time in nanoseconds (default: 50ns)
        gate_time_2q: Two-qubit gate time in nanoseconds (default: 300ns)
    """
    return await create_thermal_noise_model(t1, t2, gate_time_1q, gate_time_2q)


@mcp.tool()
async def simulate_with_noise_tool(
    circuit_qasm: str,
    noise_type: str = "depolarizing",
    single_qubit_error: float = 0.001,
    two_qubit_error: float = 0.01,
    shots: int = 1024,
):
    """Simulate circuit with realistic noise model.

    Args:
        circuit_qasm: QASM representation of circuit
        noise_type: Type of noise (depolarizing, thermal, amplitude_damping, phase_damping)
        single_qubit_error: Single-qubit error rate (default: 0.001)
        two_qubit_error: Two-qubit error rate (default: 0.01)
        shots: Number of measurement shots (default: 1024)
    """
    return await simulate_with_noise(
        circuit_qasm, noise_type, single_qubit_error, two_qubit_error, shots
    )


@mcp.tool()
async def compare_ideal_vs_noisy_tool(
    circuit_qasm: str,
    noise_type: str = "depolarizing",
    error_rate: float = 0.01,
    shots: int = 1024,
):
    """Compare ideal and noisy simulation results.

    Args:
        circuit_qasm: QASM representation of circuit
        noise_type: Type of noise model (default: depolarizing)
        error_rate: Error rate for noise (default: 0.01)
        shots: Number of measurement shots (default: 1024)
    """
    return await compare_ideal_vs_noisy(circuit_qasm, noise_type, error_rate, shots)


# State Preparation Tools
@mcp.tool()
async def prepare_uniform_superposition_tool(num_qubits: int):
    """Prepare a uniform superposition state |+⟩^⊗n.

    Args:
        num_qubits: Number of qubits
    """
    return await prepare_uniform_superposition(num_qubits)


@mcp.tool()
async def prepare_w_state_tool(num_qubits: int):
    """Prepare a W state: (|100...⟩ + |010...⟩ + ... + |00...1⟩)/sqrt(n).

    Args:
        num_qubits: Number of qubits (minimum 2)
    """
    return await prepare_w_state(num_qubits)


@mcp.tool()
async def prepare_ghz_state_tool(num_qubits: int):
    """Prepare a GHZ state: (|00...0⟩ + |11...1⟩)/sqrt(2).

    Args:
        num_qubits: Number of qubits
    """
    return await prepare_ghz_state(num_qubits)


@mcp.tool()
async def prepare_dicke_state_tool(num_qubits: int, num_excitations: int):
    """Prepare a Dicke state with fixed number of excitations.

    Args:
        num_qubits: Number of qubits
        num_excitations: Number of |1⟩ states
    """
    return await prepare_dicke_state(num_qubits, num_excitations)


@mcp.tool()
async def prepare_product_state_tool(state_string: str):
    """Prepare a product state from a string like '0101' or '+++−'.

    Args:
        state_string: String specifying the state (use 0/1 or +/−)
    """
    return await prepare_product_state(state_string)


@mcp.tool()
async def generate_tomography_circuits_tool(
    circuit_qasm: str, measurement_basis: str = "pauli"
):
    """Generate measurement circuits for quantum state tomography.

    Args:
        circuit_qasm: QASM representation of state preparation circuit
        measurement_basis: Basis for measurements (pauli) - default: pauli
    """
    return await generate_tomography_circuits(circuit_qasm, measurement_basis)


@mcp.tool()
async def verify_state_preparation_tool(prepared_qasm: str, target_qasm: str):
    """Verify state preparation by comparing with target state.

    Args:
        prepared_qasm: QASM of circuit that prepares the state
        target_qasm: QASM of circuit that prepares the target state
    """
    return await verify_state_preparation(prepared_qasm, target_qasm)


# Circuit Equivalence and Optimization Tools
@mcp.tool()
async def check_circuit_equivalence_tool(
    circuit1_qasm: str, circuit2_qasm: str, tolerance: float = 1e-7
):
    """Check if two circuits are equivalent up to a global phase.

    Args:
        circuit1_qasm: QASM representation of first circuit
        circuit2_qasm: QASM representation of second circuit
        tolerance: Numerical tolerance for comparison (default: 1e-7)
    """
    return await check_circuit_equivalence(circuit1_qasm, circuit2_qasm, tolerance)


@mcp.tool()
async def check_unitary_equivalence_tool(circuit1_qasm: str, circuit2_qasm: str):
    """Check if two circuits implement the same unitary operation.

    Args:
        circuit1_qasm: QASM representation of first circuit
        circuit2_qasm: QASM representation of second circuit
    """
    return await check_unitary_equivalence(circuit1_qasm, circuit2_qasm)


@mcp.tool()
async def compare_circuit_resources_tool(circuit1_qasm: str, circuit2_qasm: str):
    """Compare resource usage of two circuits (depth, gates, etc.).

    Args:
        circuit1_qasm: QASM representation of first circuit
        circuit2_qasm: QASM representation of second circuit
    """
    return await compare_circuit_resources(circuit1_qasm, circuit2_qasm)


@mcp.tool()
async def find_circuit_optimizations_tool(circuit_qasm: str):
    """Find and suggest optimizations for a circuit.

    Args:
        circuit_qasm: QASM representation of circuit
    """
    return await find_circuit_optimizations(circuit_qasm)


@mcp.tool()
async def verify_optimization_tool(original_qasm: str, optimized_qasm: str):
    """Verify that an optimized circuit is equivalent to the original.

    Args:
        original_qasm: QASM representation of original circuit
        optimized_qasm: QASM representation of optimized circuit
    """
    return await verify_optimization(original_qasm, optimized_qasm)


# Circuit Utilities
@mcp.tool()
async def circuit_inverse_tool(circuit_qasm: str):
    """Get the inverse of a quantum circuit.

    Args:
        circuit_qasm: QASM representation of circuit
    """
    return await circuit_inverse(circuit_qasm)


@mcp.tool()
async def circuit_copy_tool(circuit_qasm: str, name: str = None):
    """Create a deep copy of a quantum circuit.

    Args:
        circuit_qasm: QASM representation of circuit
        name: Optional name for the copied circuit
    """
    return await circuit_copy(circuit_qasm, name)


@mcp.tool()
async def circuit_reverse_bits_tool(circuit_qasm: str):
    """Reverse the order of bits in a circuit.

    Args:
        circuit_qasm: QASM representation of circuit
    """
    return await circuit_reverse_bits(circuit_qasm)


@mcp.tool()
async def circuit_to_gate_tool(circuit_qasm: str, label: str = None):
    """Convert a quantum circuit to a Gate object.

    Args:
        circuit_qasm: QASM representation of circuit
        label: Optional label for the gate
    """
    return await circuit_to_gate(circuit_qasm, label)


@mcp.tool()
async def circuit_to_instruction_tool(circuit_qasm: str, label: str = None):
    """Convert a quantum circuit to an Instruction object.

    Args:
        circuit_qasm: QASM representation of circuit
        label: Optional label for the instruction
    """
    return await circuit_to_instruction(circuit_qasm, label)


@mcp.tool()
async def load_qasm2_file_tool(file_path: str):
    """Load a quantum circuit from a QASM 2.0 file.

    Args:
        file_path: Path to the QASM file
    """
    return await load_qasm2_file(file_path)


@mcp.tool()
async def save_qasm2_file_tool(circuit_qasm: str, file_path: str):
    """Save a quantum circuit to a QASM 2.0 file.

    Args:
        circuit_qasm: QASM representation of circuit
        file_path: Path where the file should be saved
    """
    return await save_qasm2_file(circuit_qasm, file_path)


@mcp.tool()
async def load_qasm3_file_tool(file_path: str):
    """Load a quantum circuit from a QASM 3.0 file.

    Args:
        file_path: Path to the QASM 3.0 file
    """
    return await load_qasm3_file(file_path)


@mcp.tool()
async def save_qasm3_file_tool(circuit_qasm: str, file_path: str):
    """Save a quantum circuit to a QASM 3.0 file.

    Args:
        circuit_qasm: QASM representation of circuit
        file_path: Path where the file should be saved
    """
    return await save_qasm3_file(circuit_qasm, file_path)


@mcp.tool()
async def convert_circuit_to_dag_tool(circuit_qasm: str):
    """Convert a quantum circuit to a Directed Acyclic Graph (DAG) representation.

    Args:
        circuit_qasm: QASM representation of circuit
    """
    return await convert_circuit_to_dag(circuit_qasm)


@mcp.tool()
async def convert_dag_to_circuit_tool(circuit_qasm: str):
    """Convert a DAG back to a quantum circuit (demonstration using circuit->DAG->circuit).

    Args:
        circuit_qasm: QASM representation of original circuit
    """
    return await convert_dag_to_circuit_wrapper(circuit_qasm)


@mcp.tool()
async def decompose_circuit_utility_tool(
    circuit_qasm: str, gates_to_decompose: str = None, reps: int = 1
):
    """Decompose a circuit by expanding composite gates.

    Args:
        circuit_qasm: QASM representation of circuit
        gates_to_decompose: Comma-separated list of gate names to decompose (None = all)
        reps: Number of decomposition repetitions
    """
    return await decompose_circuit_utility(circuit_qasm, gates_to_decompose, reps)


# Enhanced Transpilation
@mcp.tool()
async def transpile_with_backend_tool(
    circuit_qasm: str,
    backend_name: str,
    optimization_level: int = 1,
    seed_transpiler: int = None,
    layout_method: str = None,
    routing_method: str = None,
):
    """Transpile a circuit for a specific IBM Quantum backend.

    Args:
        circuit_qasm: QASM representation of the circuit
        backend_name: Name of the target backend (e.g., 'ibm_brisbane', 'ibm_kyoto')
        optimization_level: Optimization level (0-3), default 1
        seed_transpiler: Random seed for reproducibility (optional)
        layout_method: Layout selection method - 'trivial', 'dense', 'sabre' (optional)
        routing_method: Routing method - 'basic', 'lookahead', 'stochastic', 'sabre' (optional)
    """
    return await transpile_with_backend(
        circuit_qasm,
        backend_name,
        optimization_level,
        seed_transpiler,
        layout_method,
        routing_method,
    )


@mcp.tool()
async def transpile_with_coupling_map_tool(
    circuit_qasm: str,
    coupling_map_json: str,
    optimization_level: int = 1,
    initial_layout_json: str = None,
    layout_method: str = None,
    routing_method: str = None,
    seed_transpiler: int = None,
):
    """Transpile a circuit with a custom coupling map (hardware topology).

    Args:
        circuit_qasm: QASM representation of the circuit
        coupling_map_json: JSON array of edges [[0,1], [1,2], ...] or "linear:N" or "grid:NxM"
        optimization_level: Optimization level (0-3), default 1
        initial_layout_json: JSON dict mapping virtual to physical qubits {"0": 5, "1": 10, ...} (optional)
        layout_method: Layout selection method - 'trivial', 'dense', 'sabre' (optional)
        routing_method: Routing method - 'basic', 'lookahead', 'stochastic', 'sabre' (optional)
        seed_transpiler: Random seed for reproducibility (optional)
    """
    return await transpile_with_coupling_map(
        circuit_qasm,
        coupling_map_json,
        optimization_level,
        initial_layout_json,
        layout_method,
        routing_method,
        seed_transpiler,
    )


@mcp.tool()
async def transpile_with_layout_strategy_tool(
    circuit_qasm: str,
    layout_method: str,
    routing_method: str = "sabre",
    optimization_level: int = 2,
    basis_gates: str = None,
    coupling_map_json: str = None,
    seed_transpiler: int = None,
):
    """Transpile a circuit with specific layout and routing strategies.

    Args:
        circuit_qasm: QASM representation of the circuit
        layout_method: Layout method - 'trivial', 'dense', 'sabre'
        routing_method: Routing method - 'basic', 'lookahead', 'stochastic', 'sabre', default 'sabre'
        optimization_level: Optimization level (0-3), default 2
        basis_gates: Comma-separated basis gates (optional)
        coupling_map_json: JSON array of edges or "linear:N" or "grid:NxM" (optional)
        seed_transpiler: Random seed for reproducibility (optional)
    """
    return await transpile_with_layout_strategy(
        circuit_qasm,
        layout_method,
        routing_method,
        optimization_level,
        basis_gates,
        coupling_map_json,
        seed_transpiler,
    )


@mcp.tool()
async def compare_transpilation_strategies_tool(
    circuit_qasm: str,
    optimization_level: int = 2,
    coupling_map_json: str = None,
    seed_transpiler: int = None,
):
    """Compare different transpilation strategies on the same circuit.

    Tests multiple combinations of layout and routing methods to find the best strategy.

    Args:
        circuit_qasm: QASM representation of the circuit
        optimization_level: Optimization level (0-3), default 2
        coupling_map_json: JSON array of edges or "linear:N" or "grid:NxM" (optional)
        seed_transpiler: Random seed for reproducibility (optional)
    """
    return await compare_transpilation_strategies(
        circuit_qasm, optimization_level, coupling_map_json, seed_transpiler
    )


@mcp.tool()
async def transpile_for_basis_gates_tool(
    circuit_qasm: str,
    basis_gates: str,
    optimization_level: int = 1,
    seed_transpiler: int = None,
):
    """Transpile a circuit to a specific basis gate set.

    Args:
        circuit_qasm: QASM representation of the circuit
        basis_gates: Comma-separated basis gates (e.g., "u1,u2,u3,cx" or "sx,x,rz,cx")
        optimization_level: Optimization level (0-3), default 1
        seed_transpiler: Random seed for reproducibility (optional)
    """
    return await transpile_for_basis_gates(
        circuit_qasm, basis_gates, optimization_level, seed_transpiler
    )


@mcp.tool()
async def get_available_backends_tool():
    """Get list of available IBM Quantum backends.

    Returns information about available backends including their properties,
    number of qubits, and operational status.
    """
    return await get_available_backends()


# PassManager
@mcp.tool()
async def run_preset_pass_manager_tool(
    circuit_qasm: str,
    optimization_level: int = 1,
    backend_name: str = None,
    coupling_map_json: str = None,
    basis_gates: str = None,
    seed_transpiler: int = None,
):
    """Run a preset pass manager on a circuit.

    Uses Qiskit's built-in preset pass managers which contain optimized
    sequences of transpiler passes for different optimization levels.

    Args:
        circuit_qasm: QASM representation of the circuit
        optimization_level: Optimization level (0-3), default 1
        backend_name: Optional backend name for automatic configuration
        coupling_map_json: JSON array of edges or "linear:N" or "grid:NxM" (optional)
        basis_gates: Comma-separated basis gates (optional)
        seed_transpiler: Random seed for reproducibility (optional)
    """
    return await run_preset_pass_manager(
        circuit_qasm,
        optimization_level,
        backend_name,
        coupling_map_json,
        basis_gates,
        seed_transpiler,
    )


@mcp.tool()
async def run_optimization_passes_tool(circuit_qasm: str, iterations: int = 2):
    """Run optimization passes on a circuit.

    Applies a sequence of optimization passes including:
    - Optimize1qGates: Optimize single-qubit gates
    - CXCancellation: Cancel adjacent CX gates
    - CommutativeCancellation: Cancel commuting gates
    - RemoveDiagonalGatesBeforeMeasure: Remove unnecessary gates before measurement

    Args:
        circuit_qasm: QASM representation of the circuit
        iterations: Number of optimization iterations, default 2
    """
    return await run_optimization_passes(circuit_qasm, iterations)


@mcp.tool()
async def run_analysis_passes_tool(circuit_qasm: str):
    """Run analysis passes on a circuit to gather detailed metrics.

    Applies analysis passes to extract circuit properties:
    - Depth: Circuit depth calculation
    - Size: Circuit size (gate count)
    - Width: Circuit width (qubit count)
    - CountOps: Count operations by type
    - NumTensorFactors: Count tensor factors

    Args:
        circuit_qasm: QASM representation of the circuit
    """
    return await run_analysis_passes(circuit_qasm)


@mcp.tool()
async def run_unroll_passes_tool(circuit_qasm: str, basis_gates: str):
    """Run unrolling passes to decompose gates to a basis set.

    Uses the Unroller pass to decompose all gates to the specified basis.

    Args:
        circuit_qasm: QASM representation of the circuit
        basis_gates: Comma-separated basis gates (e.g., "u3,cx" or "sx,x,rz,cx")
    """
    return await run_unroll_passes(circuit_qasm, basis_gates)


@mcp.tool()
async def run_combined_passes_tool(
    circuit_qasm: str,
    basis_gates: str,
    optimization_level: int = 2,
    coupling_map_json: str = None,
):
    """Run a combined pipeline of unrolling and optimization passes.

    Executes a multi-stage pipeline:
    1. Unroll to basis gates
    2. Optimize (multiple iterations based on level)
    3. Final cleanup

    Args:
        circuit_qasm: QASM representation of the circuit
        basis_gates: Comma-separated basis gates
        optimization_level: Number of optimization iterations (0-3), default 2
        coupling_map_json: Optional coupling map JSON
    """
    return await run_combined_passes(
        circuit_qasm, basis_gates, optimization_level, coupling_map_json
    )


# Result Processing Tools
@mcp.tool()
async def marginal_counts_tool(counts_json: str, indices: str):
    """Marginalize measurement counts to specific qubit indices.

    Extracts outcomes for a subset of qubits, summing over the rest.
    Essential for analyzing subsystem behavior in multi-qubit circuits.

    Args:
        counts_json: JSON string of counts (e.g., '{"000": 100, "111": 200}')
        indices: Comma-separated qubit indices to keep (e.g., "0,2")
    """
    return await marginal_counts(counts_json, indices)


@mcp.tool()
async def marginal_distribution_tool(counts_json: str, indices: str):
    """Compute marginal probability distribution for specific qubits.

    Similar to marginal_counts but returns normalized probabilities instead of raw counts.

    Args:
        counts_json: JSON string of counts
        indices: Comma-separated qubit indices to keep
    """
    return await marginal_distribution(counts_json, indices)


@mcp.tool()
async def counts_to_probabilities_tool(counts_json: str):
    """Convert measurement counts to probability distribution.

    Normalizes raw counts by total number of shots to get probabilities.

    Args:
        counts_json: JSON string of counts
    """
    return await counts_to_probabilities(counts_json)


@mcp.tool()
async def filter_counts_tool(counts_json: str, pattern: str):
    """Filter measurement counts by bit pattern matching.

    Keeps only outcomes matching a specific bit pattern (use 'x' for wildcards).
    Example: pattern "10x" matches "100" and "101" but not "110".

    Args:
        counts_json: JSON string of counts
        pattern: Bit pattern with 'x' wildcards (e.g., "10x", "x1x")
    """
    return await filter_counts(counts_json, pattern)


@mcp.tool()
async def combine_counts_tool(counts_json_list: str):
    """Combine multiple count dictionaries into one.

    Merges counts from multiple experiments/shots by summing matching outcomes.

    Args:
        counts_json_list: JSON list of count dictionaries
    """
    return await combine_counts(counts_json_list)


@mcp.tool()
async def expectation_from_counts_tool(counts_json: str, operator: str):
    """Calculate expectation value from measurement counts.

    Computes <operator> using measurement outcomes in computational basis.

    Args:
        counts_json: JSON string of counts
        operator: Diagonal operator as comma-separated eigenvalues (e.g., "1,-1,1,-1")
    """
    return await expectation_from_counts(counts_json, operator)


@mcp.tool()
async def analyze_measurement_results_tool(counts_json: str):
    """Comprehensive analysis of measurement results.

    Returns statistics including most/least likely outcomes, entropy, variance, etc.

    Args:
        counts_json: JSON string of counts
    """
    return await analyze_measurement_results(counts_json)


# Visualization Extended Tools
@mcp.tool()
async def plot_bloch_multivector_tool(circuit_qasm: str):
    """Visualize quantum state on Bloch sphere (multi-qubit).

    Creates Bloch sphere representation showing each qubit's state vector.
    For n qubits, creates n Bloch spheres showing individual qubit states.

    Args:
        circuit_qasm: QASM representation of circuit (measurements removed automatically)

    Returns:
        Base64-encoded PNG image
    """
    return await plot_bloch_multivector(circuit_qasm)


@mcp.tool()
async def plot_state_qsphere_tool(circuit_qasm: str):
    """Visualize quantum state using Q-sphere representation.

    Q-sphere shows probability amplitudes as circles on a sphere.
    Size represents amplitude magnitude, color represents phase.

    Args:
        circuit_qasm: QASM representation of circuit

    Returns:
        Base64-encoded PNG image
    """
    return await plot_state_qsphere(circuit_qasm)


@mcp.tool()
async def plot_state_hinton_tool(circuit_qasm: str):
    """Visualize density matrix using Hinton diagram.

    Hinton diagram shows matrix elements as squares - size = magnitude, color = sign.
    Useful for visualizing mixed states and entanglement.

    Args:
        circuit_qasm: QASM representation of circuit

    Returns:
        Base64-encoded PNG image
    """
    return await plot_state_hinton(circuit_qasm)


@mcp.tool()
async def plot_state_city_tool(circuit_qasm: str):
    """Visualize quantum state using city/bar plot representation.

    3D bar chart showing real and imaginary parts of state amplitudes.

    Args:
        circuit_qasm: QASM representation of circuit

    Returns:
        Base64-encoded PNG image
    """
    return await plot_state_city(circuit_qasm)


@mcp.tool()
async def plot_state_paulivec_tool(circuit_qasm: str):
    """Visualize quantum state using Pauli vector representation.

    Shows decomposition of density matrix in Pauli basis.
    Useful for understanding quantum state structure.

    Args:
        circuit_qasm: QASM representation of circuit

    Returns:
        Base64-encoded PNG image
    """
    return await plot_state_paulivec(circuit_qasm)


@mcp.tool()
async def plot_histogram_tool(counts_json: str):
    """Plot measurement outcome histogram.

    Standard bar chart showing measurement counts/probabilities.
    Most commonly used visualization for measurement results.

    Args:
        counts_json: JSON string of counts

    Returns:
        Base64-encoded PNG image
    """
    return await plot_histogram(counts_json)


@mcp.tool()
async def plot_distribution_tool(counts_json: str):
    """Plot probability distribution from measurement counts.

    Normalized probability distribution visualization.

    Args:
        counts_json: JSON string of counts

    Returns:
        Base64-encoded PNG image
    """
    return await plot_distribution(counts_json)


# Circuit Library Extended Tools
@mcp.tool()
async def create_two_local_circuit_tool(
    num_qubits: int,
    rotation_blocks: str = "ry",
    entanglement_blocks: str = "cx",
    reps: int = 3,
    entanglement: str = "full",
):
    """Create TwoLocal variational circuit (alternative implementation).

    Note: Similar to create_general_two_local_circuit_tool but from circuit_library_extended.
    Provides additional flexibility for variational quantum algorithms.

    Args:
        num_qubits: Number of qubits
        rotation_blocks: Rotation gates (e.g., "ry", "rx,rz")
        entanglement_blocks: Entanglement gates (default: cx)
        reps: Number of repetitions
        entanglement: Pattern (full, linear, circular, sca)
    """
    return await create_two_local_circuit(
        num_qubits, rotation_blocks, entanglement_blocks, reps, entanglement
    )


@mcp.tool()
async def create_n_local_circuit_tool(
    num_qubits: int,
    rotation_blocks: str = "ry",
    entanglement_blocks: str = "cx",
    num_qubits_entanglement: int = 2,
    reps: int = 3,
    entanglement: str = "full",
):
    """Create NLocal variational circuit (generalization of TwoLocal).

    Allows entanglement blocks acting on more than 2 qubits.

    Args:
        num_qubits: Number of qubits
        rotation_blocks: Rotation gates
        entanglement_blocks: Entanglement gates
        num_qubits_entanglement: Number of qubits per entanglement block (default: 2)
        reps: Number of repetitions
        entanglement: Pattern (full, linear, circular)
    """
    return await create_n_local_circuit(
        num_qubits,
        rotation_blocks,
        entanglement_blocks,
        num_qubits_entanglement,
        reps,
    )


@mcp.tool()
async def create_pauli_feature_map_tool(
    feature_dimension: int,
    reps: int = 2,
    paulis: str = None,
):
    """Create Pauli feature map for quantum machine learning.

    Encodes classical data into quantum states using Pauli rotations.
    Essential for quantum kernel methods and QML.

    Args:
        feature_dimension: Number of features (= number of qubits)
        reps: Number of encoding repetitions (default: 2)
        paulis: Pauli strings to use (optional, defaults to all)
    """
    return await create_pauli_feature_map(feature_dimension, reps, paulis)


@mcp.tool()
async def create_z_feature_map_tool(
    feature_dimension: int,
    reps: int = 2,
):
    """Create Z feature map for quantum machine learning.

    Simple feature encoding using Z rotations only.
    Fast and efficient for initial QML experiments.

    Args:
        feature_dimension: Number of features
        reps: Number of encoding repetitions
    """
    return await create_z_feature_map(feature_dimension, reps)


@mcp.tool()
async def create_zz_feature_map_tool(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str = "full",
):
    """Create ZZ feature map with entangling gates for QML.

    Encodes features with Z rotations + ZZ entangling gates.
    Captures feature interactions, more expressive than Z feature map.

    Args:
        feature_dimension: Number of features
        reps: Number of encoding repetitions
        entanglement: Entanglement pattern (full, linear, circular)
    """
    return await create_zz_feature_map(feature_dimension, reps, entanglement)


@mcp.tool()
async def create_qaoa_ansatz_tool(
    cost_operator: str,
    reps: int = 1,
):
    """Create QAOA ansatz circuit for combinatorial optimization.

    Quantum Approximate Optimization Algorithm ansatz.
    Used for solving optimization problems like MaxCut, graph coloring, etc.

    Args:
        cost_operator: Cost Hamiltonian as Pauli string (e.g., "ZIZI + IZZI")
        reps: Number of QAOA layers (p parameter)
    """
    return await create_qaoa_ansatz(cost_operator, reps)


@mcp.tool()
async def create_and_gate_tool(num_variable_qubits: int):
    """Create quantum AND gate circuit.

    Boolean AND oracle for quantum algorithms.
    Outputs 1 if all inputs are 1, else 0.

    Args:
        num_variable_qubits: Number of input qubits
    """
    return await create_and_gate(num_variable_qubits)


@mcp.tool()
async def create_or_gate_tool(num_variable_qubits: int):
    """Create quantum OR gate circuit.

    Boolean OR oracle for quantum algorithms.
    Outputs 1 if any input is 1, else 0.

    Args:
        num_variable_qubits: Number of input qubits
    """
    return await create_or_gate(num_variable_qubits)


@mcp.tool()
async def create_xor_gate_tool(num_qubits: int):
    """Create quantum XOR gate circuit.

    Boolean XOR oracle (parity check).
    Outputs 1 if odd number of inputs are 1.

    Args:
        num_qubits: Number of input qubits
    """
    return await create_xor_gate(num_qubits)


@mcp.tool()
async def create_hidden_linear_function_tool(num_qubits: int):
    """Create hidden linear function circuit.

    Quantum algorithm demonstrating oracle separation.
    Used in complexity theory and algorithm research.

    Args:
        num_qubits: Number of qubits
    """
    return await create_hidden_linear_function(num_qubits)


@mcp.tool()
async def create_iqp_circuit_tool(num_qubits: int):
    """Create Instantaneous Quantum Polynomial (IQP) circuit.

    Circuit class believed hard to simulate classically.
    Used for quantum supremacy demonstrations.

    Args:
        num_qubits: Number of qubits
    """
    return await create_iqp_circuit(num_qubits)


@mcp.tool()
async def create_phase_estimation_circuit_tool(
    unitary_qasm: str,
    num_evaluation_qubits: int,
):
    """Create quantum phase estimation circuit.

    Fundamental algorithm for estimating eigenvalues.
    Core subroutine in Shor's algorithm, HHL, and VQE.

    Args:
        unitary_qasm: QASM representation of unitary operator
        num_evaluation_qubits: Number of qubits for phase precision
    """
    return await create_phase_estimation_circuit(num_evaluation_qubits, unitary_qasm)


# Backend Execution Tools
@mcp.tool()
async def list_available_backends_tool(simulator_only: bool = False):
    """List all available quantum backends.

    Returns information about available backends including IBM Quantum hardware,
    Aer simulators, and statevector simulators.

    Args:
        simulator_only: If True, only list simulator backends (default: False)
    """
    return await list_available_backends(simulator_only)


@mcp.tool()
async def get_backend_properties_tool(backend_name: str):
    """Get detailed properties of a specific backend.

    Retrieves configuration, coupling map, basis gates, and status information.

    Args:
        backend_name: Name of the backend (e.g., "ibm_kyoto", "aer_simulator")
    """
    return await get_backend_properties(backend_name)


@mcp.tool()
async def execute_circuit_local_tool(
    circuit_qasm: str,
    shots: int = 1024,
    backend_name: str = "statevector_simulator",
):
    """Execute circuit on local simulator backend.

    Runs circuit on local simulators (statevector or Aer) and returns measurement counts.

    Args:
        circuit_qasm: QASM representation of circuit
        shots: Number of measurement shots (default: 1024)
        backend_name: Local backend name (default: statevector_simulator)
    """
    return await execute_circuit_local(circuit_qasm, shots, backend_name)


@mcp.tool()
async def submit_job_to_ibm_tool(
    circuit_qasm: str,
    backend_name: str,
    shots: int = 1024,
):
    """Submit job to IBM Quantum backend.

    Transpiles circuit for the target backend and submits for execution.
    Returns job ID for later retrieval.

    Args:
        circuit_qasm: QASM representation of circuit
        backend_name: IBM backend name (e.g., "ibm_kyoto")
        shots: Number of measurement shots (default: 1024)
    """
    return await submit_job_to_ibm(circuit_qasm, backend_name, shots)


@mcp.tool()
async def retrieve_job_result_tool(job_id: str):
    """Retrieve results from a submitted IBM Quantum job.

    Check job status and retrieve measurement counts if completed.

    Args:
        job_id: Job ID from submit_job_to_ibm
    """
    return await retrieve_job_result(job_id)


@mcp.tool()
async def cancel_job_tool(job_id: str):
    """Cancel a submitted IBM Quantum job.

    Cancels a pending or running job.

    Args:
        job_id: Job ID to cancel
    """
    return await cancel_job(job_id)


@mcp.tool()
async def get_job_status_tool(job_id: str):
    """Get status of a submitted IBM Quantum job.

    Returns job status, queue position, and backend information.

    Args:
        job_id: Job ID to check
    """
    return await get_job_status(job_id)


@mcp.tool()
async def estimate_circuit_cost_tool(
    circuit_qasm: str,
    backend_name: str,
):
    """Estimate resource cost for running circuit on a backend.

    Provides gate count, depth, and estimated execution time/cost.

    Args:
        circuit_qasm: QASM representation of circuit
        backend_name: Backend to estimate for
    """
    return await estimate_circuit_cost(circuit_qasm, backend_name)


# Algorithm Solver Tools
@mcp.tool()
async def run_vqe_tool(
    hamiltonian: str,
    ansatz_qasm: str,
    initial_point: str = None,
    optimizer: str = "COBYLA",
    max_iterations: int = 100,
):
    """Run Variational Quantum Eigensolver (VQE) algorithm.

    Finds ground state energy of a Hamiltonian using variational optimization.
    Core algorithm for quantum chemistry and materials science.

    Args:
        hamiltonian: Hamiltonian as Pauli string (e.g., "II + 0.5*ZZ + 0.3*XX")
        ansatz_qasm: QASM representation of parametric ansatz circuit
        initial_point: JSON list of initial parameter values (optional)
        optimizer: Classical optimizer (COBYLA, SLSQP, NELDER-MEAD)
        max_iterations: Maximum optimization iterations (default: 100)
    """
    return await run_vqe(
        hamiltonian, ansatz_qasm, initial_point, optimizer, max_iterations
    )


@mcp.tool()
async def evaluate_expectation_value_tool(
    circuit_qasm: str,
    observable: str,
):
    """Evaluate expectation value <ψ|O|ψ> for a circuit and observable.

    Computes quantum expectation value for any observable.

    Args:
        circuit_qasm: QASM representation of circuit
        observable: Observable as Pauli string (e.g., "ZZ", "0.5*XX + 0.3*YY")
    """
    return await evaluate_expectation_value(circuit_qasm, observable)


@mcp.tool()
async def run_qaoa_tool(
    cost_hamiltonian: str,
    num_qubits: int,
    num_layers: int = 1,
    optimizer: str = "COBYLA",
    max_iterations: int = 100,
    initial_point: str = None,
):
    """Run Quantum Approximate Optimization Algorithm (QAOA).

    Solves combinatorial optimization problems like MaxCut, graph coloring, etc.
    One of the most important near-term quantum algorithms.

    Args:
        cost_hamiltonian: Cost Hamiltonian as Pauli string (e.g., "ZIZI + IZZI")
        num_qubits: Number of qubits
        num_layers: Number of QAOA layers/p parameter (default: 1)
        optimizer: Classical optimizer (COBYLA, SLSQP)
        max_iterations: Maximum optimization iterations (default: 100)
        initial_point: JSON list of initial parameter values (optional)
    """
    return await run_qaoa(
        cost_hamiltonian,
        num_qubits,
        num_layers,
        optimizer,
        max_iterations,
        initial_point,
    )


@mcp.tool()
async def optimize_parameters_tool(
    circuit_qasm: str,
    cost_function_type: str,
    observable: str = None,
    optimizer: str = "COBYLA",
    max_iterations: int = 100,
    initial_point: str = None,
):
    """Generic parameter optimization for variational circuits.

    Optimizes circuit parameters using expectation-based or sampling-based cost functions.

    Args:
        circuit_qasm: QASM representation of parametric circuit
        cost_function_type: Type of cost function (expectation, sampling)
        observable: Observable for expectation-based cost (Pauli string, optional)
        optimizer: Classical optimizer (default: COBYLA)
        max_iterations: Maximum iterations (default: 100)
        initial_point: JSON list of initial parameter values (optional)
    """
    return await optimize_parameters(
        circuit_qasm,
        cost_function_type,
        observable,
        optimizer,
        max_iterations,
        initial_point,
    )


# Error Mitigation Tools
@mcp.tool()
async def create_measurement_calibration_tool(
    num_qubits: int,
    qubit_list: str = None,
):
    """Create measurement calibration circuits.

    Generates circuits to characterize measurement errors for error mitigation.

    Args:
        num_qubits: Total number of qubits
        qubit_list: Comma-separated list of qubits to calibrate (optional, defaults to all)
    """
    return await create_measurement_calibration(num_qubits, qubit_list)


@mcp.tool()
async def apply_measurement_mitigation_tool(
    measured_counts_json: str,
    calibration_results_json: str,
):
    """Apply measurement error mitigation to measurement results.

    Uses calibration data to correct measurement errors via matrix inversion.

    Args:
        measured_counts_json: JSON string of measured counts
        calibration_results_json: JSON string of calibration measurement results
    """
    return await apply_measurement_mitigation(
        measured_counts_json, calibration_results_json
    )


@mcp.tool()
async def zero_noise_extrapolation_tool(
    circuit_qasm: str,
    observable: str,
    scale_factors: str = "1.0,1.5,2.0,2.5,3.0",
    extrapolation_method: str = "linear",
):
    """Apply zero-noise extrapolation (ZNE) to mitigate errors.

    Runs circuit at multiple noise levels and extrapolates to zero noise.
    Powerful error mitigation technique for NISQ devices.

    Args:
        circuit_qasm: QASM representation of circuit
        observable: Observable as Pauli string
        scale_factors: Comma-separated noise scale factors (default: 1.0,1.5,2.0,2.5,3.0)
        extrapolation_method: Extrapolation method (linear, exponential, polynomial)
    """
    return await zero_noise_extrapolation(
        circuit_qasm, observable, scale_factors, extrapolation_method
    )


@mcp.tool()
async def create_readout_error_model_tool(error_rates_json: str):
    """Create a readout error model from error rates.

    Builds confusion matrices for measurement error simulation and mitigation.

    Args:
        error_rates_json: JSON dict of qubit error rates (e.g., {"0": 0.02, "1": 0.03})
    """
    return await create_readout_error_model(error_rates_json)


@mcp.tool()
async def probabilistic_error_cancellation_tool(
    circuit_qasm: str,
    noise_model_json: str,
    num_samples: int = 100,
):
    """Apply probabilistic error cancellation (PEC).

    Advanced error mitigation using quasi-probability representations.
    Note: Full PEC requires detailed noise characterization.

    Args:
        circuit_qasm: QASM representation of circuit
        noise_model_json: JSON representation of noise model
        num_samples: Number of samples for quasi-probability sampling (default: 100)
    """
    return await probabilistic_error_cancellation(
        circuit_qasm, noise_model_json, num_samples
    )


@mcp.tool()
async def apply_dynamical_decoupling_tool(
    circuit_qasm: str,
    dd_sequence: str = "XY4",
):
    """Apply dynamical decoupling sequence to mitigate decoherence.

    Inserts pulse sequences during idle times to suppress noise.
    Effective for reducing T1/T2 errors.

    Args:
        circuit_qasm: QASM representation of circuit
        dd_sequence: DD sequence type (XY4, CPMG, Uhrig) (default: XY4)
    """
    return await apply_dynamical_decoupling(circuit_qasm, dd_sequence)


# Clifford & Stabilizer Tools
@mcp.tool()
async def create_random_clifford_tool(num_qubits: int, seed: int = None):
    """Create a random Clifford circuit.

    Clifford circuits are efficiently simulable classically.
    Essential for randomized benchmarking and error characterization.

    Args:
        num_qubits: Number of qubits
        seed: Random seed for reproducibility (optional)
    """
    return await create_random_clifford(num_qubits, seed)


@mcp.tool()
async def circuit_to_clifford_tool(circuit_qasm: str):
    """Convert circuit to Clifford representation.

    Verifies that circuit is Clifford and provides stabilizer tableau.
    Enables efficient classical simulation.

    Args:
        circuit_qasm: QASM representation (must be Clifford gates only)
    """
    return await circuit_to_clifford(circuit_qasm)


@mcp.tool()
async def compose_cliffords_tool(clifford1_qasm: str, clifford2_qasm: str):
    """Compose two Clifford circuits efficiently.

    Clifford composition preserves the Clifford property.

    Args:
        clifford1_qasm: QASM of first Clifford circuit
        clifford2_qasm: QASM of second Clifford circuit
    """
    return await compose_cliffords(clifford1_qasm, clifford2_qasm)


@mcp.tool()
async def create_stabilizer_state_tool(circuit_qasm: str):
    """Create stabilizer state from Clifford circuit.

    Stabilizer states enable efficient quantum error correction simulation.

    Args:
        circuit_qasm: QASM of Clifford circuit
    """
    return await create_stabilizer_state(circuit_qasm)


@mcp.tool()
async def measure_stabilizer_state_tool(circuit_qasm: str, qubits: str = None):
    """Measure a stabilizer state efficiently.

    Stabilizer measurements can be simulated classically.

    Args:
        circuit_qasm: QASM of Clifford circuit
        qubits: Comma-separated qubit indices (optional, defaults to all)
    """
    return await measure_stabilizer_state(circuit_qasm, qubits)


@mcp.tool()
async def check_if_clifford_tool(circuit_qasm: str):
    """Check if a circuit is Clifford.

    Determines if circuit can be efficiently simulated via stabilizer formalism.

    Args:
        circuit_qasm: QASM representation of circuit
    """
    return await check_if_clifford(circuit_qasm)


@mcp.tool()
async def pauli_tracking_tool(circuit_qasm: str, initial_pauli: str):
    """Track Pauli operator evolution through Clifford circuit.

    Essential for fault-tolerant quantum computing and error propagation analysis.

    Args:
        circuit_qasm: QASM of Clifford circuit
        initial_pauli: Initial Pauli string (e.g., "XYZ", "IIZX")
    """
    return await pauli_tracking(circuit_qasm, initial_pauli)


# Dynamic Circuit Tools
@mcp.tool()
async def add_mid_circuit_measurement_tool(
    circuit_qasm: str,
    measure_qubit: int,
    classical_bit: int,
):
    """Add mid-circuit measurement for dynamic quantum circuits.

    Enables measurement before circuit end for feedback and control.

    Args:
        circuit_qasm: QASM representation of circuit
        measure_qubit: Qubit index to measure
        classical_bit: Classical bit to store result
    """
    return await add_mid_circuit_measurement(circuit_qasm, measure_qubit, classical_bit)


@mcp.tool()
async def add_conditional_gate_tool(
    circuit_qasm: str,
    gate_name: str,
    target_qubit: int,
    classical_bit: int,
    condition_value: int = 1,
):
    """Add gate conditioned on classical bit value.

    Implements classical control flow for dynamic circuits.

    Args:
        circuit_qasm: QASM representation of circuit
        gate_name: Gate to apply (x, y, z, h, etc.)
        target_qubit: Qubit to apply gate to
        classical_bit: Classical bit to condition on
        condition_value: Classical value to trigger gate (0 or 1, default: 1)
    """
    return await add_conditional_gate(
        circuit_qasm, gate_name, target_qubit, classical_bit, condition_value
    )


@mcp.tool()
async def add_conditional_reset_tool(
    circuit_qasm: str,
    target_qubit: int,
    classical_bit: int,
    reset_on_value: int = 1,
):
    """Add conditional reset based on measurement outcome.

    Essential for quantum error correction protocols.

    Args:
        circuit_qasm: QASM representation of circuit
        target_qubit: Qubit to reset
        classical_bit: Classical bit to condition on
        reset_on_value: Classical value to trigger reset (default: 1)
    """
    return await add_conditional_reset(
        circuit_qasm, target_qubit, classical_bit, reset_on_value
    )


@mcp.tool()
async def create_teleportation_circuit_tool():
    """Create quantum teleportation circuit.

    Demonstrates mid-circuit measurement and conditional operations.
    Classic example of dynamic quantum circuit.
    """
    return await create_teleportation_circuit()


@mcp.tool()
async def create_repeat_until_success_circuit_tool(
    operation_qasm: str,
    max_attempts: int = 3,
):
    """Create repeat-until-success circuit pattern.

    Probabilistic gate implementation with post-selection.

    Args:
        operation_qasm: QASM of operation to repeat
        max_attempts: Maximum number of attempts (default: 3)
    """
    return await create_repeat_until_success_circuit(operation_qasm, max_attempts)


@mcp.tool()
async def analyze_dynamic_circuit_tool(circuit_qasm: str):
    """Analyze circuit for dynamic features.

    Identifies mid-circuit measurements, conditional operations, and resets.

    Args:
        circuit_qasm: QASM representation of circuit
    """
    return await analyze_dynamic_circuit(circuit_qasm)


# Advanced Synthesis Tools
@mcp.tool()
async def decompose_single_qubit_unitary_tool(
    unitary_matrix: str,
    basis: str = "U3",
):
    """Decompose arbitrary single-qubit unitary into basis gates.

    Converts any single-qubit operation to implementable gates.

    Args:
        unitary_matrix: JSON string of 2x2 unitary matrix
        basis: Basis for decomposition (U3, ZYZ, ZXZ, XYX)
    """
    return await decompose_single_qubit_unitary(unitary_matrix, basis)


@mcp.tool()
async def decompose_two_qubit_unitary_tool(
    unitary_matrix: str,
    basis_gates: str = "cx",
):
    """Decompose arbitrary two-qubit unitary using KAK decomposition.

    Implements arbitrary two-qubit gates with optimal CNOT count.

    Args:
        unitary_matrix: JSON string of 4x4 unitary matrix
        basis_gates: Basis gates (cx, cz, iswap, default: cx)
    """
    return await decompose_two_qubit_unitary(unitary_matrix, basis_gates)


@mcp.tool()
async def synthesize_clifford_tool(
    clifford_tableau: str = None,
    num_qubits: int = None,
    random: bool = False,
):
    """Synthesize Clifford circuit from tableau or generate random.

    Creates optimal Clifford implementation.

    Args:
        clifford_tableau: JSON representation of Clifford tableau (optional)
        num_qubits: Number of qubits for random Clifford (if random=True)
        random: Generate random Clifford (default: False)
    """
    return await synthesize_clifford(clifford_tableau, num_qubits, random)


@mcp.tool()
async def synthesize_unitary_tool(
    unitary_matrix: str,
    num_qubits: int,
    basis_gates: str = "cx,u3",
    optimization_level: int = 2,
):
    """Synthesize arbitrary multi-qubit unitary into basis gates.

    Decomposes any unitary operation into implementable gates.

    Args:
        unitary_matrix: JSON string of unitary matrix
        num_qubits: Number of qubits
        basis_gates: Comma-separated basis gates (default: cx,u3)
        optimization_level: Optimization level 0-3 (default: 2)
    """
    return await synthesize_unitary(
        unitary_matrix, num_qubits, basis_gates, optimization_level
    )


@mcp.tool()
async def optimize_circuit_synthesis_tool(
    circuit_qasm: str,
    target_basis: str = "cx,u3",
    optimization_level: int = 3,
):
    """Re-synthesize circuit for better gate count and depth.

    Applies advanced optimization techniques to reduce circuit resources.

    Args:
        circuit_qasm: QASM representation of circuit
        target_basis: Target basis gates (default: cx,u3)
        optimization_level: Optimization level 0-3 (default: 3)
    """
    return await optimize_circuit_synthesis(
        circuit_qasm, target_basis, optimization_level
    )


# Resources
@mcp.resource("qiskit://version", mime_type="application/json")
async def get_qiskit_version_resource():
    """Get current Qiskit version information."""
    return await get_qiskit_version()


def main():
    """Run the server."""
    mcp.run()


if __name__ == "__main__":
    main()


# Assisted by watsonx Code Assistant
