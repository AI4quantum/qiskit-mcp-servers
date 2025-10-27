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

"""Synchronous wrappers for async Qiskit SDK functions.

This module provides synchronous versions of the async functions for use with
frameworks that don't support async operations.
"""

import asyncio
from typing import Any, Dict, Optional

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

# Apply nest_asyncio to allow running async code in environments with existing event loops
try:
    import nest_asyncio  # type: ignore[import-untyped]

    nest_asyncio.apply()
except ImportError:
    pass


def _run_async(coro):
    """Helper to run async functions synchronously.

    This handles both cases:
    - Running in a Jupyter notebook or other environment with an existing event loop
    - Running in a standard Python script without an event loop
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're in a running loop (e.g., Jupyter), use run_until_complete
            # This works because nest_asyncio allows nested loops
            return loop.run_until_complete(coro)
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop exists, create one
        return asyncio.run(coro)


def create_quantum_circuit_sync(
    num_qubits: int, num_classical_bits: int = 0, name: str = "circuit"
) -> Dict[str, Any]:
    """Create a new quantum circuit.

    Synchronous version of create_quantum_circuit.

    Args:
        num_qubits: Number of quantum bits
        num_classical_bits: Number of classical bits
        name: Circuit name

    Returns:
        Circuit information including QASM representation
    """
    return _run_async(create_quantum_circuit(num_qubits, num_classical_bits, name))


def add_gates_to_circuit_sync(circuit_qasm: str, gates: str) -> Dict[str, Any]:
    """Add gates to an existing circuit.

    Synchronous version of add_gates_to_circuit.

    Args:
        circuit_qasm: QASM representation of the circuit
        gates: Gates to add

    Returns:
        Updated circuit information
    """
    return _run_async(add_gates_to_circuit(circuit_qasm, gates))


def transpile_circuit_sync(
    circuit_qasm: str, optimization_level: int = 1, basis_gates: str = ""
) -> Dict[str, Any]:
    """Transpile a circuit.

    Synchronous version of transpile_circuit.

    Args:
        circuit_qasm: QASM representation of the circuit
        optimization_level: Optimization level (0-3)
        basis_gates: Comma-separated basis gates

    Returns:
        Transpiled circuit information
    """
    return _run_async(transpile_circuit(circuit_qasm, optimization_level, basis_gates))


def get_circuit_depth_sync(circuit_qasm: str) -> Dict[str, Any]:
    """Get circuit depth.

    Synchronous version of get_circuit_depth.

    Args:
        circuit_qasm: QASM representation of the circuit

    Returns:
        Circuit depth information
    """
    return _run_async(get_circuit_depth(circuit_qasm))


def get_circuit_qasm_sync(circuit_qasm: str) -> Dict[str, Any]:
    """Get QASM representation of a circuit.

    Synchronous version of get_circuit_qasm.

    Args:
        circuit_qasm: QASM representation of the circuit

    Returns:
        Formatted QASM string
    """
    return _run_async(get_circuit_qasm(circuit_qasm))


def get_statevector_sync(circuit_qasm: str) -> Dict[str, Any]:
    """Get the statevector from a circuit simulation.

    Synchronous version of get_statevector.

    Args:
        circuit_qasm: QASM representation of the circuit

    Returns:
        Statevector information
    """
    return _run_async(get_statevector(circuit_qasm))


def visualize_circuit_sync(
    circuit_qasm: str, output_format: str = "text"
) -> Dict[str, Any]:
    """Visualize a circuit.

    Synchronous version of visualize_circuit.

    Args:
        circuit_qasm: QASM representation of the circuit
        output_format: Output format

    Returns:
        Circuit visualization
    """
    return _run_async(visualize_circuit(circuit_qasm, output_format))


def create_random_circuit_sync(
    num_qubits: int, depth: int, measure: bool = False, seed: Optional[int] = None
) -> Dict[str, Any]:
    """Create a random circuit.

    Synchronous version of create_random_circuit.

    Args:
        num_qubits: Number of qubits
        depth: Circuit depth
        measure: Add measurements
        seed: Random seed

    Returns:
        Random circuit information
    """
    return _run_async(create_random_circuit(num_qubits, depth, measure, seed))


def get_qiskit_version_sync() -> Dict[str, Any]:
    """Get Qiskit version information.

    Synchronous version of get_qiskit_version.

    Returns:
        Version information
    """
    return _run_async(get_qiskit_version())


# Quantum Information synchronous wrappers


def create_pauli_operator_sync(pauli_strings: str, coeffs: str = "") -> Dict[str, Any]:
    """Create a SparsePauliOp from Pauli strings.

    Synchronous version of create_pauli_operator.

    Args:
        pauli_strings: Comma-separated Pauli strings
        coeffs: Optional comma-separated coefficients

    Returns:
        SparsePauliOp information
    """
    return _run_async(create_pauli_operator(pauli_strings, coeffs))


def create_operator_from_circuit_sync(circuit_qasm: str) -> Dict[str, Any]:
    """Create an Operator from a circuit.

    Synchronous version of create_operator_from_circuit.

    Args:
        circuit_qasm: QASM representation of the circuit

    Returns:
        Operator information
    """
    return _run_async(create_operator_from_circuit(circuit_qasm))


def create_density_matrix_sync(circuit_qasm: str) -> Dict[str, Any]:
    """Create a DensityMatrix from a circuit.

    Synchronous version of create_density_matrix.

    Args:
        circuit_qasm: QASM representation of the circuit

    Returns:
        DensityMatrix information
    """
    return _run_async(create_density_matrix(circuit_qasm))


def calculate_state_fidelity_sync(
    circuit_qasm1: str, circuit_qasm2: str
) -> Dict[str, Any]:
    """Calculate fidelity between two quantum states.

    Synchronous version of calculate_state_fidelity.

    Args:
        circuit_qasm1: QASM representation of first circuit
        circuit_qasm2: QASM representation of second circuit

    Returns:
        Fidelity value
    """
    return _run_async(calculate_state_fidelity(circuit_qasm1, circuit_qasm2))


def calculate_gate_fidelity_sync(circuit_qasm: str, target_qasm: str) -> Dict[str, Any]:
    """Calculate gate fidelity between two operations.

    Synchronous version of calculate_gate_fidelity.

    Args:
        circuit_qasm: QASM representation of actual circuit
        target_qasm: QASM representation of target circuit

    Returns:
        Gate fidelity information
    """
    return _run_async(calculate_gate_fidelity(circuit_qasm, target_qasm))


def calculate_entropy_sync(
    circuit_qasm: str, subsystem_qubits: str = ""
) -> Dict[str, Any]:
    """Calculate von Neumann entropy.

    Synchronous version of calculate_entropy.

    Args:
        circuit_qasm: QASM representation of the circuit
        subsystem_qubits: Optional comma-separated qubit indices

    Returns:
        Entropy value
    """
    return _run_async(calculate_entropy(circuit_qasm, subsystem_qubits))


def calculate_entanglement_sync(circuit_qasm: str) -> Dict[str, Any]:
    """Calculate entanglement of formation.

    Synchronous version of calculate_entanglement.

    Args:
        circuit_qasm: QASM representation of a 2-qubit circuit

    Returns:
        Entanglement of formation value
    """
    return _run_async(calculate_entanglement(circuit_qasm))


def partial_trace_state_sync(circuit_qasm: str, trace_qubits: str) -> Dict[str, Any]:
    """Compute partial trace.

    Synchronous version of partial_trace_state.

    Args:
        circuit_qasm: QASM representation of the circuit
        trace_qubits: Comma-separated qubit indices to trace out

    Returns:
        Reduced density matrix information
    """
    return _run_async(partial_trace_state(circuit_qasm, trace_qubits))


def expectation_value_sync(
    circuit_qasm: str, pauli_strings: str, coeffs: str = ""
) -> Dict[str, Any]:
    """Calculate expectation value of an observable.

    Synchronous version of expectation_value.

    Args:
        circuit_qasm: QASM representation of the circuit
        pauli_strings: Comma-separated Pauli strings
        coeffs: Optional comma-separated coefficients

    Returns:
        Expectation value
    """
    return _run_async(expectation_value(circuit_qasm, pauli_strings, coeffs))


def random_quantum_state_sync(
    num_qubits: int, seed: Optional[int] = None
) -> Dict[str, Any]:
    """Generate a random quantum statevector.

    Synchronous version of random_quantum_state.

    Args:
        num_qubits: Number of qubits
        seed: Random seed for reproducibility

    Returns:
        Random statevector information
    """
    return _run_async(random_quantum_state(num_qubits, seed))


def random_density_matrix_state_sync(
    num_qubits: int, rank: Optional[int] = None, seed: Optional[int] = None
) -> Dict[str, Any]:
    """Generate a random density matrix.

    Synchronous version of random_density_matrix_state.

    Args:
        num_qubits: Number of qubits
        rank: Rank of the density matrix
        seed: Random seed for reproducibility

    Returns:
        Random density matrix information
    """
    return _run_async(random_density_matrix_state(num_qubits, rank, seed))


# Primitives synchronous wrappers


def sample_circuit_sync(
    circuit_qasm: str, shots: int = 1024, seed: Optional[int] = None
) -> Dict[str, Any]:
    """Sample from a quantum circuit using the Sampler primitive.

    Synchronous version of sample_circuit.

    Args:
        circuit_qasm: QASM representation of the circuit
        shots: Number of measurement shots
        seed: Random seed for reproducibility

    Returns:
        Sampling results with counts
    """
    return _run_async(sample_circuit(circuit_qasm, shots, seed))


def estimate_expectation_values_sync(
    circuit_qasm: str, observables: str, coeffs: str = "", precision: float = 0.0
) -> Dict[str, Any]:
    """Estimate expectation values using the Estimator primitive.

    Synchronous version of estimate_expectation_values.

    Args:
        circuit_qasm: QASM representation of the circuit
        observables: Comma-separated Pauli strings
        coeffs: Optional comma-separated coefficients
        precision: Target precision

    Returns:
        Expectation value results
    """
    return _run_async(
        estimate_expectation_values(circuit_qasm, observables, coeffs, precision)
    )


def run_variational_estimation_sync(
    circuit_qasms: list, observables: str, coeffs: str = ""
) -> Dict[str, Any]:
    """Run Estimator on multiple circuit variations.

    Synchronous version of run_variational_estimation.

    Args:
        circuit_qasms: List of QASM strings
        observables: Comma-separated Pauli strings
        coeffs: Optional comma-separated coefficients

    Returns:
        Expectation values for all variations
    """
    return _run_async(run_variational_estimation(circuit_qasms, observables, coeffs))


# Advanced Circuits synchronous wrappers


def compose_circuits_sync(
    circuit1_qasm: str, circuit2_qasm: str, inplace: bool = False
) -> Dict[str, Any]:
    """Compose two quantum circuits.

    Synchronous version of compose_circuits.

    Args:
        circuit1_qasm: QASM representation of first circuit
        circuit2_qasm: QASM representation of second circuit
        inplace: If True, modify first circuit

    Returns:
        Composed circuit information
    """
    return _run_async(compose_circuits(circuit1_qasm, circuit2_qasm, inplace))


def tensor_circuits_sync(circuit1_qasm: str, circuit2_qasm: str) -> Dict[str, Any]:
    """Tensor product of two quantum circuits.

    Synchronous version of tensor_circuits.

    Args:
        circuit1_qasm: QASM representation of first circuit
        circuit2_qasm: QASM representation of second circuit

    Returns:
        Tensored circuit information
    """
    return _run_async(tensor_circuits(circuit1_qasm, circuit2_qasm))


def create_parametric_circuit_sync(
    num_qubits: int, parameter_names: str, num_classical_bits: int = 0
) -> Dict[str, Any]:
    """Create a circuit with named parameters.

    Synchronous version of create_parametric_circuit.

    Args:
        num_qubits: Number of qubits
        parameter_names: Comma-separated parameter names
        num_classical_bits: Number of classical bits

    Returns:
        Circuit with parameters information
    """
    return _run_async(
        create_parametric_circuit(num_qubits, parameter_names, num_classical_bits)
    )


def bind_parameters_sync(circuit_qasm: str, parameter_values: str) -> Dict[str, Any]:
    """Bind parameter values to a parametric circuit.

    Synchronous version of bind_parameters.

    Args:
        circuit_qasm: QASM representation of parametric circuit
        parameter_values: JSON dict of parameter names to values

    Returns:
        Bound circuit information
    """
    return _run_async(bind_parameters(circuit_qasm, parameter_values))


def decompose_circuit_sync(circuit_qasm: str, reps: int = 1) -> Dict[str, Any]:
    """Decompose circuit gates into basis gates.

    Synchronous version of decompose_circuit.

    Args:
        circuit_qasm: QASM representation of circuit
        reps: Number of decomposition repetitions

    Returns:
        Decomposed circuit information
    """
    return _run_async(decompose_circuit(circuit_qasm, reps))


def add_controlled_gate_sync(
    circuit_qasm: str, gate_name: str, control_qubits: str, target_qubits: str
) -> Dict[str, Any]:
    """Add a controlled version of a gate to the circuit.

    Synchronous version of add_controlled_gate.

    Args:
        circuit_qasm: QASM representation of circuit
        gate_name: Name of gate to control
        control_qubits: Comma-separated control qubit indices
        target_qubits: Comma-separated target qubit indices

    Returns:
        Updated circuit information
    """
    return _run_async(
        add_controlled_gate(circuit_qasm, gate_name, control_qubits, target_qubits)
    )


def add_power_gate_sync(
    circuit_qasm: str, gate_name: str, power: float, qubit: int
) -> Dict[str, Any]:
    """Add a gate raised to a power.

    Synchronous version of add_power_gate.

    Args:
        circuit_qasm: QASM representation of circuit
        gate_name: Name of gate
        power: Power to raise gate to
        qubit: Target qubit index

    Returns:
        Updated circuit information
    """
    return _run_async(add_power_gate(circuit_qasm, gate_name, power, qubit))


def create_qft_circuit_sync(
    num_qubits: int, inverse: bool = False, do_swaps: bool = True
) -> Dict[str, Any]:
    """Create a Quantum Fourier Transform circuit.

    Synchronous version of create_qft_circuit.

    Args:
        num_qubits: Number of qubits
        inverse: If True, create inverse QFT
        do_swaps: If True, include swap gates

    Returns:
        QFT circuit information
    """
    return _run_async(create_qft_circuit(num_qubits, inverse, do_swaps))


def create_grover_operator_sync(num_qubits: int, oracle_qasm: str) -> Dict[str, Any]:
    """Create a Grover operator circuit.

    Synchronous version of create_grover_operator.

    Args:
        num_qubits: Number of qubits
        oracle_qasm: QASM representation of oracle circuit

    Returns:
        Grover operator circuit information
    """
    return _run_async(create_grover_operator(num_qubits, oracle_qasm))


def create_efficient_su2_sync(
    num_qubits: int, reps: int = 3, entanglement: str = "full"
) -> Dict[str, Any]:
    """Create an EfficientSU2 variational circuit.

    Synchronous version of create_efficient_su2.

    Args:
        num_qubits: Number of qubits
        reps: Number of repetitions
        entanglement: Entanglement strategy

    Returns:
        EfficientSU2 circuit information
    """
    return _run_async(create_efficient_su2(num_qubits, reps, entanglement))


def create_real_amplitudes_sync(
    num_qubits: int, reps: int = 3, entanglement: str = "full"
) -> Dict[str, Any]:
    """Create a RealAmplitudes variational circuit.

    Synchronous version of create_real_amplitudes.

    Args:
        num_qubits: Number of qubits
        reps: Number of repetitions
        entanglement: Entanglement strategy

    Returns:
        RealAmplitudes circuit information
    """
    return _run_async(create_real_amplitudes(num_qubits, reps, entanglement))


def create_pauli_evolution_circuit_sync(
    pauli_string: str, time: float, num_qubits: Optional[int] = None
) -> Dict[str, Any]:
    """Create a circuit for Pauli operator time evolution.

    Synchronous version of create_pauli_evolution_circuit.

    Args:
        pauli_string: Pauli string (e.g., 'XYZI', 'XY', 'ZZ')
        time: Evolution time parameter
        num_qubits: Number of qubits (optional)

    Returns:
        Pauli evolution circuit information
    """
    return _run_async(create_pauli_evolution_circuit(pauli_string, time, num_qubits))


def create_phase_oracle_circuit_sync(
    expression: str, num_qubits: int
) -> Dict[str, Any]:
    """Create a phase oracle circuit from a boolean expression.

    Synchronous version of create_phase_oracle_circuit.

    Args:
        expression: Boolean expression (e.g., '(a & b) | (~c)', 'a ^ b')
        num_qubits: Number of qubits

    Returns:
        Phase oracle circuit information
    """
    return _run_async(create_phase_oracle_circuit(expression, num_qubits))


def create_general_two_local_circuit_sync(
    num_qubits: int,
    rotation_blocks: str = "ry",
    entanglement_blocks: str = "cx",
    entanglement: str = "full",
    reps: int = 3,
    insert_barriers: bool = False,
) -> Dict[str, Any]:
    """Create a general TwoLocal variational circuit.

    Synchronous version of create_general_two_local_circuit.

    Args:
        num_qubits: Number of qubits
        rotation_blocks: Single-qubit rotation gates
        entanglement_blocks: Two-qubit entanglement gates
        entanglement: Entanglement pattern
        reps: Number of repetitions
        insert_barriers: Insert barriers between layers

    Returns:
        General TwoLocal circuit information
    """
    return _run_async(
        create_general_two_local_circuit(
            num_qubits,
            rotation_blocks,
            entanglement_blocks,
            entanglement,
            reps,
            insert_barriers,
        )
    )


def create_parametric_circuit_with_vector_sync(
    num_qubits: int, num_parameters: int, structure: str = "ry_cx"
) -> Dict[str, Any]:
    """Create a parametric circuit using ParameterVector.

    Synchronous version of create_parametric_circuit_with_vector.

    Args:
        num_qubits: Number of qubits
        num_parameters: Number of parameters to create
        structure: Circuit structure

    Returns:
        Parametric circuit with ParameterVector information
    """
    return _run_async(
        create_parametric_circuit_with_vector(num_qubits, num_parameters, structure)
    )


def simulate_with_aer_sync(
    circuit_qasm: str, shots: int = 1024, backend_name: str = "aer_simulator"
) -> Dict[str, Any]:
    """Simulate circuit using Aer simulator.

    Synchronous version of simulate_with_aer.

    Args:
        circuit_qasm: QASM representation of circuit
        shots: Number of shots
        backend_name: Aer backend name

    Returns:
        Simulation results
    """
    return _run_async(simulate_with_aer(circuit_qasm, shots, backend_name))


def get_unitary_matrix_sync(circuit_qasm: str) -> Dict[str, Any]:
    """Get the unitary matrix of a circuit.

    Synchronous version of get_unitary_matrix.

    Args:
        circuit_qasm: QASM representation of circuit

    Returns:
        Unitary matrix information
    """
    return _run_async(get_unitary_matrix(circuit_qasm))


def analyze_circuit_sync(circuit_qasm: str) -> Dict[str, Any]:
    """Perform comprehensive circuit analysis.

    Synchronous version of analyze_circuit.

    Args:
        circuit_qasm: QASM representation of circuit

    Returns:
        Circuit analysis information
    """
    return _run_async(analyze_circuit(circuit_qasm))


def get_circuit_instructions_sync(circuit_qasm: str) -> Dict[str, Any]:
    """Get detailed list of circuit instructions.

    Synchronous version of get_circuit_instructions.

    Args:
        circuit_qasm: QASM representation of circuit

    Returns:
        List of circuit instructions
    """
    return _run_async(get_circuit_instructions(circuit_qasm))


def convert_to_qasm3_sync(circuit_qasm: str) -> Dict[str, Any]:
    """Convert circuit to OpenQASM 3.0 format.

    Synchronous version of convert_to_qasm3.

    Args:
        circuit_qasm: QASM 2.0 representation of circuit

    Returns:
        QASM 3.0 representation
    """
    return _run_async(convert_to_qasm3(circuit_qasm))


def load_qasm3_circuit_sync(qasm3_str: str) -> Dict[str, Any]:
    """Load circuit from OpenQASM 3.0 format.

    Synchronous version of load_qasm3_circuit.

    Args:
        qasm3_str: QASM 3.0 representation

    Returns:
        Circuit information
    """
    return _run_async(load_qasm3_circuit(qasm3_str))


def draw_circuit_text_sync(circuit_qasm: str, fold: int = -1) -> Dict[str, Any]:
    """Draw circuit as text/ASCII art.

    Synchronous version of draw_circuit_text.

    Args:
        circuit_qasm: QASM representation of circuit
        fold: Column to fold the circuit at

    Returns:
        Text visualization of circuit
    """
    return _run_async(draw_circuit_text(circuit_qasm, fold))


def draw_circuit_matplotlib_sync(
    circuit_qasm: str, style: str = "default"
) -> Dict[str, Any]:
    """Draw circuit using matplotlib.

    Synchronous version of draw_circuit_matplotlib.

    Args:
        circuit_qasm: QASM representation of circuit
        style: Drawing style

    Returns:
        Base64 encoded PNG image
    """
    return _run_async(draw_circuit_matplotlib(circuit_qasm, style))


# Noise and Mitigation synchronous wrappers


def create_depolarizing_noise_model_sync(
    single_qubit_error: float = 0.001, two_qubit_error: float = 0.01
) -> Dict[str, Any]:
    """Create a depolarizing noise model.

    Synchronous version of create_depolarizing_noise_model.

    Args:
        single_qubit_error: Error probability for single-qubit gates
        two_qubit_error: Error probability for two-qubit gates

    Returns:
        Noise model information
    """
    return _run_async(
        create_depolarizing_noise_model(single_qubit_error, two_qubit_error)
    )


def create_thermal_noise_model_sync(
    t1: float = 50000.0,
    t2: float = 70000.0,
    gate_time_1q: float = 50.0,
    gate_time_2q: float = 300.0,
) -> Dict[str, Any]:
    """Create a thermal relaxation noise model.

    Synchronous version of create_thermal_noise_model.

    Args:
        t1: T1 relaxation time in nanoseconds
        t2: T2 dephasing time in nanoseconds
        gate_time_1q: Single-qubit gate time in nanoseconds
        gate_time_2q: Two-qubit gate time in nanoseconds

    Returns:
        Noise model information
    """
    return _run_async(create_thermal_noise_model(t1, t2, gate_time_1q, gate_time_2q))


def simulate_with_noise_sync(
    circuit_qasm: str,
    noise_type: str = "depolarizing",
    single_qubit_error: float = 0.001,
    two_qubit_error: float = 0.01,
    shots: int = 1024,
) -> Dict[str, Any]:
    """Simulate circuit with noise model.

    Synchronous version of simulate_with_noise.

    Args:
        circuit_qasm: QASM representation of circuit
        noise_type: Type of noise
        single_qubit_error: Single-qubit error rate
        two_qubit_error: Two-qubit error rate
        shots: Number of shots

    Returns:
        Noisy simulation results
    """
    return _run_async(
        simulate_with_noise(
            circuit_qasm, noise_type, single_qubit_error, two_qubit_error, shots
        )
    )


def compare_ideal_vs_noisy_sync(
    circuit_qasm: str,
    noise_type: str = "depolarizing",
    error_rate: float = 0.01,
    shots: int = 1024,
) -> Dict[str, Any]:
    """Compare ideal and noisy simulation results.

    Synchronous version of compare_ideal_vs_noisy.

    Args:
        circuit_qasm: QASM representation of circuit
        noise_type: Type of noise model
        error_rate: Error rate for noise
        shots: Number of shots

    Returns:
        Comparison of ideal vs noisy results
    """
    return _run_async(
        compare_ideal_vs_noisy(circuit_qasm, noise_type, error_rate, shots)
    )


# State Preparation and Tomography synchronous wrappers


def prepare_uniform_superposition_sync(num_qubits: int) -> Dict[str, Any]:
    """Prepare a uniform superposition state.

    Synchronous version of prepare_uniform_superposition.

    Args:
        num_qubits: Number of qubits

    Returns:
        Circuit creating uniform superposition
    """
    return _run_async(prepare_uniform_superposition(num_qubits))


def prepare_w_state_sync(num_qubits: int) -> Dict[str, Any]:
    """Prepare a W state.

    Synchronous version of prepare_w_state.

    Args:
        num_qubits: Number of qubits

    Returns:
        Circuit creating W state
    """
    return _run_async(prepare_w_state(num_qubits))


def prepare_ghz_state_sync(num_qubits: int) -> Dict[str, Any]:
    """Prepare a GHZ state.

    Synchronous version of prepare_ghz_state.

    Args:
        num_qubits: Number of qubits

    Returns:
        Circuit creating GHZ state
    """
    return _run_async(prepare_ghz_state(num_qubits))


def prepare_dicke_state_sync(num_qubits: int, num_excitations: int) -> Dict[str, Any]:
    """Prepare a Dicke state.

    Synchronous version of prepare_dicke_state.

    Args:
        num_qubits: Number of qubits
        num_excitations: Number of excitations

    Returns:
        Circuit creating Dicke state
    """
    return _run_async(prepare_dicke_state(num_qubits, num_excitations))


def prepare_product_state_sync(state_string: str) -> Dict[str, Any]:
    """Prepare a product state.

    Synchronous version of prepare_product_state.

    Args:
        state_string: String specifying the state

    Returns:
        Circuit creating product state
    """
    return _run_async(prepare_product_state(state_string))


def generate_tomography_circuits_sync(
    circuit_qasm: str, measurement_basis: str = "pauli"
) -> Dict[str, Any]:
    """Generate measurement circuits for state tomography.

    Synchronous version of generate_tomography_circuits.

    Args:
        circuit_qasm: QASM representation of state preparation circuit
        measurement_basis: Basis for measurements

    Returns:
        List of measurement circuits
    """
    return _run_async(generate_tomography_circuits(circuit_qasm, measurement_basis))


def verify_state_preparation_sync(
    prepared_qasm: str, target_qasm: str
) -> Dict[str, Any]:
    """Verify state preparation.

    Synchronous version of verify_state_preparation.

    Args:
        prepared_qasm: QASM of prepared circuit
        target_qasm: QASM of target circuit

    Returns:
        Fidelity between states
    """
    return _run_async(verify_state_preparation(prepared_qasm, target_qasm))


# Circuit Equivalence synchronous wrappers


def check_circuit_equivalence_sync(
    circuit1_qasm: str, circuit2_qasm: str, tolerance: float = 1e-7
) -> Dict[str, Any]:
    """Check if two circuits are equivalent.

    Synchronous version of check_circuit_equivalence.

    Args:
        circuit1_qasm: QASM representation of first circuit
        circuit2_qasm: QASM representation of second circuit
        tolerance: Numerical tolerance

    Returns:
        Equivalence information
    """
    return _run_async(
        check_circuit_equivalence(circuit1_qasm, circuit2_qasm, tolerance)
    )


def check_unitary_equivalence_sync(
    circuit1_qasm: str, circuit2_qasm: str
) -> Dict[str, Any]:
    """Check if two circuits implement the same unitary.

    Synchronous version of check_unitary_equivalence.

    Args:
        circuit1_qasm: QASM representation of first circuit
        circuit2_qasm: QASM representation of second circuit

    Returns:
        Detailed equivalence information
    """
    return _run_async(check_unitary_equivalence(circuit1_qasm, circuit2_qasm))


def compare_circuit_resources_sync(
    circuit1_qasm: str, circuit2_qasm: str
) -> Dict[str, Any]:
    """Compare resource usage of two circuits.

    Synchronous version of compare_circuit_resources.

    Args:
        circuit1_qasm: QASM representation of first circuit
        circuit2_qasm: QASM representation of second circuit

    Returns:
        Resource comparison
    """
    return _run_async(compare_circuit_resources(circuit1_qasm, circuit2_qasm))


def find_circuit_optimizations_sync(circuit_qasm: str) -> Dict[str, Any]:
    """Suggest optimizations for a circuit.

    Synchronous version of find_circuit_optimizations.

    Args:
        circuit_qasm: QASM representation of circuit

    Returns:
        Optimization suggestions
    """
    return _run_async(find_circuit_optimizations(circuit_qasm))


def verify_optimization_sync(original_qasm: str, optimized_qasm: str) -> Dict[str, Any]:
    """Verify that an optimized circuit is equivalent.

    Synchronous version of verify_optimization.

    Args:
        original_qasm: QASM representation of original circuit
        optimized_qasm: QASM representation of optimized circuit

    Returns:
        Verification results
    """
    return _run_async(verify_optimization(original_qasm, optimized_qasm))


# ============================================================================
# Circuit Utilities - Synchronous Wrappers
# ============================================================================


def circuit_inverse_sync(circuit_qasm: str) -> Dict[str, Any]:
    """Get the inverse of a quantum circuit.

    Synchronous version of circuit_inverse.

    Args:
        circuit_qasm: QASM representation of circuit

    Returns:
        Inverse circuit information
    """
    return _run_async(circuit_inverse(circuit_qasm))


def circuit_copy_sync(circuit_qasm: str, name: str = None) -> Dict[str, Any]:
    """Create a deep copy of a quantum circuit.

    Synchronous version of circuit_copy.

    Args:
        circuit_qasm: QASM representation of circuit
        name: Optional name for the copied circuit

    Returns:
        Copied circuit information
    """
    return _run_async(circuit_copy(circuit_qasm, name))


def circuit_reverse_bits_sync(circuit_qasm: str) -> Dict[str, Any]:
    """Reverse the order of bits in a circuit.

    Synchronous version of circuit_reverse_bits.

    Args:
        circuit_qasm: QASM representation of circuit

    Returns:
        Circuit with reversed bit order
    """
    return _run_async(circuit_reverse_bits(circuit_qasm))


def circuit_to_gate_sync(circuit_qasm: str, label: str = None) -> Dict[str, Any]:
    """Convert a quantum circuit to a Gate object.

    Synchronous version of circuit_to_gate.

    Args:
        circuit_qasm: QASM representation of circuit
        label: Optional label for the gate

    Returns:
        Information about the created gate
    """
    return _run_async(circuit_to_gate(circuit_qasm, label))


def circuit_to_instruction_sync(circuit_qasm: str, label: str = None) -> Dict[str, Any]:
    """Convert a quantum circuit to an Instruction object.

    Synchronous version of circuit_to_instruction.

    Args:
        circuit_qasm: QASM representation of circuit
        label: Optional label for the instruction

    Returns:
        Information about the created instruction
    """
    return _run_async(circuit_to_instruction(circuit_qasm, label))


def load_qasm2_file_sync(file_path: str) -> Dict[str, Any]:
    """Load a quantum circuit from a QASM 2.0 file.

    Synchronous version of load_qasm2_file.

    Args:
        file_path: Path to the QASM file

    Returns:
        Loaded circuit information
    """
    return _run_async(load_qasm2_file(file_path))


def save_qasm2_file_sync(circuit_qasm: str, file_path: str) -> Dict[str, Any]:
    """Save a quantum circuit to a QASM 2.0 file.

    Synchronous version of save_qasm2_file.

    Args:
        circuit_qasm: QASM representation of circuit
        file_path: Path where the file should be saved

    Returns:
        Save operation result
    """
    return _run_async(save_qasm2_file(circuit_qasm, file_path))


def load_qasm3_file_sync(file_path: str) -> Dict[str, Any]:
    """Load a quantum circuit from a QASM 3.0 file.

    Synchronous version of load_qasm3_file.

    Args:
        file_path: Path to the QASM 3.0 file

    Returns:
        Loaded circuit information
    """
    return _run_async(load_qasm3_file(file_path))


def save_qasm3_file_sync(circuit_qasm: str, file_path: str) -> Dict[str, Any]:
    """Save a quantum circuit to a QASM 3.0 file.

    Synchronous version of save_qasm3_file.

    Args:
        circuit_qasm: QASM representation of circuit
        file_path: Path where the file should be saved

    Returns:
        Save operation result
    """
    return _run_async(save_qasm3_file(circuit_qasm, file_path))


def convert_circuit_to_dag_sync(circuit_qasm: str) -> Dict[str, Any]:
    """Convert a quantum circuit to a DAG representation.

    Synchronous version of convert_circuit_to_dag.

    Args:
        circuit_qasm: QASM representation of circuit

    Returns:
        DAG information
    """
    return _run_async(convert_circuit_to_dag(circuit_qasm))


def convert_dag_to_circuit_sync(circuit_qasm: str) -> Dict[str, Any]:
    """Convert a DAG back to a quantum circuit.

    Synchronous version of convert_dag_to_circuit_wrapper.

    Args:
        circuit_qasm: QASM representation of original circuit

    Returns:
        Reconstructed circuit information
    """
    return _run_async(convert_dag_to_circuit_wrapper(circuit_qasm))


def decompose_circuit_utility_sync(
    circuit_qasm: str, gates_to_decompose: str = None, reps: int = 1
) -> Dict[str, Any]:
    """Decompose a circuit by expanding composite gates.

    Synchronous version of decompose_circuit_utility.

    Args:
        circuit_qasm: QASM representation of circuit
        gates_to_decompose: Comma-separated list of gate names to decompose
        reps: Number of decomposition repetitions

    Returns:
        Decomposed circuit information
    """
    return _run_async(decompose_circuit_utility(circuit_qasm, gates_to_decompose, reps))


# ============================================================================
# Enhanced Transpilation - Synchronous Wrappers
# ============================================================================


def transpile_with_backend_sync(
    circuit_qasm: str,
    backend_name: str,
    optimization_level: int = 1,
    seed_transpiler: Optional[int] = None,
    layout_method: Optional[str] = None,
    routing_method: Optional[str] = None,
) -> Dict[str, Any]:
    """Transpile a circuit for a specific backend.

    Synchronous version of transpile_with_backend.

    Args:
        circuit_qasm: QASM representation of the circuit
        backend_name: Name of the target backend
        optimization_level: Optimization level (0-3)
        seed_transpiler: Random seed for reproducibility
        layout_method: Layout selection method
        routing_method: Routing method

    Returns:
        Transpiled circuit information
    """
    return _run_async(
        transpile_with_backend(
            circuit_qasm,
            backend_name,
            optimization_level,
            seed_transpiler,
            layout_method,
            routing_method,
        )
    )


def transpile_with_coupling_map_sync(
    circuit_qasm: str,
    coupling_map_json: str,
    optimization_level: int = 1,
    initial_layout_json: Optional[str] = None,
    layout_method: Optional[str] = None,
    routing_method: Optional[str] = None,
    seed_transpiler: Optional[int] = None,
) -> Dict[str, Any]:
    """Transpile a circuit with a custom coupling map.

    Synchronous version of transpile_with_coupling_map.

    Args:
        circuit_qasm: QASM representation of the circuit
        coupling_map_json: JSON array of edges or shorthand notation
        optimization_level: Optimization level (0-3)
        initial_layout_json: JSON dict mapping virtual to physical qubits
        layout_method: Layout selection method
        routing_method: Routing method
        seed_transpiler: Random seed for reproducibility

    Returns:
        Transpiled circuit information
    """
    return _run_async(
        transpile_with_coupling_map(
            circuit_qasm,
            coupling_map_json,
            optimization_level,
            initial_layout_json,
            layout_method,
            routing_method,
            seed_transpiler,
        )
    )


def transpile_with_layout_strategy_sync(
    circuit_qasm: str,
    layout_method: str,
    routing_method: str = "sabre",
    optimization_level: int = 2,
    basis_gates: Optional[str] = None,
    coupling_map_json: Optional[str] = None,
    seed_transpiler: Optional[int] = None,
) -> Dict[str, Any]:
    """Transpile a circuit with specific layout and routing strategies.

    Synchronous version of transpile_with_layout_strategy.

    Args:
        circuit_qasm: QASM representation of the circuit
        layout_method: Layout method
        routing_method: Routing method
        optimization_level: Optimization level (0-3)
        basis_gates: Comma-separated basis gates
        coupling_map_json: JSON array of edges or shorthand notation
        seed_transpiler: Random seed for reproducibility

    Returns:
        Transpiled circuit with strategy details
    """
    return _run_async(
        transpile_with_layout_strategy(
            circuit_qasm,
            layout_method,
            routing_method,
            optimization_level,
            basis_gates,
            coupling_map_json,
            seed_transpiler,
        )
    )


def compare_transpilation_strategies_sync(
    circuit_qasm: str,
    optimization_level: int = 2,
    coupling_map_json: Optional[str] = None,
    seed_transpiler: Optional[int] = None,
) -> Dict[str, Any]:
    """Compare different transpilation strategies.

    Synchronous version of compare_transpilation_strategies.

    Args:
        circuit_qasm: QASM representation of the circuit
        optimization_level: Optimization level (0-3)
        coupling_map_json: JSON array of edges or shorthand notation
        seed_transpiler: Random seed for reproducibility

    Returns:
        Comparison of transpilation strategies
    """
    return _run_async(
        compare_transpilation_strategies(
            circuit_qasm, optimization_level, coupling_map_json, seed_transpiler
        )
    )


def transpile_for_basis_gates_sync(
    circuit_qasm: str,
    basis_gates: str,
    optimization_level: int = 1,
    seed_transpiler: Optional[int] = None,
) -> Dict[str, Any]:
    """Transpile a circuit to a specific basis gate set.

    Synchronous version of transpile_for_basis_gates.

    Args:
        circuit_qasm: QASM representation of the circuit
        basis_gates: Comma-separated basis gates
        optimization_level: Optimization level (0-3)
        seed_transpiler: Random seed for reproducibility

    Returns:
        Transpiled circuit information
    """
    return _run_async(
        transpile_for_basis_gates(
            circuit_qasm, basis_gates, optimization_level, seed_transpiler
        )
    )


def get_available_backends_sync() -> Dict[str, Any]:
    """Get list of available IBM Quantum backends.

    Synchronous version of get_available_backends.

    Returns:
        List of available backends with their properties
    """
    return _run_async(get_available_backends())


# ============================================================================
# PassManager - Synchronous Wrappers
# ============================================================================


def run_preset_pass_manager_sync(
    circuit_qasm: str,
    optimization_level: int = 1,
    backend_name: Optional[str] = None,
    coupling_map_json: Optional[str] = None,
    basis_gates: Optional[str] = None,
    seed_transpiler: Optional[int] = None,
) -> Dict[str, Any]:
    """Run a preset pass manager on a circuit.

    Synchronous version of run_preset_pass_manager.

    Args:
        circuit_qasm: QASM representation of the circuit
        optimization_level: Optimization level (0-3)
        backend_name: Optional backend name
        coupling_map_json: Optional coupling map JSON
        basis_gates: Optional basis gates
        seed_transpiler: Random seed for reproducibility

    Returns:
        Transpiled circuit with pass manager details
    """
    return _run_async(
        run_preset_pass_manager(
            circuit_qasm,
            optimization_level,
            backend_name,
            coupling_map_json,
            basis_gates,
            seed_transpiler,
        )
    )


def run_optimization_passes_sync(
    circuit_qasm: str, iterations: int = 2
) -> Dict[str, Any]:
    """Run optimization passes on a circuit.

    Synchronous version of run_optimization_passes.

    Args:
        circuit_qasm: QASM representation of the circuit
        iterations: Number of optimization iterations

    Returns:
        Optimized circuit information
    """
    return _run_async(run_optimization_passes(circuit_qasm, iterations))


def run_analysis_passes_sync(circuit_qasm: str) -> Dict[str, Any]:
    """Run analysis passes on a circuit.

    Synchronous version of run_analysis_passes.

    Args:
        circuit_qasm: QASM representation of the circuit

    Returns:
        Circuit analysis information
    """
    return _run_async(run_analysis_passes(circuit_qasm))


def run_unroll_passes_sync(circuit_qasm: str, basis_gates: str) -> Dict[str, Any]:
    """Run unrolling passes to decompose gates.

    Synchronous version of run_unroll_passes.

    Args:
        circuit_qasm: QASM representation of the circuit
        basis_gates: Comma-separated basis gates

    Returns:
        Unrolled circuit information
    """
    return _run_async(run_unroll_passes(circuit_qasm, basis_gates))


def run_combined_passes_sync(
    circuit_qasm: str,
    basis_gates: str,
    optimization_level: int = 2,
    coupling_map_json: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a combined pipeline of unrolling and optimization passes.

    Synchronous version of run_combined_passes.

    Args:
        circuit_qasm: QASM representation of the circuit
        basis_gates: Comma-separated basis gates
        optimization_level: Number of optimization iterations
        coupling_map_json: Optional coupling map JSON

    Returns:
        Processed circuit information
    """
    return _run_async(
        run_combined_passes(
            circuit_qasm, basis_gates, optimization_level, coupling_map_json
        )
    )


# Result Processing Sync Wrappers
def marginal_counts_sync(counts_json: str, indices: str) -> Dict[str, Any]:
    """Marginalize counts to specific qubit indices - sync version."""
    return _run_async(marginal_counts(counts_json, indices))


def marginal_distribution_sync(counts_json: str, indices: str) -> Dict[str, Any]:
    """Compute marginal probability distribution - sync version."""
    return _run_async(marginal_distribution(counts_json, indices))


def counts_to_probabilities_sync(counts_json: str) -> Dict[str, Any]:
    """Convert counts to probabilities - sync version."""
    return _run_async(counts_to_probabilities(counts_json))


def filter_counts_sync(counts_json: str, pattern: str) -> Dict[str, Any]:
    """Filter counts by bit pattern - sync version."""
    return _run_async(filter_counts(counts_json, pattern))


def combine_counts_sync(counts_json_list: str) -> Dict[str, Any]:
    """Combine multiple count dictionaries - sync version."""
    return _run_async(combine_counts(counts_json_list))


def expectation_from_counts_sync(counts_json: str, operator: str) -> Dict[str, Any]:
    """Calculate expectation value from counts - sync version."""
    return _run_async(expectation_from_counts(counts_json, operator))


def analyze_measurement_results_sync(counts_json: str) -> Dict[str, Any]:
    """Comprehensive measurement analysis - sync version."""
    return _run_async(analyze_measurement_results(counts_json))


# Visualization Extended Sync Wrappers
def plot_bloch_multivector_sync(circuit_qasm: str) -> Dict[str, Any]:
    """Plot Bloch sphere visualization - sync version."""
    return _run_async(plot_bloch_multivector(circuit_qasm))


def plot_state_qsphere_sync(circuit_qasm: str) -> Dict[str, Any]:
    """Plot Q-sphere visualization - sync version."""
    return _run_async(plot_state_qsphere(circuit_qasm))


def plot_state_hinton_sync(circuit_qasm: str) -> Dict[str, Any]:
    """Plot Hinton diagram - sync version."""
    return _run_async(plot_state_hinton(circuit_qasm))


def plot_state_city_sync(circuit_qasm: str) -> Dict[str, Any]:
    """Plot city/bar plot - sync version."""
    return _run_async(plot_state_city(circuit_qasm))


def plot_state_paulivec_sync(circuit_qasm: str) -> Dict[str, Any]:
    """Plot Pauli vector representation - sync version."""
    return _run_async(plot_state_paulivec(circuit_qasm))


def plot_histogram_sync(counts_json: str) -> Dict[str, Any]:
    """Plot measurement histogram - sync version."""
    return _run_async(plot_histogram(counts_json))


def plot_distribution_sync(counts_json: str) -> Dict[str, Any]:
    """Plot probability distribution - sync version."""
    return _run_async(plot_distribution(counts_json))


# Circuit Library Extended Sync Wrappers
def create_two_local_circuit_sync(
    num_qubits: int,
    rotation_blocks: str = "ry",
    entanglement_blocks: str = "cx",
    reps: int = 3,
    entanglement: str = "full",
) -> Dict[str, Any]:
    """Create TwoLocal circuit - sync version."""
    return _run_async(
        create_two_local_circuit(
            num_qubits, rotation_blocks, entanglement_blocks, reps, entanglement
        )
    )


def create_n_local_circuit_sync(
    num_qubits: int,
    rotation_blocks: str = "ry",
    entanglement_blocks: str = "cx",
    num_qubits_entanglement: int = 2,
    reps: int = 3,
    entanglement: str = "full",
) -> Dict[str, Any]:
    """Create NLocal circuit - sync version."""
    return _run_async(
        create_n_local_circuit(
            num_qubits,
            rotation_blocks,
            entanglement_blocks,
            num_qubits_entanglement,
            reps,
            entanglement,
        )
    )


def create_pauli_feature_map_sync(
    feature_dimension: int,
    reps: int = 2,
    paulis: Optional[str] = None,
) -> Dict[str, Any]:
    """Create Pauli feature map - sync version."""
    return _run_async(create_pauli_feature_map(feature_dimension, reps, paulis))


def create_z_feature_map_sync(
    feature_dimension: int,
    reps: int = 2,
) -> Dict[str, Any]:
    """Create Z feature map - sync version."""
    return _run_async(create_z_feature_map(feature_dimension, reps))


def create_zz_feature_map_sync(
    feature_dimension: int,
    reps: int = 2,
    entanglement: str = "full",
) -> Dict[str, Any]:
    """Create ZZ feature map - sync version."""
    return _run_async(create_zz_feature_map(feature_dimension, reps, entanglement))


def create_qaoa_ansatz_sync(
    cost_operator: str,
    reps: int = 1,
) -> Dict[str, Any]:
    """Create QAOA ansatz - sync version."""
    return _run_async(create_qaoa_ansatz(cost_operator, reps))


def create_and_gate_sync(num_variable_qubits: int) -> Dict[str, Any]:
    """Create AND gate - sync version."""
    return _run_async(create_and_gate(num_variable_qubits))


def create_or_gate_sync(num_variable_qubits: int) -> Dict[str, Any]:
    """Create OR gate - sync version."""
    return _run_async(create_or_gate(num_variable_qubits))


def create_xor_gate_sync(num_qubits: int) -> Dict[str, Any]:
    """Create XOR gate - sync version."""
    return _run_async(create_xor_gate(num_qubits))


def create_hidden_linear_function_sync(num_qubits: int) -> Dict[str, Any]:
    """Create hidden linear function - sync version."""
    return _run_async(create_hidden_linear_function(num_qubits))


def create_iqp_circuit_sync(num_qubits: int) -> Dict[str, Any]:
    """Create IQP circuit - sync version."""
    return _run_async(create_iqp_circuit(num_qubits))


def create_phase_estimation_circuit_sync(
    unitary_qasm: str,
    num_evaluation_qubits: int,
) -> Dict[str, Any]:
    """Create phase estimation circuit - sync version."""
    return _run_async(
        create_phase_estimation_circuit(unitary_qasm, num_evaluation_qubits)
    )


# Backend Execution Sync Wrappers
def list_available_backends_sync(simulator_only: bool = False) -> Dict[str, Any]:
    """List available backends - sync version."""
    return _run_async(list_available_backends(simulator_only))


def get_backend_properties_sync(backend_name: str) -> Dict[str, Any]:
    """Get backend properties - sync version."""
    return _run_async(get_backend_properties(backend_name))


def execute_circuit_local_sync(
    circuit_qasm: str, shots: int = 1024, backend_name: str = "statevector_simulator"
) -> Dict[str, Any]:
    """Execute circuit locally - sync version."""
    return _run_async(execute_circuit_local(circuit_qasm, shots, backend_name))


def submit_job_to_ibm_sync(
    circuit_qasm: str, backend_name: str, shots: int = 1024
) -> Dict[str, Any]:
    """Submit job to IBM - sync version."""
    return _run_async(submit_job_to_ibm(circuit_qasm, backend_name, shots))


def retrieve_job_result_sync(job_id: str) -> Dict[str, Any]:
    """Retrieve job result - sync version."""
    return _run_async(retrieve_job_result(job_id))


def cancel_job_sync(job_id: str) -> Dict[str, Any]:
    """Cancel job - sync version."""
    return _run_async(cancel_job(job_id))


def get_job_status_sync(job_id: str) -> Dict[str, Any]:
    """Get job status - sync version."""
    return _run_async(get_job_status(job_id))


def estimate_circuit_cost_sync(circuit_qasm: str, backend_name: str) -> Dict[str, Any]:
    """Estimate circuit cost - sync version."""
    return _run_async(estimate_circuit_cost(circuit_qasm, backend_name))


# Algorithm Solver Sync Wrappers
def run_vqe_sync(
    hamiltonian: str,
    ansatz_qasm: str,
    initial_point: Optional[str] = None,
    optimizer: str = "COBYLA",
    max_iterations: int = 100,
) -> Dict[str, Any]:
    """Run VQE - sync version."""
    return _run_async(
        run_vqe(hamiltonian, ansatz_qasm, initial_point, optimizer, max_iterations)
    )


def evaluate_expectation_value_sync(
    circuit_qasm: str, observable: str
) -> Dict[str, Any]:
    """Evaluate expectation value - sync version."""
    return _run_async(evaluate_expectation_value(circuit_qasm, observable))


def run_qaoa_sync(
    cost_hamiltonian: str,
    num_qubits: int,
    num_layers: int = 1,
    optimizer: str = "COBYLA",
    max_iterations: int = 100,
    initial_point: Optional[str] = None,
) -> Dict[str, Any]:
    """Run QAOA - sync version."""
    return _run_async(
        run_qaoa(
            cost_hamiltonian,
            num_qubits,
            num_layers,
            optimizer,
            max_iterations,
            initial_point,
        )
    )


def optimize_parameters_sync(
    circuit_qasm: str,
    cost_function_type: str,
    observable: Optional[str] = None,
    optimizer: str = "COBYLA",
    max_iterations: int = 100,
    initial_point: Optional[str] = None,
) -> Dict[str, Any]:
    """Optimize parameters - sync version."""
    return _run_async(
        optimize_parameters(
            circuit_qasm,
            cost_function_type,
            observable,
            optimizer,
            max_iterations,
            initial_point,
        )
    )


# Error Mitigation Sync Wrappers
def create_measurement_calibration_sync(
    num_qubits: int, qubit_list: Optional[str] = None
) -> Dict[str, Any]:
    """Create measurement calibration - sync version."""
    return _run_async(create_measurement_calibration(num_qubits, qubit_list))


def apply_measurement_mitigation_sync(
    measured_counts_json: str, calibration_results_json: str
) -> Dict[str, Any]:
    """Apply measurement mitigation - sync version."""
    return _run_async(
        apply_measurement_mitigation(measured_counts_json, calibration_results_json)
    )


def zero_noise_extrapolation_sync(
    circuit_qasm: str,
    observable: str,
    scale_factors: str = "1.0,1.5,2.0,2.5,3.0",
    extrapolation_method: str = "linear",
) -> Dict[str, Any]:
    """Zero-noise extrapolation - sync version."""
    return _run_async(
        zero_noise_extrapolation(
            circuit_qasm, observable, scale_factors, extrapolation_method
        )
    )


def create_readout_error_model_sync(error_rates_json: str) -> Dict[str, Any]:
    """Create readout error model - sync version."""
    return _run_async(create_readout_error_model(error_rates_json))


def probabilistic_error_cancellation_sync(
    circuit_qasm: str, noise_model_json: str, num_samples: int = 100
) -> Dict[str, Any]:
    """Probabilistic error cancellation - sync version."""
    return _run_async(
        probabilistic_error_cancellation(circuit_qasm, noise_model_json, num_samples)
    )


def apply_dynamical_decoupling_sync(
    circuit_qasm: str, dd_sequence: str = "XY4"
) -> Dict[str, Any]:
    """Apply dynamical decoupling - sync version."""
    return _run_async(apply_dynamical_decoupling(circuit_qasm, dd_sequence))


# Assisted by watsonx Code Assistant
