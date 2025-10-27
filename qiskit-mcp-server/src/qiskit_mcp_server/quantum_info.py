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

"""Quantum information functions for the MCP server."""

import logging
from typing import Any, Dict, Optional

from qiskit import qasm2
from qiskit.quantum_info import (
    SparsePauliOp,
    Operator,
    DensityMatrix,
    Statevector,
    state_fidelity,
    average_gate_fidelity,
    process_fidelity,
    entropy,
    entanglement_of_formation,
    partial_trace,
    random_statevector,
    random_density_matrix,
)

logger = logging.getLogger(__name__)


async def create_pauli_operator(pauli_strings: str, coeffs: str = "") -> Dict[str, Any]:
    """Create a SparsePauliOp from Pauli strings and coefficients.

    Args:
        pauli_strings: Comma-separated Pauli strings (e.g., "XX,YZ,II")
        coeffs: Optional comma-separated coefficients (e.g., "1.0,0.5,0.25")
                If not provided, all coefficients are 1.0

    Returns:
        SparsePauliOp information
    """
    try:
        # Parse Pauli strings
        paulis = [p.strip() for p in pauli_strings.split(",")]

        # Parse coefficients if provided
        if coeffs:
            coefficients = [float(c.strip()) for c in coeffs.split(",")]
            if len(coefficients) != len(paulis):
                return {
                    "status": "error",
                    "message": f"Number of coefficients ({len(coefficients)}) must match number of Pauli strings ({len(paulis)})",
                }
        else:
            coefficients = [1.0] * len(paulis)

        # Create SparsePauliOp
        op = SparsePauliOp(paulis, coeffs=coefficients)

        return {
            "status": "success",
            "operator": {
                "paulis": [str(p) for p in op.paulis],
                "coeffs": op.coeffs.tolist(),
                "num_qubits": op.num_qubits,
                "size": op.size,
            },
            "representation": str(op),
        }
    except Exception as e:
        logger.error(f"Failed to create Pauli operator: {e}")
        return {
            "status": "error",
            "message": f"Failed to create Pauli operator: {str(e)}",
        }


async def create_operator_from_circuit(circuit_qasm: str) -> Dict[str, Any]:
    """Create an Operator from a quantum circuit.

    Args:
        circuit_qasm: QASM representation of the circuit

    Returns:
        Operator information
    """
    try:
        circuit = qasm2.loads(circuit_qasm)

        # Create operator from circuit
        op = Operator(circuit)

        return {
            "status": "success",
            "operator": {
                "num_qubits": op.num_qubits,
                "dim": op.dim,
                "is_unitary": op.is_unitary(),
            },
            "matrix": op.data.tolist(),
        }
    except Exception as e:
        logger.error(f"Failed to create operator: {e}")
        return {"status": "error", "message": f"Failed to create operator: {str(e)}"}


async def create_density_matrix(circuit_qasm: str) -> Dict[str, Any]:
    """Create a DensityMatrix from a quantum circuit.

    Args:
        circuit_qasm: QASM representation of the circuit

    Returns:
        DensityMatrix information
    """
    try:
        circuit = qasm2.loads(circuit_qasm)

        # Remove measurements for density matrix
        circuit_copy = circuit.copy()
        circuit_copy.remove_final_measurements(inplace=True)

        # Create density matrix
        rho = DensityMatrix.from_instruction(circuit_copy)

        return {
            "status": "success",
            "density_matrix": {
                "num_qubits": rho.num_qubits,
                "dim": rho.dim,
                "is_valid": rho.is_valid(),
                "purity": rho.purity(),
            },
            "matrix": rho.data.tolist(),
            "probabilities": rho.probabilities().tolist(),
        }
    except Exception as e:
        logger.error(f"Failed to create density matrix: {e}")
        return {
            "status": "error",
            "message": f"Failed to create density matrix: {str(e)}",
        }


async def calculate_state_fidelity(
    circuit_qasm1: str, circuit_qasm2: str
) -> Dict[str, Any]:
    """Calculate fidelity between two quantum states.

    Args:
        circuit_qasm1: QASM representation of first circuit
        circuit_qasm2: QASM representation of second circuit

    Returns:
        Fidelity value
    """
    try:
        circuit1 = qasm2.loads(circuit_qasm1)
        circuit2 = qasm2.loads(circuit_qasm2)

        # Remove measurements
        circuit1.remove_final_measurements(inplace=True)
        circuit2.remove_final_measurements(inplace=True)

        # Create statevectors
        state1 = Statevector.from_instruction(circuit1)
        state2 = Statevector.from_instruction(circuit2)

        # Calculate fidelity
        fidelity = state_fidelity(state1, state2)

        return {
            "status": "success",
            "fidelity": float(fidelity),
            "message": f"State fidelity: {fidelity:.6f}",
        }
    except Exception as e:
        logger.error(f"Failed to calculate fidelity: {e}")
        return {"status": "error", "message": f"Failed to calculate fidelity: {str(e)}"}


async def calculate_gate_fidelity(
    circuit_qasm: str, target_qasm: str
) -> Dict[str, Any]:
    """Calculate average gate fidelity between two operations.

    Args:
        circuit_qasm: QASM representation of actual circuit
        target_qasm: QASM representation of target circuit

    Returns:
        Gate fidelity information
    """
    try:
        circuit = qasm2.loads(circuit_qasm)
        target = qasm2.loads(target_qasm)

        # Create operators
        op1 = Operator(circuit)
        op2 = Operator(target)

        # Calculate fidelities
        avg_fidelity = average_gate_fidelity(op1, op2)
        proc_fidelity = process_fidelity(op1, op2)

        return {
            "status": "success",
            "average_gate_fidelity": float(avg_fidelity),
            "process_fidelity": float(proc_fidelity),
        }
    except Exception as e:
        logger.error(f"Failed to calculate gate fidelity: {e}")
        return {
            "status": "error",
            "message": f"Failed to calculate gate fidelity: {str(e)}",
        }


async def calculate_entropy(
    circuit_qasm: str, subsystem_qubits: str = ""
) -> Dict[str, Any]:
    """Calculate von Neumann entropy of a quantum state.

    Args:
        circuit_qasm: QASM representation of the circuit
        subsystem_qubits: Optional comma-separated qubit indices for partial trace
                         (e.g., "0,1" to trace out qubits 0 and 1)

    Returns:
        Entropy value
    """
    try:
        circuit = qasm2.loads(circuit_qasm)
        circuit.remove_final_measurements(inplace=True)

        # Create density matrix
        rho = DensityMatrix.from_instruction(circuit)

        # Perform partial trace if subsystem specified
        if subsystem_qubits:
            qubits = [int(q.strip()) for q in subsystem_qubits.split(",")]
            rho = partial_trace(rho, qubits)

        # Calculate entropy
        ent = entropy(rho)

        return {
            "status": "success",
            "entropy": float(ent),
            "num_qubits": rho.num_qubits,
            "message": f"von Neumann entropy: {ent:.6f}",
        }
    except Exception as e:
        logger.error(f"Failed to calculate entropy: {e}")
        return {"status": "error", "message": f"Failed to calculate entropy: {str(e)}"}


async def calculate_entanglement(circuit_qasm: str) -> Dict[str, Any]:
    """Calculate entanglement of formation for a two-qubit state.

    Args:
        circuit_qasm: QASM representation of a 2-qubit circuit

    Returns:
        Entanglement of formation value
    """
    try:
        circuit = qasm2.loads(circuit_qasm)
        circuit.remove_final_measurements(inplace=True)

        if circuit.num_qubits != 2:
            return {
                "status": "error",
                "message": f"Entanglement of formation requires exactly 2 qubits, got {circuit.num_qubits}",
            }

        # Create density matrix
        rho = DensityMatrix.from_instruction(circuit)

        # Calculate entanglement
        eof = entanglement_of_formation(rho)

        return {
            "status": "success",
            "entanglement_of_formation": float(eof),
            "message": f"Entanglement of formation: {eof:.6f}",
        }
    except Exception as e:
        logger.error(f"Failed to calculate entanglement: {e}")
        return {
            "status": "error",
            "message": f"Failed to calculate entanglement: {str(e)}",
        }


async def partial_trace_state(circuit_qasm: str, trace_qubits: str) -> Dict[str, Any]:
    """Compute partial trace over specified qubits.

    Args:
        circuit_qasm: QASM representation of the circuit
        trace_qubits: Comma-separated qubit indices to trace out (e.g., "0,2")

    Returns:
        Reduced density matrix information
    """
    try:
        circuit = qasm2.loads(circuit_qasm)
        circuit.remove_final_measurements(inplace=True)

        # Parse qubits to trace out
        qubits = [int(q.strip()) for q in trace_qubits.split(",")]

        # Create density matrix and perform partial trace
        rho = DensityMatrix.from_instruction(circuit)
        reduced_rho = partial_trace(rho, qubits)

        return {
            "status": "success",
            "reduced_state": {
                "num_qubits": reduced_rho.num_qubits,
                "dim": reduced_rho.dim,
                "purity": reduced_rho.purity(),
                "is_valid": reduced_rho.is_valid(),
            },
            "matrix": reduced_rho.data.tolist(),
            "traced_qubits": qubits,
        }
    except Exception as e:
        logger.error(f"Failed to compute partial trace: {e}")
        return {
            "status": "error",
            "message": f"Failed to compute partial trace: {str(e)}",
        }


async def expectation_value(
    circuit_qasm: str, pauli_strings: str, coeffs: str = ""
) -> Dict[str, Any]:
    """Calculate expectation value of an observable for a quantum state.

    Args:
        circuit_qasm: QASM representation of the circuit
        pauli_strings: Comma-separated Pauli strings (e.g., "XX,YZ,ZZ")
        coeffs: Optional comma-separated coefficients

    Returns:
        Expectation value
    """
    try:
        circuit = qasm2.loads(circuit_qasm)
        circuit.remove_final_measurements(inplace=True)

        # Create observable
        paulis = [p.strip() for p in pauli_strings.split(",")]
        if coeffs:
            coefficients = [float(c.strip()) for c in coeffs.split(",")]
        else:
            coefficients = [1.0] * len(paulis)

        observable = SparsePauliOp(paulis, coeffs=coefficients)

        # Create statevector
        state = Statevector.from_instruction(circuit)

        # Calculate expectation value
        exp_val = state.expectation_value(observable)

        return {
            "status": "success",
            "expectation_value": complex(exp_val).real,
            "observable": str(observable),
            "num_qubits": circuit.num_qubits,
        }
    except Exception as e:
        logger.error(f"Failed to calculate expectation value: {e}")
        return {
            "status": "error",
            "message": f"Failed to calculate expectation value: {str(e)}",
        }


async def random_quantum_state(
    num_qubits: int, seed: Optional[int] = None
) -> Dict[str, Any]:
    """Generate a random quantum statevector.

    Args:
        num_qubits: Number of qubits
        seed: Random seed for reproducibility

    Returns:
        Random statevector information
    """
    try:
        if num_qubits <= 0:
            return {
                "status": "error",
                "message": "Number of qubits must be positive",
            }

        # Generate random statevector
        state = random_statevector(2**num_qubits, seed=seed)

        return {
            "status": "success",
            "statevector": str(state),
            "num_qubits": num_qubits,
            "probabilities": state.probabilities().tolist(),
            "is_valid": state.is_valid(),
        }
    except Exception as e:
        logger.error(f"Failed to generate random state: {e}")
        return {
            "status": "error",
            "message": f"Failed to generate random state: {str(e)}",
        }


async def random_density_matrix_state(
    num_qubits: int, rank: Optional[int] = None, seed: Optional[int] = None
) -> Dict[str, Any]:
    """Generate a random density matrix.

    Args:
        num_qubits: Number of qubits
        rank: Rank of the density matrix (if None, generates full rank)
        seed: Random seed for reproducibility

    Returns:
        Random density matrix information
    """
    try:
        if num_qubits <= 0:
            return {
                "status": "error",
                "message": "Number of qubits must be positive",
            }

        # Generate random density matrix
        rho = random_density_matrix(2**num_qubits, rank=rank, seed=seed)

        return {
            "status": "success",
            "density_matrix": {
                "num_qubits": num_qubits,
                "dim": rho.dim,
                "rank": rank if rank else 2**num_qubits,
                "purity": rho.purity(),
                "is_valid": rho.is_valid(),
            },
            "matrix": rho.data.tolist(),
            "probabilities": rho.probabilities().tolist(),
        }
    except Exception as e:
        logger.error(f"Failed to generate random density matrix: {e}")
        return {
            "status": "error",
            "message": f"Failed to generate random density matrix: {str(e)}",
        }


# Assisted by watsonx Code Assistant
