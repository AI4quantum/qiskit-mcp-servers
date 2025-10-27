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

"""Circuit equivalence checking and comparison operations."""

import logging
from typing import Any, Dict
import numpy as np

from qiskit import qasm2
from qiskit.quantum_info import Operator, process_fidelity

logger = logging.getLogger(__name__)


# ============================================================================
# Circuit Equivalence Checking
# ============================================================================


async def check_circuit_equivalence(
    circuit1_qasm: str, circuit2_qasm: str, tolerance: float = 1e-7
) -> Dict[str, Any]:
    """Check if two circuits are equivalent up to a global phase.

    Args:
        circuit1_qasm: QASM representation of first circuit
        circuit2_qasm: QASM representation of second circuit
        tolerance: Numerical tolerance for comparison

    Returns:
        Equivalence information including fidelity
    """
    try:
        circuit1 = qasm2.loads(circuit1_qasm)
        circuit2 = qasm2.loads(circuit2_qasm)

        # Remove measurements
        circuit1.remove_final_measurements(inplace=True)
        circuit2.remove_final_measurements(inplace=True)

        if circuit1.num_qubits != circuit2.num_qubits:
            return {
                "status": "success",
                "equivalent": False,
                "reason": f"Different number of qubits: {circuit1.num_qubits} vs {circuit2.num_qubits}",
            }

        # Get operators
        op1 = Operator(circuit1)
        op2 = Operator(circuit2)

        # Check equivalence
        equivalent = op1.equiv(op2, atol=tolerance)

        # Calculate process fidelity
        fidelity = process_fidelity(op1, op2)

        return {
            "status": "success",
            "equivalent": bool(equivalent),
            "process_fidelity": float(fidelity),
            "tolerance": tolerance,
            "num_qubits": circuit1.num_qubits,
            "circuit1_depth": circuit1.depth(),
            "circuit2_depth": circuit2.depth(),
        }
    except Exception as e:
        logger.error(f"Failed to check equivalence: {e}")
        return {"status": "error", "message": f"Failed to check equivalence: {str(e)}"}


async def check_unitary_equivalence(
    circuit1_qasm: str, circuit2_qasm: str
) -> Dict[str, Any]:
    """Check if two circuits implement the same unitary operation.

    Args:
        circuit1_qasm: QASM representation of first circuit
        circuit2_qasm: QASM representation of second circuit

    Returns:
        Detailed equivalence information
    """
    try:
        circuit1 = qasm2.loads(circuit1_qasm)
        circuit2 = qasm2.loads(circuit2_qasm)

        # Remove measurements
        circuit1.remove_final_measurements(inplace=True)
        circuit2.remove_final_measurements(inplace=True)

        if circuit1.num_qubits != circuit2.num_qubits:
            return {
                "status": "success",
                "equivalent": False,
                "reason": "Different qubit counts",
            }

        # Get unitary matrices
        op1 = Operator(circuit1)
        op2 = Operator(circuit2)

        # Check if they differ by only a global phase
        # U2 = e^(iφ) * U1
        matrix1 = op1.data
        matrix2 = op2.data

        # Calculate the phase difference
        ratio = matrix2[0, 0] / matrix1[0, 0] if matrix1[0, 0] != 0 else 1.0
        phase = np.angle(ratio)

        # Remove global phase from matrix2
        matrix2_adjusted = matrix2 * np.exp(-1j * phase)

        # Check equivalence
        diff = np.max(np.abs(matrix1 - matrix2_adjusted))
        equivalent = diff < 1e-7

        return {
            "status": "success",
            "equivalent": bool(equivalent),
            "global_phase_difference": float(phase),
            "max_matrix_difference": float(diff),
            "process_fidelity": float(process_fidelity(op1, op2)),
        }
    except Exception as e:
        logger.error(f"Failed to check unitary equivalence: {e}")
        return {
            "status": "error",
            "message": f"Failed to check unitary equivalence: {str(e)}",
        }


async def compare_circuit_resources(
    circuit1_qasm: str, circuit2_qasm: str
) -> Dict[str, Any]:
    """Compare resource usage of two circuits.

    Args:
        circuit1_qasm: QASM representation of first circuit
        circuit2_qasm: QASM representation of second circuit

    Returns:
        Resource comparison (depth, gate count, etc.)
    """
    try:
        circuit1 = qasm2.loads(circuit1_qasm)
        circuit2 = qasm2.loads(circuit2_qasm)

        # Count gates by type
        def count_gates(circuit):
            gate_counts = {}
            for instr in circuit.data:
                gate_name = instr.operation.name
                gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1
            return gate_counts

        gates1 = count_gates(circuit1)
        gates2 = count_gates(circuit2)

        # Count two-qubit gates
        two_q_1 = sum(1 for inst in circuit1.data if len(inst.qubits) == 2)
        two_q_2 = sum(1 for inst in circuit2.data if len(inst.qubits) == 2)

        return {
            "status": "success",
            "circuit1": {
                "depth": circuit1.depth(),
                "size": circuit1.size(),
                "num_qubits": circuit1.num_qubits,
                "gate_counts": gates1,
                "two_qubit_gates": two_q_1,
            },
            "circuit2": {
                "depth": circuit2.depth(),
                "size": circuit2.size(),
                "num_qubits": circuit2.num_qubits,
                "gate_counts": gates2,
                "two_qubit_gates": two_q_2,
            },
            "comparison": {
                "depth_ratio": circuit1.depth() / circuit2.depth()
                if circuit2.depth() > 0
                else float("inf"),
                "size_ratio": circuit1.size() / circuit2.size()
                if circuit2.size() > 0
                else float("inf"),
                "two_qubit_gate_ratio": two_q_1 / two_q_2
                if two_q_2 > 0
                else float("inf"),
            },
        }
    except Exception as e:
        logger.error(f"Failed to compare resources: {e}")
        return {"status": "error", "message": f"Failed to compare resources: {str(e)}"}


async def find_circuit_optimizations(circuit_qasm: str) -> Dict[str, Any]:
    """Suggest optimizations for a circuit.

    Args:
        circuit_qasm: QASM representation of circuit

    Returns:
        Optimization suggestions
    """
    try:
        from qiskit import transpile

        circuit = qasm2.loads(circuit_qasm)

        # Try different optimization levels
        optimizations = {}

        for level in [0, 1, 2, 3]:
            optimized = transpile(circuit, optimization_level=level)

            # Count two-qubit gates
            two_q_gates = sum(1 for inst in optimized.data if len(inst.qubits) == 2)

            optimizations[f"level_{level}"] = {
                "depth": optimized.depth(),
                "size": optimized.size(),
                "two_qubit_gates": two_q_gates,
                "qasm": qasm2.dumps(optimized),
            }

        # Find best optimization
        best_level = min(
            range(4), key=lambda i: optimizations[f"level_{i}"]["two_qubit_gates"]
        )

        return {
            "status": "success",
            "original": {
                "depth": circuit.depth(),
                "size": circuit.size(),
            },
            "optimizations": optimizations,
            "best_optimization_level": best_level,
            "recommendation": f"Use optimization level {best_level} for best two-qubit gate count",
        }
    except Exception as e:
        logger.error(f"Failed to find optimizations: {e}")
        return {"status": "error", "message": f"Failed to find optimizations: {str(e)}"}


async def verify_optimization(
    original_qasm: str, optimized_qasm: str
) -> Dict[str, Any]:
    """Verify that an optimized circuit is equivalent to the original.

    Args:
        original_qasm: QASM representation of original circuit
        optimized_qasm: QASM representation of optimized circuit

    Returns:
        Verification results
    """
    try:
        # Check equivalence
        equiv_result = await check_circuit_equivalence(original_qasm, optimized_qasm)

        if equiv_result["status"] == "error":
            return equiv_result

        # Compare resources
        resource_result = await compare_circuit_resources(original_qasm, optimized_qasm)

        return {
            "status": "success",
            "equivalent": equiv_result.get("equivalent", False),
            "process_fidelity": equiv_result.get("process_fidelity", 0.0),
            "depth_reduction": resource_result["circuit1"]["depth"]
            - resource_result["circuit2"]["depth"],
            "size_reduction": resource_result["circuit1"]["size"]
            - resource_result["circuit2"]["size"],
            "two_qubit_gate_reduction": resource_result["circuit1"]["two_qubit_gates"]
            - resource_result["circuit2"]["two_qubit_gates"],
            "optimization_valid": equiv_result.get("equivalent", False)
            and equiv_result.get("process_fidelity", 0.0) > 0.99,
        }
    except Exception as e:
        logger.error(f"Failed to verify optimization: {e}")
        return {
            "status": "error",
            "message": f"Failed to verify optimization: {str(e)}",
        }


# Assisted by watsonx Code Assistant
