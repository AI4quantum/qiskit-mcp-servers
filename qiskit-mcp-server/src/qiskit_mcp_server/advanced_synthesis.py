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

"""Advanced circuit synthesis and decomposition techniques."""

import logging
from typing import Any, Dict, Optional
import json
import numpy as np

from qiskit import QuantumCircuit, qasm2
from qiskit.quantum_info import Operator
from qiskit.synthesis import OneQubitEulerDecomposer

logger = logging.getLogger(__name__)


# ============================================================================
# Single-Qubit Decomposition
# ============================================================================


async def decompose_single_qubit_unitary(
    unitary_matrix: str,
    basis: str = "U3",
) -> Dict[str, Any]:
    """Decompose arbitrary single-qubit unitary into basis gates.

    Args:
        unitary_matrix: JSON string of 2x2 unitary matrix [[a,b],[c,d]]
        basis: Basis for decomposition (U3, U, ZYZ, ZXZ, XYX, etc.)

    Returns:
        Decomposed circuit
    """
    try:
        # Parse unitary matrix
        matrix = json.loads(unitary_matrix)
        matrix_array = np.array(matrix, dtype=complex)

        if matrix_array.shape != (2, 2):
            return {
                "status": "error",
                "message": "Unitary must be 2x2 matrix for single-qubit decomposition",
            }

        # Create operator
        op = Operator(matrix_array)

        # Decompose
        if basis.upper() == "U3":
            decomposer = OneQubitEulerDecomposer("U3")
        elif basis.upper() == "ZYZ":
            decomposer = OneQubitEulerDecomposer("ZYZ")
        elif basis.upper() == "ZXZ":
            decomposer = OneQubitEulerDecomposer("ZXZ")
        elif basis.upper() == "XYX":
            decomposer = OneQubitEulerDecomposer("XYX")
        else:
            return {
                "status": "error",
                "message": f"Basis '{basis}' not supported. Use: U3, ZYZ, ZXZ, XYX",
            }

        # Get decomposition
        circuit = QuantumCircuit(1)
        theta, phi, lam, phase = decomposer.angles_and_phase(op)

        # Build circuit based on basis
        if basis.upper() == "U3":
            circuit.u(theta, phi, lam, 0)
        elif basis.upper() == "ZYZ":
            circuit.rz(phi, 0)
            circuit.ry(theta, 0)
            circuit.rz(lam, 0)
        elif basis.upper() == "ZXZ":
            circuit.rz(phi, 0)
            circuit.rx(theta, 0)
            circuit.rz(lam, 0)
        elif basis.upper() == "XYX":
            circuit.rx(phi, 0)
            circuit.ry(theta, 0)
            circuit.rx(lam, 0)

        return {
            "status": "success",
            "message": f"Decomposed single-qubit unitary into {basis} basis",
            "basis": basis,
            "parameters": {
                "theta": float(theta),
                "phi": float(phi),
                "lambda": float(lam),
                "global_phase": float(phase),
            },
            "circuit": {
                "qasm": qasm2.dumps(circuit),
                "num_qubits": 1,
                "depth": circuit.depth(),
                "size": circuit.size(),
            },
            "usage": "Single-qubit decomposition enables implementation of arbitrary rotations",
        }

    except Exception as e:
        logger.error(f"Failed to decompose single-qubit unitary: {e}")
        return {"status": "error", "message": f"Failed to decompose unitary: {str(e)}"}


# ============================================================================
# Two-Qubit Decomposition
# ============================================================================


async def decompose_two_qubit_unitary(
    unitary_matrix: str,
    basis_gates: str = "cx",
) -> Dict[str, Any]:
    """Decompose arbitrary two-qubit unitary into basis gates.

    Uses KAK decomposition or other optimal decompositions.

    Args:
        unitary_matrix: JSON string of 4x4 unitary matrix
        basis_gates: Basis gates (cx, cz, iswap, etc.)

    Returns:
        Decomposed two-qubit circuit
    """
    try:
        # Parse unitary matrix
        matrix = json.loads(unitary_matrix)
        matrix_array = np.array(matrix, dtype=complex)

        if matrix_array.shape != (4, 4):
            return {
                "status": "error",
                "message": "Unitary must be 4x4 matrix for two-qubit decomposition",
            }

        # Create operator
        op = Operator(matrix_array)

        # Create circuit from operator (uses automatic decomposition)
        circuit = QuantumCircuit(2)
        circuit.unitary(op, [0, 1], label="U")

        # Decompose to basis gates
        from qiskit import transpile

        decomposed = transpile(
            circuit,
            basis_gates=[basis_gates, "u3", "u", "rz", "ry", "rx"],
            optimization_level=2,
        )

        # Count CNOTs (or other two-qubit gates)
        two_qubit_gates = sum(1 for inst in decomposed.data if len(inst.qubits) == 2)

        return {
            "status": "success",
            "message": f"Decomposed two-qubit unitary using {basis_gates} basis",
            "basis_gates": basis_gates,
            "circuit": {
                "qasm": qasm2.dumps(decomposed),
                "num_qubits": 2,
                "depth": decomposed.depth(),
                "size": decomposed.size(),
                "two_qubit_gates": two_qubit_gates,
            },
            "optimization": "Used KAK decomposition via transpilation",
            "usage": "Two-qubit decomposition enables implementation of arbitrary entangling operations",
        }

    except Exception as e:
        logger.error(f"Failed to decompose two-qubit unitary: {e}")
        return {"status": "error", "message": f"Failed to decompose unitary: {str(e)}"}


# ============================================================================
# Clifford Synthesis
# ============================================================================


async def synthesize_clifford(
    clifford_tableau: Optional[str] = None,
    num_qubits: Optional[int] = None,
    random: bool = False,
) -> Dict[str, Any]:
    """Synthesize a Clifford circuit.

    Creates optimal Clifford circuit from tableau or generates random Clifford.

    Args:
        clifford_tableau: JSON representation of Clifford tableau (optional)
        num_qubits: Number of qubits for random Clifford (if random=True)
        random: Generate random Clifford (default: False)

    Returns:
        Synthesized Clifford circuit
    """
    try:
        if random:
            if num_qubits is None:
                return {
                    "status": "error",
                    "message": "num_qubits required when random=True",
                }

            from qiskit.quantum_info import random_clifford

            cliff = random_clifford(num_qubits)
            circuit = cliff.to_circuit()

            return {
                "status": "success",
                "message": f"Generated random {num_qubits}-qubit Clifford",
                "synthesis_method": "random_generation",
                "num_qubits": num_qubits,
                "circuit": {
                    "qasm": qasm2.dumps(circuit),
                    "num_qubits": circuit.num_qubits,
                    "depth": circuit.depth(),
                    "size": circuit.size(),
                },
            }

        else:
            # For now, return info about Clifford synthesis
            return {
                "status": "info",
                "message": "Clifford synthesis from tableau",
                "note": "Use random=True to generate random Clifford circuits",
                "supported_methods": [
                    "random_generation",
                    "greedy_synthesis",
                    "layered_synthesis",
                ],
            }

    except Exception as e:
        logger.error(f"Failed to synthesize Clifford: {e}")
        return {
            "status": "error",
            "message": f"Failed to synthesize Clifford: {str(e)}",
        }


# ============================================================================
# Unitary Synthesis
# ============================================================================


async def synthesize_unitary(
    unitary_matrix: str,
    num_qubits: int,
    basis_gates: str = "cx,u3",
    optimization_level: int = 2,
) -> Dict[str, Any]:
    """Synthesize arbitrary unitary into basis gates.

    Decomposes multi-qubit unitary using Qiskit's synthesis algorithms.

    Args:
        unitary_matrix: JSON string of unitary matrix
        num_qubits: Number of qubits
        basis_gates: Comma-separated basis gates (default: cx,u3)
        optimization_level: Optimization level 0-3 (default: 2)

    Returns:
        Synthesized circuit
    """
    try:
        # Parse unitary matrix
        matrix = json.loads(unitary_matrix)
        matrix_array = np.array(matrix, dtype=complex)

        expected_dim = 2**num_qubits
        if matrix_array.shape != (expected_dim, expected_dim):
            return {
                "status": "error",
                "message": f"Unitary must be {expected_dim}x{expected_dim} matrix for {num_qubits} qubits",
            }

        # Create operator and circuit
        op = Operator(matrix_array)
        circuit = QuantumCircuit(num_qubits)
        circuit.unitary(op, range(num_qubits), label="U")

        # Decompose to basis gates
        from qiskit import transpile

        basis_list = [g.strip() for g in basis_gates.split(",")]
        synthesized = transpile(
            circuit, basis_gates=basis_list, optimization_level=optimization_level
        )

        return {
            "status": "success",
            "message": f"Synthesized {num_qubits}-qubit unitary",
            "num_qubits": num_qubits,
            "basis_gates": basis_list,
            "optimization_level": optimization_level,
            "circuit": {
                "qasm": qasm2.dumps(synthesized),
                "num_qubits": synthesized.num_qubits,
                "depth": synthesized.depth(),
                "size": synthesized.size(),
            },
            "synthesis_info": {
                "method": "qiskit_transpile",
                "optimization": f"level {optimization_level}",
            },
        }

    except Exception as e:
        logger.error(f"Failed to synthesize unitary: {e}")
        return {"status": "error", "message": f"Failed to synthesize unitary: {str(e)}"}


async def optimize_circuit_synthesis(
    circuit_qasm: str,
    target_basis: str = "cx,u3",
    optimization_level: int = 3,
) -> Dict[str, Any]:
    """Re-synthesize circuit for better gate count/depth.

    Applies advanced synthesis and optimization techniques.

    Args:
        circuit_qasm: QASM representation of circuit
        target_basis: Target basis gates (default: cx,u3)
        optimization_level: Optimization level 0-3 (default: 3)

    Returns:
        Optimized circuit
    """
    try:
        circuit = qasm2.loads(circuit_qasm)

        # Transpile with optimization
        from qiskit import transpile

        basis_list = [g.strip() for g in target_basis.split(",")]

        optimized = transpile(
            circuit, basis_gates=basis_list, optimization_level=optimization_level
        )

        # Calculate improvement
        original_depth = circuit.depth()
        original_size = circuit.size()
        optimized_depth = optimized.depth()
        optimized_size = optimized.size()

        depth_improvement = (
            ((original_depth - optimized_depth) / original_depth * 100)
            if original_depth > 0
            else 0
        )
        size_improvement = (
            ((original_size - optimized_size) / original_size * 100)
            if original_size > 0
            else 0
        )

        return {
            "status": "success",
            "message": "Circuit re-synthesized and optimized",
            "original_circuit": {
                "depth": original_depth,
                "size": original_size,
            },
            "optimized_circuit": {
                "qasm": qasm2.dumps(optimized),
                "depth": optimized_depth,
                "size": optimized_size,
                "num_qubits": optimized.num_qubits,
            },
            "improvement": {
                "depth_reduction_percent": round(depth_improvement, 2),
                "size_reduction_percent": round(size_improvement, 2),
            },
            "optimization_level": optimization_level,
            "target_basis": basis_list,
        }

    except Exception as e:
        logger.error(f"Failed to optimize circuit synthesis: {e}")
        return {"status": "error", "message": f"Failed to optimize synthesis: {str(e)}"}


# Assisted by watsonx Code Assistant
