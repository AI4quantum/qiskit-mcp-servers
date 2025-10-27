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

"""Extended circuit library with N-local, Boolean logic, arithmetic, and data encoding circuits."""

import logging
from typing import Any, Dict

from qiskit import QuantumCircuit, qasm2
from qiskit.circuit.library import (
    TwoLocal,
    NLocal,
    PauliFeatureMap,
    ZFeatureMap,
    ZZFeatureMap,
    QAOAAnsatz,
    AND,
    OR,
    XOR,
    HiddenLinearFunction,
    IQP,
    PhaseEstimation,
)

logger = logging.getLogger(__name__)


# ============================================================================
# N-Local Circuits
# ============================================================================


async def create_two_local_circuit(
    num_qubits: int,
    rotation_blocks: str = "ry",
    entanglement_blocks: str = "cx",
    reps: int = 3,
    entanglement: str = "full",
) -> Dict[str, Any]:
    """Create a TwoLocal variational circuit.

    Args:
        num_qubits: Number of qubits
        rotation_blocks: Rotation gates (ry, rx, rz) or comma-separated list
        entanglement_blocks: Entanglement gates (cx, cz, etc.)
        reps: Number of repetitions
        entanglement: Entanglement pattern (full, linear, circular, sca)

    Returns:
        TwoLocal circuit information
    """
    try:
        # Parse rotation blocks
        if "," in rotation_blocks:
            rot_blocks = [b.strip() for b in rotation_blocks.split(",")]
        else:
            rot_blocks = rotation_blocks

        circuit = TwoLocal(
            num_qubits=num_qubits,
            rotation_blocks=rot_blocks,
            entanglement_blocks=entanglement_blocks,
            reps=reps,
            entanglement=entanglement,
        )

        return {
            "status": "success",
            "message": f"Created TwoLocal circuit with {num_qubits} qubits and {reps} reps",
            "circuit": {
                "name": "TwoLocal",
                "num_qubits": circuit.num_qubits,
                "num_clbits": circuit.num_clbits,
                "depth": circuit.depth(),
                "size": circuit.size(),
                "num_parameters": circuit.num_parameters,
                "qasm": qasm2.dumps(circuit),
            },
            "parameters": [str(p) for p in circuit.parameters],
        }
    except Exception as e:
        logger.error(f"Failed to create TwoLocal circuit: {e}")
        return {
            "status": "error",
            "message": f"Failed to create TwoLocal circuit: {str(e)}",
        }


async def create_n_local_circuit(
    num_qubits: int,
    rotation_blocks: str = "ry",
    entanglement_blocks: str = "cx",
    num_qubits_per_block: int = 2,
    reps: int = 3,
) -> Dict[str, Any]:
    """Create an NLocal variational circuit.

    Args:
        num_qubits: Number of qubits
        rotation_blocks: Rotation gates
        entanglement_blocks: Entanglement gates
        num_qubits_per_block: Qubits per entanglement block
        reps: Number of repetitions

    Returns:
        NLocal circuit information
    """
    try:
        circuit = NLocal(
            num_qubits=num_qubits,
            rotation_blocks=rotation_blocks,
            entanglement_blocks=entanglement_blocks,
            entanglement=[[i, (i + 1) % num_qubits] for i in range(num_qubits)],
            reps=reps,
        )

        return {
            "status": "success",
            "message": f"Created NLocal circuit with {num_qubits} qubits",
            "circuit": {
                "name": "NLocal",
                "num_qubits": circuit.num_qubits,
                "num_clbits": circuit.num_clbits,
                "depth": circuit.depth(),
                "size": circuit.size(),
                "num_parameters": circuit.num_parameters,
                "qasm": qasm2.dumps(circuit),
            },
        }
    except Exception as e:
        logger.error(f"Failed to create NLocal circuit: {e}")
        return {
            "status": "error",
            "message": f"Failed to create NLocal circuit: {str(e)}",
        }


# ============================================================================
# Data Encoding Circuits
# ============================================================================


async def create_pauli_feature_map(
    feature_dimension: int,
    reps: int = 2,
    paulis: str = "Z,ZZ",
) -> Dict[str, Any]:
    """Create a PauliFeatureMap for data encoding.

    Args:
        feature_dimension: Number of features (qubits)
        reps: Number of repetitions
        paulis: Comma-separated Pauli strings (e.g., "Z,ZZ,ZZZ")

    Returns:
        PauliFeatureMap circuit information
    """
    try:
        pauli_list = [p.strip() for p in paulis.split(",")]

        circuit = PauliFeatureMap(
            feature_dimension=feature_dimension,
            reps=reps,
            paulis=pauli_list,
        )

        return {
            "status": "success",
            "message": f"Created PauliFeatureMap with {feature_dimension} features",
            "circuit": {
                "name": "PauliFeatureMap",
                "num_qubits": circuit.num_qubits,
                "num_clbits": circuit.num_clbits,
                "depth": circuit.depth(),
                "size": circuit.size(),
                "num_parameters": circuit.num_parameters,
                "qasm": qasm2.dumps(circuit),
            },
            "parameters": [str(p) for p in circuit.parameters],
        }
    except Exception as e:
        logger.error(f"Failed to create PauliFeatureMap: {e}")
        return {
            "status": "error",
            "message": f"Failed to create PauliFeatureMap: {str(e)}",
        }


async def create_z_feature_map(feature_dimension: int, reps: int = 2) -> Dict[str, Any]:
    """Create a ZFeatureMap for data encoding.

    Args:
        feature_dimension: Number of features (qubits)
        reps: Number of repetitions

    Returns:
        ZFeatureMap circuit information
    """
    try:
        circuit = ZFeatureMap(feature_dimension=feature_dimension, reps=reps)

        return {
            "status": "success",
            "message": f"Created ZFeatureMap with {feature_dimension} features",
            "circuit": {
                "name": "ZFeatureMap",
                "num_qubits": circuit.num_qubits,
                "num_clbits": circuit.num_clbits,
                "depth": circuit.depth(),
                "size": circuit.size(),
                "num_parameters": circuit.num_parameters,
                "qasm": qasm2.dumps(circuit),
            },
        }
    except Exception as e:
        logger.error(f"Failed to create ZFeatureMap: {e}")
        return {"status": "error", "message": f"Failed to create ZFeatureMap: {str(e)}"}


async def create_zz_feature_map(
    feature_dimension: int, reps: int = 2, entanglement: str = "full"
) -> Dict[str, Any]:
    """Create a ZZFeatureMap for data encoding with entanglement.

    Args:
        feature_dimension: Number of features (qubits)
        reps: Number of repetitions
        entanglement: Entanglement pattern (full, linear, circular)

    Returns:
        ZZFeatureMap circuit information
    """
    try:
        circuit = ZZFeatureMap(
            feature_dimension=feature_dimension,
            reps=reps,
            entanglement=entanglement,
        )

        return {
            "status": "success",
            "message": f"Created ZZFeatureMap with {feature_dimension} features",
            "circuit": {
                "name": "ZZFeatureMap",
                "num_qubits": circuit.num_qubits,
                "num_clbits": circuit.num_clbits,
                "depth": circuit.depth(),
                "size": circuit.size(),
                "num_parameters": circuit.num_parameters,
                "qasm": qasm2.dumps(circuit),
            },
        }
    except Exception as e:
        logger.error(f"Failed to create ZZFeatureMap: {e}")
        return {
            "status": "error",
            "message": f"Failed to create ZZFeatureMap: {str(e)}",
        }


async def create_qaoa_ansatz(cost_operator: str, reps: int = 1) -> Dict[str, Any]:
    """Create a QAOA ansatz circuit.

    Args:
        cost_operator: Pauli string for cost Hamiltonian (e.g., "ZZIIZI")
        reps: Number of QAOA layers

    Returns:
        QAOAAnsatz circuit information
    """
    try:
        from qiskit.quantum_info import SparsePauliOp

        # Create cost operator
        hamiltonian = SparsePauliOp(cost_operator)

        circuit = QAOAAnsatz(cost_operator=hamiltonian, reps=reps)

        return {
            "status": "success",
            "message": f"Created QAOA ansatz with {reps} repetitions",
            "circuit": {
                "name": "QAOAAnsatz",
                "num_qubits": circuit.num_qubits,
                "num_clbits": circuit.num_clbits,
                "depth": circuit.depth(),
                "size": circuit.size(),
                "num_parameters": circuit.num_parameters,
                "qasm": qasm2.dumps(circuit),
            },
            "parameters": [str(p) for p in circuit.parameters],
        }
    except Exception as e:
        logger.error(f"Failed to create QAOA ansatz: {e}")
        return {"status": "error", "message": f"Failed to create QAOA ansatz: {str(e)}"}


# ============================================================================
# Boolean Logic Circuits
# ============================================================================


async def create_and_gate(num_variable_qubits: int) -> Dict[str, Any]:
    """Create a quantum AND gate circuit.

    Args:
        num_variable_qubits: Number of input qubits

    Returns:
        AND gate circuit information
    """
    try:
        gate = AND(num_variable_qubits=num_variable_qubits)
        circuit = QuantumCircuit(num_variable_qubits + 1)
        circuit.append(gate, range(num_variable_qubits + 1))

        return {
            "status": "success",
            "message": f"Created AND gate with {num_variable_qubits} input qubits",
            "circuit": {
                "name": "AND",
                "num_qubits": circuit.num_qubits,
                "num_clbits": circuit.num_clbits,
                "depth": circuit.depth(),
                "size": circuit.size(),
                "qasm": qasm2.dumps(circuit),
            },
        }
    except Exception as e:
        logger.error(f"Failed to create AND gate: {e}")
        return {"status": "error", "message": f"Failed to create AND gate: {str(e)}"}


async def create_or_gate(num_variable_qubits: int) -> Dict[str, Any]:
    """Create a quantum OR gate circuit.

    Args:
        num_variable_qubits: Number of input qubits

    Returns:
        OR gate circuit information
    """
    try:
        gate = OR(num_variable_qubits=num_variable_qubits)
        circuit = QuantumCircuit(num_variable_qubits + 1)
        circuit.append(gate, range(num_variable_qubits + 1))

        return {
            "status": "success",
            "message": f"Created OR gate with {num_variable_qubits} input qubits",
            "circuit": {
                "name": "OR",
                "num_qubits": circuit.num_qubits,
                "num_clbits": circuit.num_clbits,
                "depth": circuit.depth(),
                "size": circuit.size(),
                "qasm": qasm2.dumps(circuit),
            },
        }
    except Exception as e:
        logger.error(f"Failed to create OR gate: {e}")
        return {"status": "error", "message": f"Failed to create OR gate: {str(e)}"}


async def create_xor_gate(num_qubits: int) -> Dict[str, Any]:
    """Create a quantum XOR gate circuit.

    Args:
        num_qubits: Number of qubits

    Returns:
        XOR gate circuit information
    """
    try:
        gate = XOR(num_qubits=num_qubits)
        circuit = QuantumCircuit(num_qubits + 1)
        circuit.append(gate, range(num_qubits + 1))

        return {
            "status": "success",
            "message": f"Created XOR gate with {num_qubits} qubits",
            "circuit": {
                "name": "XOR",
                "num_qubits": circuit.num_qubits,
                "num_clbits": circuit.num_clbits,
                "depth": circuit.depth(),
                "size": circuit.size(),
                "qasm": qasm2.dumps(circuit),
            },
        }
    except Exception as e:
        logger.error(f"Failed to create XOR gate: {e}")
        return {"status": "error", "message": f"Failed to create XOR gate: {str(e)}"}


# ============================================================================
# Particular Quantum Circuits
# ============================================================================


async def create_hidden_linear_function(num_qubits: int) -> Dict[str, Any]:
    """Create a HiddenLinearFunction circuit.

    Args:
        num_qubits: Number of qubits

    Returns:
        HiddenLinearFunction circuit information
    """
    try:
        import numpy as np

        # Create random binary matrix
        adjacency_matrix = np.random.randint(0, 2, size=(num_qubits, num_qubits))

        circuit = HiddenLinearFunction(adjacency_matrix)

        return {
            "status": "success",
            "message": f"Created HiddenLinearFunction with {num_qubits} qubits",
            "circuit": {
                "name": "HiddenLinearFunction",
                "num_qubits": circuit.num_qubits,
                "num_clbits": circuit.num_clbits,
                "depth": circuit.depth(),
                "size": circuit.size(),
                "qasm": qasm2.dumps(circuit),
            },
        }
    except Exception as e:
        logger.error(f"Failed to create HiddenLinearFunction: {e}")
        return {
            "status": "error",
            "message": f"Failed to create HiddenLinearFunction: {str(e)}",
        }


async def create_iqp_circuit(num_qubits: int) -> Dict[str, Any]:
    """Create an IQP (Instantaneous Quantum Polynomial) circuit.

    Args:
        num_qubits: Number of qubits

    Returns:
        IQP circuit information
    """
    try:
        import numpy as np

        # Create random interaction matrix
        interactions = np.random.rand(num_qubits, num_qubits)
        interactions = (interactions + interactions.T) / 2  # Make symmetric

        circuit = IQP(interactions)

        return {
            "status": "success",
            "message": f"Created IQP circuit with {num_qubits} qubits",
            "circuit": {
                "name": "IQP",
                "num_qubits": circuit.num_qubits,
                "num_clbits": circuit.num_clbits,
                "depth": circuit.depth(),
                "size": circuit.size(),
                "qasm": qasm2.dumps(circuit),
            },
        }
    except Exception as e:
        logger.error(f"Failed to create IQP circuit: {e}")
        return {"status": "error", "message": f"Failed to create IQP circuit: {str(e)}"}


async def create_phase_estimation_circuit(
    num_evaluation_qubits: int, unitary_circuit_qasm: str
) -> Dict[str, Any]:
    """Create a Phase Estimation circuit.

    Args:
        num_evaluation_qubits: Number of evaluation qubits
        unitary_circuit_qasm: QASM of the unitary operator

    Returns:
        PhaseEstimation circuit information
    """
    try:
        from qiskit import qasm2

        unitary_circuit = qasm2.loads(unitary_circuit_qasm)

        circuit = PhaseEstimation(num_evaluation_qubits, unitary_circuit)

        return {
            "status": "success",
            "message": f"Created PhaseEstimation with {num_evaluation_qubits} evaluation qubits",
            "circuit": {
                "name": "PhaseEstimation",
                "num_qubits": circuit.num_qubits,
                "num_clbits": circuit.num_clbits,
                "depth": circuit.depth(),
                "size": circuit.size(),
                "qasm": qasm2.dumps(circuit),
            },
        }
    except Exception as e:
        logger.error(f"Failed to create PhaseEstimation: {e}")
        return {
            "status": "error",
            "message": f"Failed to create PhaseEstimation: {str(e)}",
        }


# Assisted by watsonx Code Assistant
