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

"""Clifford operations and stabilizer formalism for efficient quantum simulation."""

import logging
from typing import Any, Dict, Optional
import numpy as np

from qiskit import qasm2
from qiskit.quantum_info import Clifford, random_clifford, StabilizerState, Pauli

logger = logging.getLogger(__name__)


# ============================================================================
# Clifford Circuit Operations
# ============================================================================


async def create_random_clifford(
    num_qubits: int, seed: Optional[int] = None
) -> Dict[str, Any]:
    """Create a random Clifford circuit.

    Clifford circuits can be efficiently simulated classically.
    Useful for randomized benchmarking and testing.

    Args:
        num_qubits: Number of qubits
        seed: Random seed for reproducibility (optional)

    Returns:
        Random Clifford circuit information
    """
    try:
        if seed is not None:
            np.random.seed(seed)

        # Generate random Clifford
        cliff = random_clifford(num_qubits, seed=seed)

        # Convert to circuit
        circuit = cliff.to_circuit()

        return {
            "status": "success",
            "message": f"Created random {num_qubits}-qubit Clifford circuit",
            "num_qubits": num_qubits,
            "circuit": {
                "qasm": qasm2.dumps(circuit),
                "num_qubits": circuit.num_qubits,
                "depth": circuit.depth(),
                "size": circuit.size(),
            },
            "clifford_properties": {
                "is_unitary": True,
                "is_clifford": True,
                "num_qubits": num_qubits,
            },
            "seed": seed,
            "usage": "Clifford circuits are efficiently simulable and used in randomized benchmarking",
        }

    except Exception as e:
        logger.error(f"Failed to create random Clifford: {e}")
        return {
            "status": "error",
            "message": f"Failed to create random Clifford: {str(e)}",
        }


async def circuit_to_clifford(circuit_qasm: str) -> Dict[str, Any]:
    """Convert a Clifford circuit to Clifford representation.

    Verifies that the circuit is Clifford and provides Clifford tableau.

    Args:
        circuit_qasm: QASM representation of circuit (must be Clifford)

    Returns:
        Clifford representation and properties
    """
    try:
        circuit = qasm2.loads(circuit_qasm)

        # Convert to Clifford
        try:
            cliff = Clifford(circuit)
        except Exception as e:
            return {
                "status": "error",
                "message": f"Circuit is not Clifford: {str(e)}",
                "note": "Clifford circuits contain only H, S, CX, and Pauli gates",
            }

        # Get stabilizer tableau
        tableau = cliff.tableau

        return {
            "status": "success",
            "message": "Circuit is Clifford",
            "num_qubits": cliff.num_qubits,
            "clifford_properties": {
                "is_unitary": True,
                "is_clifford": True,
                "num_qubits": cliff.num_qubits,
            },
            "tableau": {
                "shape": list(tableau.shape),
                "description": "Stabilizer tableau representing Clifford operation",
            },
            "original_circuit": {
                "depth": circuit.depth(),
                "size": circuit.size(),
            },
            "usage": "Clifford representation enables efficient simulation via stabilizer formalism",
        }

    except Exception as e:
        logger.error(f"Failed to convert to Clifford: {e}")
        return {
            "status": "error",
            "message": f"Failed to convert to Clifford: {str(e)}",
        }


async def compose_cliffords(clifford1_qasm: str, clifford2_qasm: str) -> Dict[str, Any]:
    """Compose two Clifford circuits.

    Clifford composition is efficient and maintains Clifford property.

    Args:
        clifford1_qasm: QASM of first Clifford circuit
        clifford2_qasm: QASM of second Clifford circuit

    Returns:
        Composed Clifford circuit
    """
    try:
        circuit1 = qasm2.loads(clifford1_qasm)
        circuit2 = qasm2.loads(clifford2_qasm)

        # Convert to Clifford
        cliff1 = Clifford(circuit1)
        cliff2 = Clifford(circuit2)

        # Compose
        composed = cliff1.compose(cliff2)

        # Convert back to circuit
        composed_circuit = composed.to_circuit()

        return {
            "status": "success",
            "message": "Clifford circuits composed",
            "num_qubits": composed.num_qubits,
            "circuit": {
                "qasm": qasm2.dumps(composed_circuit),
                "num_qubits": composed_circuit.num_qubits,
                "depth": composed_circuit.depth(),
                "size": composed_circuit.size(),
            },
            "composition": {
                "circuit1_depth": circuit1.depth(),
                "circuit2_depth": circuit2.depth(),
                "composed_depth": composed_circuit.depth(),
            },
        }

    except Exception as e:
        logger.error(f"Failed to compose Cliffords: {e}")
        return {"status": "error", "message": f"Failed to compose Cliffords: {str(e)}"}


# ============================================================================
# Stabilizer State Operations
# ============================================================================


async def create_stabilizer_state(circuit_qasm: str) -> Dict[str, Any]:
    """Create a stabilizer state from a Clifford circuit.

    Stabilizer states can be efficiently represented and simulated.

    Args:
        circuit_qasm: QASM of Clifford circuit

    Returns:
        Stabilizer state representation
    """
    try:
        circuit = qasm2.loads(circuit_qasm)

        # Create stabilizer state
        try:
            stab_state = StabilizerState(circuit)
        except Exception as e:
            return {
                "status": "error",
                "message": f"Cannot create stabilizer state: {str(e)}",
                "note": "Input must be a Clifford circuit (H, S, CX, Pauli gates only)",
            }

        # Get stabilizers
        _ = stab_state.clifford

        return {
            "status": "success",
            "message": "Stabilizer state created",
            "num_qubits": stab_state.num_qubits,
            "state_properties": {
                "is_pure": True,
                "is_stabilizer": True,
                "num_qubits": stab_state.num_qubits,
            },
            "clifford_depth": circuit.depth(),
            "usage": "Stabilizer states are efficiently simulable and useful for error correction",
        }

    except Exception as e:
        logger.error(f"Failed to create stabilizer state: {e}")
        return {
            "status": "error",
            "message": f"Failed to create stabilizer state: {str(e)}",
        }


async def measure_stabilizer_state(
    circuit_qasm: str,
    qubits: Optional[str] = None,
) -> Dict[str, Any]:
    """Measure a stabilizer state.

    Stabilizer measurements can be efficiently simulated.

    Args:
        circuit_qasm: QASM of Clifford circuit
        qubits: Comma-separated qubit indices to measure (optional, defaults to all)

    Returns:
        Measurement outcome (deterministic for stabilizer states)
    """
    try:
        circuit = qasm2.loads(circuit_qasm)

        # Create stabilizer state
        stab_state = StabilizerState(circuit)

        # Parse qubits to measure
        if qubits:
            qubit_list = [int(q.strip()) for q in qubits.split(",")]
        else:
            qubit_list = list(range(circuit.num_qubits))

        # Measure
        outcome = stab_state.measure(qubit_list)

        # Convert outcome to bitstring
        if isinstance(outcome, (list, np.ndarray)):
            bitstring = "".join(str(int(b)) for b in outcome)
        else:
            bitstring = str(outcome)

        return {
            "status": "success",
            "message": "Stabilizer state measured",
            "measurement_outcome": bitstring,
            "measured_qubits": qubit_list,
            "num_qubits": circuit.num_qubits,
            "note": "Stabilizer measurements are deterministic or random depending on the state",
        }

    except Exception as e:
        logger.error(f"Failed to measure stabilizer state: {e}")
        return {
            "status": "error",
            "message": f"Failed to measure stabilizer state: {str(e)}",
        }


async def check_if_clifford(circuit_qasm: str) -> Dict[str, Any]:
    """Check if a circuit is Clifford.

    Determines if circuit can be efficiently simulated via stabilizer formalism.

    Args:
        circuit_qasm: QASM representation of circuit

    Returns:
        Boolean result and analysis
    """
    try:
        circuit = qasm2.loads(circuit_qasm)

        # Try to convert to Clifford
        is_clifford = False
        error_message = None

        try:
            _ = Clifford(circuit)
            is_clifford = True
        except Exception as e:
            error_message = str(e)

        # Analyze gates
        clifford_gates = {
            "h",
            "s",
            "sdg",
            "cx",
            "cy",
            "cz",
            "swap",
            "x",
            "y",
            "z",
            "id",
            "barrier",
        }
        circuit_gates = set()
        non_clifford_gates = []

        for instruction in circuit.data:
            gate_name = instruction.operation.name.lower()
            circuit_gates.add(gate_name)
            if gate_name not in clifford_gates and gate_name != "measure":
                non_clifford_gates.append(gate_name)

        return {
            "status": "success",
            "is_clifford": is_clifford,
            "num_qubits": circuit.num_qubits,
            "circuit_depth": circuit.depth(),
            "circuit_size": circuit.size(),
            "gates_used": list(circuit_gates),
            "non_clifford_gates": non_clifford_gates if non_clifford_gates else None,
            "error_message": error_message if not is_clifford else None,
            "clifford_gates": list(clifford_gates),
            "note": "Clifford circuits can be efficiently simulated classically",
        }

    except Exception as e:
        logger.error(f"Failed to check if Clifford: {e}")
        return {"status": "error", "message": f"Failed to check if Clifford: {str(e)}"}


async def pauli_tracking(circuit_qasm: str, initial_pauli: str) -> Dict[str, Any]:
    """Track how a Pauli operator evolves through a Clifford circuit.

    Uses Clifford conjugation to track Pauli evolution.

    Args:
        circuit_qasm: QASM of Clifford circuit
        initial_pauli: Initial Pauli string (e.g., "XYZ", "IIZX")

    Returns:
        Final Pauli operator after circuit evolution
    """
    try:
        circuit = qasm2.loads(circuit_qasm)

        # Convert to Clifford
        try:
            cliff = Clifford(circuit)
        except Exception as e:
            return {
                "status": "error",
                "message": f"Circuit must be Clifford for Pauli tracking: {str(e)}",
            }

        # Create initial Pauli
        initial = Pauli(initial_pauli)

        # Evolve Pauli through Clifford
        final_pauli = initial.evolve(cliff, frame="s")

        return {
            "status": "success",
            "message": "Pauli operator tracked through Clifford circuit",
            "initial_pauli": str(initial),
            "final_pauli": str(final_pauli),
            "num_qubits": circuit.num_qubits,
            "circuit_depth": circuit.depth(),
            "usage": "Pauli tracking is useful for error propagation analysis and fault-tolerant quantum computing",
        }

    except Exception as e:
        logger.error(f"Failed to track Pauli: {e}")
        return {"status": "error", "message": f"Failed to track Pauli: {str(e)}"}


# Assisted by watsonx Code Assistant
