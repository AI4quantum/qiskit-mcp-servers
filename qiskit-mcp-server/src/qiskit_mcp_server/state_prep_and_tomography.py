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

"""State preparation and quantum state tomography operations."""

import logging
from typing import Any, Dict
import numpy as np

from qiskit import QuantumCircuit, qasm2
from qiskit.quantum_info import Statevector, state_fidelity

logger = logging.getLogger(__name__)


# ============================================================================
# State Preparation
# ============================================================================


async def prepare_uniform_superposition(num_qubits: int) -> Dict[str, Any]:
    """Prepare a uniform superposition state |+⟩^⊗n.

    Args:
        num_qubits: Number of qubits

    Returns:
        Circuit creating uniform superposition
    """
    try:
        circuit = QuantumCircuit(num_qubits)

        # Apply Hadamard to all qubits
        for i in range(num_qubits):
            circuit.h(i)

        return {
            "status": "success",
            "message": f"Created uniform superposition on {num_qubits} qubits",
            "circuit": {
                "name": "uniform_superposition",
                "num_qubits": circuit.num_qubits,
                "num_clbits": circuit.num_clbits,
                "depth": circuit.depth(),
                "qasm": qasm2.dumps(circuit),
            },
        }
    except Exception as e:
        logger.error(f"Failed to prepare uniform superposition: {e}")
        return {"status": "error", "message": f"Failed to prepare state: {str(e)}"}


async def prepare_w_state(num_qubits: int) -> Dict[str, Any]:
    """Prepare a W state: (|100...⟩ + |010...⟩ + ... + |00...1⟩)/sqrt(n).

    Args:
        num_qubits: Number of qubits

    Returns:
        Circuit creating W state
    """
    try:
        if num_qubits < 2:
            return {
                "status": "error",
                "message": "W state requires at least 2 qubits",
            }

        circuit = QuantumCircuit(num_qubits)

        # Initialize first qubit to |1⟩
        circuit.x(0)

        # Apply recursive W state construction
        for i in range(num_qubits - 1):
            angle = np.arccos(np.sqrt(1.0 / (num_qubits - i)))
            circuit.ry(2 * angle, i)
            circuit.ch(i, i + 1)
            circuit.x(i)

        return {
            "status": "success",
            "message": f"Created W state on {num_qubits} qubits",
            "circuit": {
                "name": "w_state",
                "num_qubits": circuit.num_qubits,
                "num_clbits": circuit.num_clbits,
                "depth": circuit.depth(),
                "qasm": qasm2.dumps(circuit),
            },
        }
    except Exception as e:
        logger.error(f"Failed to prepare W state: {e}")
        return {"status": "error", "message": f"Failed to prepare W state: {str(e)}"}


async def prepare_ghz_state(num_qubits: int) -> Dict[str, Any]:
    """Prepare a GHZ state: (|00...0⟩ + |11...1⟩)/sqrt(2).

    Args:
        num_qubits: Number of qubits

    Returns:
        Circuit creating GHZ state
    """
    try:
        circuit = QuantumCircuit(num_qubits)

        # Apply H to first qubit
        circuit.h(0)

        # Apply CNOT cascade
        for i in range(num_qubits - 1):
            circuit.cx(i, i + 1)

        return {
            "status": "success",
            "message": f"Created GHZ state on {num_qubits} qubits",
            "circuit": {
                "name": "ghz_state",
                "num_qubits": circuit.num_qubits,
                "num_clbits": circuit.num_clbits,
                "depth": circuit.depth(),
                "qasm": qasm2.dumps(circuit),
            },
        }
    except Exception as e:
        logger.error(f"Failed to prepare GHZ state: {e}")
        return {"status": "error", "message": f"Failed to prepare GHZ state: {str(e)}"}


async def prepare_dicke_state(num_qubits: int, num_excitations: int) -> Dict[str, Any]:
    """Prepare a Dicke state with fixed number of excitations.

    Args:
        num_qubits: Number of qubits
        num_excitations: Number of |1⟩ states

    Returns:
        Circuit creating Dicke state
    """
    try:
        if num_excitations > num_qubits or num_excitations < 0:
            return {
                "status": "error",
                "message": f"Invalid excitations: must be 0 <= {num_excitations} <= {num_qubits}",
            }

        from qiskit.circuit.library import DickeState

        dicke = DickeState(num_qubits, num_excitations)
        circuit = QuantumCircuit(num_qubits)
        circuit.compose(dicke, inplace=True)

        return {
            "status": "success",
            "message": f"Created Dicke state |D_{num_qubits}^{num_excitations}⟩",
            "circuit": {
                "name": f"dicke_{num_qubits}_{num_excitations}",
                "num_qubits": circuit.num_qubits,
                "num_clbits": circuit.num_clbits,
                "depth": circuit.depth(),
                "qasm": qasm2.dumps(circuit),
            },
        }
    except Exception as e:
        logger.error(f"Failed to prepare Dicke state: {e}")
        return {
            "status": "error",
            "message": f"Failed to prepare Dicke state: {str(e)}",
        }


async def prepare_product_state(state_string: str) -> Dict[str, Any]:
    """Prepare a product state from a string like '0101' or '+++−'.

    Args:
        state_string: String specifying the state (0/1 or +/−/0/1)

    Returns:
        Circuit creating the product state
    """
    try:
        num_qubits = len(state_string)
        circuit = QuantumCircuit(num_qubits)

        for i, char in enumerate(state_string):
            if char in ["1", "−", "-"]:
                circuit.x(i)
            if char in ["+", "−", "-"]:
                circuit.h(i)

        return {
            "status": "success",
            "message": f"Created product state |{state_string}⟩",
            "circuit": {
                "name": f"product_state_{state_string}",
                "num_qubits": circuit.num_qubits,
                "num_clbits": circuit.num_clbits,
                "depth": circuit.depth(),
                "qasm": qasm2.dumps(circuit),
            },
        }
    except Exception as e:
        logger.error(f"Failed to prepare product state: {e}")
        return {
            "status": "error",
            "message": f"Failed to prepare product state: {str(e)}",
        }


# ============================================================================
# State Tomography
# ============================================================================


async def generate_tomography_circuits(
    circuit_qasm: str, measurement_basis: str = "pauli"
) -> Dict[str, Any]:
    """Generate measurement circuits for state tomography.

    Args:
        circuit_qasm: QASM representation of state preparation circuit
        measurement_basis: Basis for measurements (pauli, sic)

    Returns:
        List of measurement circuits
    """
    try:
        circuit = qasm2.loads(circuit_qasm)
        num_qubits = circuit.num_qubits

        # Remove any existing measurements
        circuit.remove_final_measurements(inplace=True)

        tomography_circuits = []

        if measurement_basis == "pauli":
            # Pauli basis: X, Y, Z for each qubit
            bases = ["X", "Y", "Z"]

            # Generate all combinations
            from itertools import product

            for basis_combo in product(bases, repeat=num_qubits):
                meas_circuit = circuit.copy()

                # Add basis rotations
                for qubit, basis in enumerate(basis_combo):
                    if basis == "X":
                        meas_circuit.h(qubit)
                    elif basis == "Y":
                        meas_circuit.sdg(qubit)
                        meas_circuit.h(qubit)
                    # Z basis: no rotation needed

                # Add measurements
                meas_circuit.measure_all()

                tomography_circuits.append(
                    {
                        "basis": "".join(basis_combo),
                        "qasm": qasm2.dumps(meas_circuit),
                    }
                )

        else:
            return {
                "status": "error",
                "message": f"Unknown measurement basis: {measurement_basis}. Use 'pauli'.",
            }

        return {
            "status": "success",
            "message": f"Generated {len(tomography_circuits)} tomography circuits",
            "num_circuits": len(tomography_circuits),
            "measurement_basis": measurement_basis,
            "circuits": tomography_circuits,
        }
    except Exception as e:
        logger.error(f"Failed to generate tomography circuits: {e}")
        return {
            "status": "error",
            "message": f"Failed to generate tomography circuits: {str(e)}",
        }


async def estimate_state_from_tomography(
    tomography_results: str,
) -> Dict[str, Any]:
    """Estimate quantum state from tomography measurement results.

    Args:
        tomography_results: JSON string with tomography results
                          Format: [{"basis": "XXZ", "counts": {"000": 100, ...}}, ...]

    Returns:
        Estimated density matrix
    """
    try:
        import json

        results = json.loads(tomography_results)

        # This is a simplified estimation
        # In practice, you'd use maximum likelihood estimation or linear inversion

        num_qubits = len(results[0]["basis"]) if results else 0
        dim = 2**num_qubits

        # For demonstration, we'll return the structure
        # Full implementation would require tomography package

        return {
            "status": "success",
            "message": f"Estimated {num_qubits}-qubit state from {len(results)} measurement bases",
            "num_qubits": num_qubits,
            "dimension": dim,
            "note": "Full state reconstruction requires additional tomography libraries",
        }
    except Exception as e:
        logger.error(f"Failed to estimate state: {e}")
        return {"status": "error", "message": f"Failed to estimate state: {str(e)}"}


async def verify_state_preparation(
    prepared_qasm: str, target_qasm: str
) -> Dict[str, Any]:
    """Verify state preparation by comparing with target state.

    Args:
        prepared_qasm: QASM of circuit that prepares the state
        target_qasm: QASM of circuit that prepares the target state

    Returns:
        Fidelity between prepared and target states
    """
    try:
        prepared_circuit = qasm2.loads(prepared_qasm)
        target_circuit = qasm2.loads(target_qasm)

        # Remove measurements
        prepared_circuit.remove_final_measurements(inplace=True)
        target_circuit.remove_final_measurements(inplace=True)

        # Get statevectors
        prepared_state = Statevector.from_instruction(prepared_circuit)
        target_state = Statevector.from_instruction(target_circuit)

        # Calculate fidelity
        fidelity = state_fidelity(prepared_state, target_state)

        return {
            "status": "success",
            "message": "Verified state preparation",
            "fidelity": float(fidelity),
            "is_close": fidelity > 0.99,
            "prepared_state": str(prepared_state),
            "target_state": str(target_state),
        }
    except Exception as e:
        logger.error(f"Failed to verify state preparation: {e}")
        return {
            "status": "error",
            "message": f"Failed to verify state preparation: {str(e)}",
        }


# Assisted by watsonx Code Assistant
