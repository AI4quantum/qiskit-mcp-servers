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

"""Qiskit Primitives (Sampler and Estimator) for the MCP server.

The Primitives are the recommended way to execute quantum circuits in Qiskit 1.0+.
- Sampler: Samples from quantum circuits to get quasi-probability distributions
- Estimator: Calculates expectation values of observables
"""

import logging
from typing import Any, Dict, List, Optional

from qiskit import qasm2
from qiskit.primitives import StatevectorSampler, StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

logger = logging.getLogger(__name__)


async def sample_circuit(
    circuit_qasm: str, shots: int = 1024, seed: Optional[int] = None
) -> Dict[str, Any]:
    """Sample from a quantum circuit using the Sampler primitive.

    Args:
        circuit_qasm: QASM representation of the circuit
        shots: Number of shots to sample (default: 1024)
        seed: Random seed for reproducibility

    Returns:
        Sampling results with counts and probabilities
    """
    try:
        circuit = qasm2.loads(circuit_qasm)

        # Check if circuit has unbound parameters
        if circuit.num_parameters > 0:
            return {
                "status": "error",
                "message": (
                    f"Circuit has {circuit.num_parameters} unbound parameters. "
                    "Please bind all parameters to concrete values before sampling. "
                    "Use circuit.assign_parameters() or provide parameter values."
                ),
            }

        # Ensure circuit has measurements
        if circuit.num_clbits == 0:
            return {
                "status": "error",
                "message": "Circuit must have classical bits for measurement. Add measurements to your circuit.",
            }

        # Create Sampler primitive
        sampler = StatevectorSampler(seed=seed)

        # Run sampler
        job = sampler.run([circuit], shots=shots)
        result = job.result()

        # Extract results
        pub_result = result[0]
        # Get counts from first classical register (DataBin API changed in Qiskit 1.0+)
        data_fields = list(pub_result.data.keys())
        if data_fields:
            counts = pub_result.data[data_fields[0]].get_counts()
        else:
            counts = {}

        return {
            "status": "success",
            "shots": shots,
            "counts": counts,
            "num_qubits": circuit.num_qubits,
            "num_clbits": circuit.num_clbits,
            "metadata": {
                "circuit_name": circuit.name,
            },
        }
    except Exception as e:
        logger.error(f"Failed to sample circuit: {e}")
        return {"status": "error", "message": f"Failed to sample circuit: {str(e)}"}


async def sample_multiple_circuits(
    circuit_qasms: List[str], shots: int = 1024, seed: Optional[int] = None
) -> Dict[str, Any]:
    """Sample from multiple quantum circuits in a single batch.

    Args:
        circuit_qasms: List of QASM representations
        shots: Number of shots per circuit
        seed: Random seed for reproducibility

    Returns:
        Results for all circuits
    """
    try:
        circuits = [qasm2.loads(qasm) for qasm in circuit_qasms]

        # Validate all circuits have measurements
        for i, circuit in enumerate(circuits):
            if circuit.num_clbits == 0:
                return {
                    "status": "error",
                    "message": f"Circuit {i} must have classical bits for measurement.",
                }

        # Create Sampler primitive
        sampler = StatevectorSampler(seed=seed)

        # Run sampler on all circuits
        job = sampler.run(circuits, shots=shots)
        result = job.result()

        # Extract results for each circuit
        results = []
        for i, pub_result in enumerate(result):
            # Get counts from first classical register (DataBin API changed in Qiskit 1.0+)
            data_fields = list(pub_result.data.keys())
            counts = pub_result.data[data_fields[0]].get_counts() if data_fields else {}
            results.append(
                {
                    "circuit_index": i,
                    "counts": counts,
                    "num_qubits": circuits[i].num_qubits,
                    "circuit_name": circuits[i].name,
                }
            )

        return {
            "status": "success",
            "num_circuits": len(circuits),
            "shots_per_circuit": shots,
            "results": results,
        }
    except Exception as e:
        logger.error(f"Failed to sample multiple circuits: {e}")
        return {
            "status": "error",
            "message": f"Failed to sample multiple circuits: {str(e)}",
        }


async def estimate_expectation_values(
    circuit_qasm: str, observables: str, coeffs: str = "", precision: float = 0.0
) -> Dict[str, Any]:
    """Estimate expectation values using the Estimator primitive.

    Args:
        circuit_qasm: QASM representation of the circuit
        observables: Comma-separated Pauli strings (e.g., "ZZ,XX,YY")
        coeffs: Optional comma-separated coefficients (default: all 1.0)
        precision: Target precision for expectation value (default: 0.0 = exact)

    Returns:
        Expectation values for the observables
    """
    try:
        circuit = qasm2.loads(circuit_qasm)

        # Check if circuit has unbound parameters
        if circuit.num_parameters > 0:
            return {
                "status": "error",
                "message": (
                    f"Circuit has {circuit.num_parameters} unbound parameters. "
                    "Please bind all parameters to concrete values before estimating expectation values. "
                    "Use circuit.assign_parameters() or provide parameter values."
                ),
            }

        # Remove measurements for statevector estimation
        circuit_copy = circuit.copy()
        circuit_copy.remove_final_measurements(inplace=True)

        # Parse observables
        pauli_strings = [p.strip() for p in observables.split(",")]

        # Parse coefficients if provided
        if coeffs:
            coefficients = [float(c.strip()) for c in coeffs.split(",")]
            if len(coefficients) != len(pauli_strings):
                return {
                    "status": "error",
                    "message": f"Number of coefficients ({len(coefficients)}) must match number of observables ({len(pauli_strings)})",
                }
        else:
            coefficients = [1.0] * len(pauli_strings)

        # Create SparsePauliOp observable
        observable = SparsePauliOp(pauli_strings, coeffs=coefficients)

        # Create Estimator primitive
        estimator = StatevectorEstimator()

        # Run estimator
        job = estimator.run([(circuit_copy, observable)], precision=precision)
        result = job.result()

        # Extract results
        pub_result = result[0]
        # evs and stds are now scalars (0-d arrays) in Qiskit 1.0+
        expectation_value = float(pub_result.data.evs)
        std_dev = (
            float(pub_result.data.stds) if hasattr(pub_result.data, "stds") else 0.0
        )

        return {
            "status": "success",
            "expectation_value": float(expectation_value),
            "std_dev": float(std_dev),
            "observable": str(observable),
            "num_qubits": circuit_copy.num_qubits,
            "metadata": {
                "precision": precision,
                "circuit_name": circuit_copy.name,
            },
        }
    except Exception as e:
        logger.error(f"Failed to estimate expectation values: {e}")
        return {
            "status": "error",
            "message": f"Failed to estimate expectation values: {str(e)}",
        }


async def estimate_multiple_observables(
    circuit_qasm: str,
    observables_list: List[str],
    coeffs_list: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Estimate expectation values for multiple observables.

    Args:
        circuit_qasm: QASM representation of the circuit
        observables_list: List of comma-separated Pauli strings
        coeffs_list: Optional list of comma-separated coefficients

    Returns:
        Expectation values for all observables
    """
    try:
        circuit = qasm2.loads(circuit_qasm)

        # Check if circuit has unbound parameters
        if circuit.num_parameters > 0:
            return {
                "status": "error",
                "message": (
                    f"Circuit has {circuit.num_parameters} unbound parameters. "
                    "Please bind all parameters to concrete values before estimating expectation values. "
                    "Use circuit.assign_parameters() or provide parameter values."
                ),
            }

        circuit.remove_final_measurements(inplace=True)

        # Parse all observables
        observables = []
        for i, obs_str in enumerate(observables_list):
            pauli_strings = [p.strip() for p in obs_str.split(",")]

            if coeffs_list and i < len(coeffs_list):
                coefficients = [float(c.strip()) for c in coeffs_list[i].split(",")]
            else:
                coefficients = [1.0] * len(pauli_strings)

            observable = SparsePauliOp(pauli_strings, coeffs=coefficients)
            observables.append(observable)

        # Create Estimator primitive
        estimator = StatevectorEstimator()

        # Prepare input as list of (circuit, observable) tuples
        pubs = [(circuit, obs) for obs in observables]

        # Run estimator
        job = estimator.run(pubs)
        result = job.result()

        # Extract results
        results = []
        for i, pub_result in enumerate(result):
            expectation_value = float(pub_result.data.evs)
            std_dev = (
                float(pub_result.data.stds) if hasattr(pub_result.data, "stds") else 0.0
            )

            results.append(
                {
                    "observable_index": i,
                    "observable": str(observables[i]),
                    "expectation_value": float(expectation_value),
                    "std_dev": float(std_dev),
                }
            )

        return {
            "status": "success",
            "num_observables": len(observables),
            "results": results,
        }
    except Exception as e:
        logger.error(f"Failed to estimate multiple observables: {e}")
        return {
            "status": "error",
            "message": f"Failed to estimate multiple observables: {str(e)}",
        }


async def run_variational_estimation(
    circuit_qasms: List[str], observables: str, coeffs: str = ""
) -> Dict[str, Any]:
    """Run estimator on multiple circuit variations (e.g., for VQE).

    Args:
        circuit_qasms: List of QASM representations (circuit variations)
        observables: Comma-separated Pauli strings for the Hamiltonian
        coeffs: Optional comma-separated coefficients

    Returns:
        Expectation values for all circuit variations
    """
    try:
        circuits = [qasm2.loads(qasm) for qasm in circuit_qasms]

        # Remove measurements
        for circuit in circuits:
            circuit.remove_final_measurements(inplace=True)

        # Parse observable (single Hamiltonian for all circuits)
        pauli_strings = [p.strip() for p in observables.split(",")]
        if coeffs:
            coefficients = [float(c.strip()) for c in coeffs.split(",")]
        else:
            coefficients = [1.0] * len(pauli_strings)

        observable = SparsePauliOp(pauli_strings, coeffs=coefficients)

        # Create Estimator primitive
        estimator = StatevectorEstimator()

        # Prepare input: same observable for all circuits
        pubs = [(circuit, observable) for circuit in circuits]

        # Run estimator
        job = estimator.run(pubs)
        result = job.result()

        # Extract results
        results = []
        for i, pub_result in enumerate(result):
            expectation_value = float(pub_result.data.evs)
            std_dev = (
                float(pub_result.data.stds) if hasattr(pub_result.data, "stds") else 0.0
            )

            results.append(
                {
                    "circuit_index": i,
                    "circuit_name": circuits[i].name,
                    "expectation_value": float(expectation_value),
                    "std_dev": float(std_dev),
                }
            )

        return {
            "status": "success",
            "num_circuits": len(circuits),
            "observable": str(observable),
            "results": results,
            "minimum_energy": min(r["expectation_value"] for r in results),
            "minimum_index": min(
                range(len(results)), key=lambda i: results[i]["expectation_value"]
            ),
        }
    except Exception as e:
        logger.error(f"Failed to run variational estimation: {e}")
        return {
            "status": "error",
            "message": f"Failed to run variational estimation: {str(e)}",
        }


async def sample_with_parameter_sweep(
    base_circuit_qasm: str, parameter_values: List[float], shots: int = 1024
) -> Dict[str, Any]:
    """Sample a parameterized circuit with different parameter values.

    Note: This is a simplified version. For full parameter binding,
    use Qiskit's Parameter class.

    Args:
        base_circuit_qasm: Base QASM representation
        parameter_values: List of parameter values to sweep
        shots: Number of shots per parameter value

    Returns:
        Sampling results for each parameter value
    """
    try:
        # For simplicity, we'll return guidance on how to use parameters
        return {
            "status": "info",
            "message": "Parameter sweeps require ParameterVector support. "
            "Use create_quantum_circuit and add_gates_to_circuit with "
            "different angle values to create circuit variations, "
            "then use sample_multiple_circuits.",
            "example": "Create circuits with different theta values and pass to sample_multiple_circuits",
        }
    except Exception as e:
        logger.error(f"Failed parameter sweep: {e}")
        return {"status": "error", "message": f"Failed parameter sweep: {str(e)}"}


# Assisted by watsonx Code Assistant
