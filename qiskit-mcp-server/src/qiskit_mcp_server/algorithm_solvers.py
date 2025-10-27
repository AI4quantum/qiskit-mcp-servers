# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
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

"""Quantum algorithm solvers including VQE, QAOA, and optimization utilities."""

import logging
from typing import Any, Dict, Optional
import json
import numpy as np

from qiskit import QuantumCircuit, qasm2
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

logger = logging.getLogger(__name__)


# ============================================================================
# VQE (Variational Quantum Eigensolver)
# ============================================================================


async def run_vqe(
    hamiltonian: str,
    ansatz_qasm: str,
    initial_point: Optional[str] = None,
    optimizer: str = "COBYLA",
    max_iterations: int = 100,
) -> Dict[str, Any]:
    """Run Variational Quantum Eigensolver (VQE) algorithm.

    Args:
        hamiltonian: Hamiltonian as Pauli string (e.g., "II + 0.5*ZZ + 0.3*XX")
        ansatz_qasm: QASM representation of ansatz circuit (must have parameters)
        initial_point: JSON list of initial parameter values (optional)
        optimizer: Optimizer name (COBYLA, SLSQP, SPSA, etc.)
        max_iterations: Maximum optimization iterations

    Returns:
        VQE results including ground state energy
    """
    try:
        from scipy.optimize import minimize

        # Parse Hamiltonian
        hamiltonian_op = SparsePauliOp.from_list(
            [
                (pauli_term.strip(), float(coeff))
                for term in hamiltonian.split("+")
                for pauli_term, coeff in [
                    term.strip().split("*") if "*" in term else (term.strip(), 1.0)
                ]
            ]
        )

        # Load ansatz
        ansatz = qasm2.loads(ansatz_qasm)
        num_params = ansatz.num_parameters

        if num_params == 0:
            return {
                "status": "error",
                "message": "Ansatz circuit must have parameters for VQE",
            }

        # Parse initial point
        if initial_point:
            initial_params = json.loads(initial_point)
        else:
            initial_params = np.random.uniform(0, 2 * np.pi, num_params).tolist()

        # Create estimator
        estimator = StatevectorEstimator()

        # Define cost function
        iteration_count = [0]
        energy_history = []

        def cost_function(params):
            iteration_count[0] += 1
            # Bind parameters
            bound_circuit = ansatz.assign_parameters(params)

            # Estimate expectation value
            job = estimator.run([(bound_circuit, hamiltonian_op)])
            result = job.result()
            energy = result[0].data.evs

            energy_history.append(float(energy))

            if iteration_count[0] % 10 == 0:
                logger.info(
                    f"VQE iteration {iteration_count[0]}: energy = {energy:.6f}"
                )

            return float(energy)

        # Run optimization
        logger.info(f"Starting VQE optimization with {optimizer}...")

        if optimizer.upper() == "COBYLA":
            result = minimize(
                cost_function,
                initial_params,
                method="COBYLA",
                options={"maxiter": max_iterations, "rhobeg": 0.1},
            )
        elif optimizer.upper() == "SLSQP":
            result = minimize(
                cost_function,
                initial_params,
                method="SLSQP",
                options={"maxiter": max_iterations},
            )
        elif optimizer.upper() == "NELDER-MEAD":
            result = minimize(
                cost_function,
                initial_params,
                method="Nelder-Mead",
                options={"maxiter": max_iterations},
            )
        else:
            return {
                "status": "error",
                "message": f"Optimizer '{optimizer}' not supported. Use COBYLA, SLSQP, or NELDER-MEAD",
            }

        return {
            "status": "success",
            "message": f"VQE completed in {iteration_count[0]} iterations",
            "algorithm": "VQE",
            "ground_state_energy": float(result.fun),
            "optimal_parameters": result.x.tolist(),
            "num_parameters": num_params,
            "optimizer": optimizer,
            "iterations": iteration_count[0],
            "energy_history": energy_history,
            "convergence": {
                "success": result.success,
                "final_energy": float(result.fun),
                "energy_variance": float(np.var(energy_history[-10:]))
                if len(energy_history) >= 10
                else None,
            },
            "hamiltonian": hamiltonian,
        }

    except Exception as e:
        logger.error(f"VQE failed: {e}")
        return {"status": "error", "message": f"VQE failed: {str(e)}"}


async def evaluate_expectation_value(
    circuit_qasm: str,
    observable: str,
) -> Dict[str, Any]:
    """Evaluate expectation value <ψ|O|ψ> for a circuit and observable.

    Args:
        circuit_qasm: QASM representation of circuit
        observable: Observable as Pauli string (e.g., "ZZ", "0.5*XX + 0.3*YY")

    Returns:
        Expectation value result
    """
    try:
        circuit = qasm2.loads(circuit_qasm)

        # Parse observable
        observable_op = SparsePauliOp.from_list(
            [
                (pauli_term.strip(), float(coeff))
                for term in observable.split("+")
                for pauli_term, coeff in [
                    term.strip().split("*") if "*" in term else (term.strip(), 1.0)
                ]
            ]
        )

        # Estimate expectation value
        estimator = StatevectorEstimator()
        job = estimator.run([(circuit, observable_op)])
        result = job.result()
        expectation = result[0].data.evs

        return {
            "status": "success",
            "expectation_value": float(expectation),
            "observable": observable,
            "num_qubits": circuit.num_qubits,
            "circuit_depth": circuit.depth(),
        }

    except Exception as e:
        logger.error(f"Failed to evaluate expectation value: {e}")
        return {
            "status": "error",
            "message": f"Failed to evaluate expectation: {str(e)}",
        }


# ============================================================================
# QAOA (Quantum Approximate Optimization Algorithm)
# ============================================================================


async def run_qaoa(
    cost_hamiltonian: str,
    num_qubits: int,
    num_layers: int = 1,
    optimizer: str = "COBYLA",
    max_iterations: int = 100,
    initial_point: Optional[str] = None,
) -> Dict[str, Any]:
    """Run Quantum Approximate Optimization Algorithm (QAOA).

    Args:
        cost_hamiltonian: Cost Hamiltonian as Pauli string (e.g., "ZIZI + IZZI")
        num_qubits: Number of qubits
        num_layers: Number of QAOA layers (p parameter)
        optimizer: Classical optimizer (COBYLA, SLSQP, etc.)
        max_iterations: Maximum optimization iterations
        initial_point: JSON list of initial parameter values

    Returns:
        QAOA results including optimal bitstring
    """
    try:
        from scipy.optimize import minimize

        # Parse cost Hamiltonian
        cost_op = SparsePauliOp.from_list(
            [
                (pauli_term.strip(), float(coeff))
                for term in cost_hamiltonian.split("+")
                for pauli_term, coeff in [
                    term.strip().split("*") if "*" in term else (term.strip(), 1.0)
                ]
            ]
        )

        # Create QAOA ansatz
        def create_qaoa_circuit(params):
            """Create QAOA circuit with given parameters."""
            qc = QuantumCircuit(num_qubits)

            # Initial superposition
            qc.h(range(num_qubits))

            # QAOA layers
            for layer in range(num_layers):
                gamma = params[layer]
                beta = params[num_layers + layer]

                # Cost unitary (exp(-i*gamma*Cost))
                # Apply evolution under cost Hamiltonian
                for pauli, coeff in cost_op.to_list():
                    if pauli == "I" * num_qubits:
                        continue
                    # Implement evolution under each Pauli term
                    for i, p in enumerate(pauli):
                        if p == "Z":
                            qc.rz(2 * gamma * coeff, i)
                        elif p == "X":
                            qc.rx(2 * gamma * coeff, i)
                        elif p == "Y":
                            qc.ry(2 * gamma * coeff, i)

                # Mixer unitary (exp(-i*beta*Mixer))
                for i in range(num_qubits):
                    qc.rx(2 * beta, i)

            return qc

        # Initialize parameters
        num_params = 2 * num_layers
        if initial_point:
            initial_params = json.loads(initial_point)
        else:
            initial_params = np.random.uniform(0, np.pi, num_params).tolist()

        # Estimator
        estimator = StatevectorEstimator()

        # Cost function
        iteration_count = [0]
        energy_history = []

        def cost_function(params):
            iteration_count[0] += 1

            circuit = create_qaoa_circuit(params)

            # Estimate cost
            job = estimator.run([(circuit, cost_op)])
            result = job.result()
            cost = result[0].data.evs

            energy_history.append(float(cost))

            if iteration_count[0] % 10 == 0:
                logger.info(f"QAOA iteration {iteration_count[0]}: cost = {cost:.6f}")

            return float(cost)

        # Optimize
        logger.info(f"Starting QAOA optimization with {optimizer}...")

        if optimizer.upper() == "COBYLA":
            result = minimize(
                cost_function,
                initial_params,
                method="COBYLA",
                options={"maxiter": max_iterations},
            )
        elif optimizer.upper() == "SLSQP":
            result = minimize(
                cost_function,
                initial_params,
                method="SLSQP",
                options={"maxiter": max_iterations},
            )
        else:
            return {
                "status": "error",
                "message": f"Optimizer '{optimizer}' not supported",
            }

        # Sample final circuit to get optimal bitstring
        from qiskit.primitives import StatevectorSampler

        optimal_circuit = create_qaoa_circuit(result.x)
        optimal_circuit.measure_all()

        sampler = StatevectorSampler()
        job = sampler.run([optimal_circuit], shots=1024)
        sampling_result = job.result()
        counts = sampling_result[0].data.meas.get_counts()

        # Get most likely bitstring
        optimal_bitstring = max(counts.items(), key=lambda x: x[1])[0]

        return {
            "status": "success",
            "message": f"QAOA completed in {iteration_count[0]} iterations",
            "algorithm": "QAOA",
            "optimal_cost": float(result.fun),
            "optimal_parameters": result.x.tolist(),
            "optimal_bitstring": optimal_bitstring,
            "num_layers": num_layers,
            "num_qubits": num_qubits,
            "optimizer": optimizer,
            "iterations": iteration_count[0],
            "cost_history": energy_history,
            "sample_distribution": dict(counts),
            "convergence": {
                "success": result.success,
                "final_cost": float(result.fun),
            },
        }

    except Exception as e:
        logger.error(f"QAOA failed: {e}")
        return {"status": "error", "message": f"QAOA failed: {str(e)}"}


async def optimize_parameters(
    circuit_qasm: str,
    cost_function_type: str,
    observable: Optional[str] = None,
    optimizer: str = "COBYLA",
    max_iterations: int = 100,
    initial_point: Optional[str] = None,
) -> Dict[str, Any]:
    """Generic parameter optimization for variational circuits.

    Args:
        circuit_qasm: QASM representation of parametric circuit
        cost_function_type: Type of cost function (expectation, sampling)
        observable: Observable for expectation-based cost (Pauli string)
        optimizer: Classical optimizer
        max_iterations: Maximum iterations
        initial_point: Initial parameter values

    Returns:
        Optimization results
    """
    try:
        from scipy.optimize import minimize

        circuit = qasm2.loads(circuit_qasm)
        num_params = circuit.num_parameters

        if num_params == 0:
            return {
                "status": "error",
                "message": "Circuit must have parameters for optimization",
            }

        # Initialize parameters
        if initial_point:
            initial_params = json.loads(initial_point)
        else:
            initial_params = np.random.uniform(0, 2 * np.pi, num_params).tolist()

        iteration_count = [0]
        cost_history = []

        if cost_function_type == "expectation":
            if not observable:
                return {
                    "status": "error",
                    "message": "Observable required for expectation-based optimization",
                }

            observable_op = SparsePauliOp.from_list(
                [
                    (pauli_term.strip(), float(coeff))
                    for term in observable.split("+")
                    for pauli_term, coeff in [
                        term.strip().split("*") if "*" in term else (term.strip(), 1.0)
                    ]
                ]
            )

            estimator = StatevectorEstimator()

            def cost_fn(params):
                iteration_count[0] += 1
                bound_circuit = circuit.assign_parameters(params)
                job = estimator.run([(bound_circuit, observable_op)])
                result = job.result()
                cost = result[0].data.evs
                cost_history.append(float(cost))
                return float(cost)

        elif cost_function_type == "sampling":
            from qiskit.primitives import StatevectorSampler

            sampler = StatevectorSampler()

            def cost_fn(params):
                iteration_count[0] += 1
                bound_circuit = circuit.assign_parameters(params)
                bound_circuit.measure_all()

                job = sampler.run([bound_circuit], shots=1024)
                result = job.result()
                counts = result[0].data.meas.get_counts()

                # Cost = 1 - (probability of most likely outcome)
                max_count = max(counts.values())
                cost = 1.0 - (max_count / 1024.0)
                cost_history.append(float(cost))
                return float(cost)
        else:
            return {
                "status": "error",
                "message": f"Cost function type '{cost_function_type}' not supported. Use 'expectation' or 'sampling'",
            }

        # Optimize
        result = minimize(
            cost_fn,
            initial_params,
            method=optimizer,
            options={"maxiter": max_iterations},
        )

        return {
            "status": "success",
            "optimal_parameters": result.x.tolist(),
            "optimal_cost": float(result.fun),
            "num_parameters": num_params,
            "optimizer": optimizer,
            "iterations": iteration_count[0],
            "cost_history": cost_history,
            "convergence_success": result.success,
        }

    except Exception as e:
        logger.error(f"Parameter optimization failed: {e}")
        return {"status": "error", "message": f"Optimization failed: {str(e)}"}


# Assisted by watsonx Code Assistant
