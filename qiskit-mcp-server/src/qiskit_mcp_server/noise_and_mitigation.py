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

"""Noise models and error mitigation operations."""

import logging
from typing import Any, Dict

from qiskit import qasm2
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    thermal_relaxation_error,
    amplitude_damping_error,
    phase_damping_error,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Noise Model Creation
# ============================================================================


async def create_depolarizing_noise_model(
    single_qubit_error: float = 0.001,
    two_qubit_error: float = 0.01,
) -> Dict[str, Any]:
    """Create a depolarizing noise model.

    Args:
        single_qubit_error: Error probability for single-qubit gates
        two_qubit_error: Error probability for two-qubit gates

    Returns:
        Noise model information
    """
    try:
        noise_model = NoiseModel()

        # Single-qubit gate errors
        single_qubit_gates = [
            "u1",
            "u2",
            "u3",
            "rx",
            "ry",
            "rz",
            "h",
            "x",
            "y",
            "z",
            "s",
            "t",
        ]
        error_1q = depolarizing_error(single_qubit_error, 1)
        for gate in single_qubit_gates:
            noise_model.add_all_qubit_quantum_error(error_1q, gate)

        # Two-qubit gate errors
        two_qubit_gates = ["cx", "cz", "swap"]
        error_2q = depolarizing_error(two_qubit_error, 2)
        for gate in two_qubit_gates:
            noise_model.add_all_qubit_quantum_error(error_2q, gate)

        return {
            "status": "success",
            "message": "Created depolarizing noise model",
            "single_qubit_error": single_qubit_error,
            "two_qubit_error": two_qubit_error,
            "noise_model_summary": str(noise_model),
        }
    except Exception as e:
        logger.error(f"Failed to create depolarizing noise model: {e}")
        return {"status": "error", "message": f"Failed to create noise model: {str(e)}"}


async def create_thermal_noise_model(
    t1: float = 50000.0,
    t2: float = 70000.0,
    gate_time_1q: float = 50.0,
    gate_time_2q: float = 300.0,
) -> Dict[str, Any]:
    """Create a thermal relaxation noise model.

    Args:
        t1: T1 relaxation time in nanoseconds (default: 50µs)
        t2: T2 dephasing time in nanoseconds (default: 70µs)
        gate_time_1q: Single-qubit gate time in nanoseconds (default: 50ns)
        gate_time_2q: Two-qubit gate time in nanoseconds (default: 300ns)

    Returns:
        Noise model information
    """
    try:
        noise_model = NoiseModel()

        # Single-qubit thermal errors
        single_qubit_gates = ["u1", "u2", "u3", "rx", "ry", "rz", "h", "x", "y", "z"]
        error_1q = thermal_relaxation_error(t1, t2, gate_time_1q)
        for gate in single_qubit_gates:
            noise_model.add_all_qubit_quantum_error(error_1q, gate)

        # Two-qubit thermal errors
        two_qubit_gates = ["cx", "cz"]
        error_2q = thermal_relaxation_error(t1, t2, gate_time_2q).tensor(
            thermal_relaxation_error(t1, t2, gate_time_2q)
        )
        for gate in two_qubit_gates:
            noise_model.add_all_qubit_quantum_error(error_2q, gate)

        return {
            "status": "success",
            "message": "Created thermal relaxation noise model",
            "t1_ns": t1,
            "t2_ns": t2,
            "gate_time_1q_ns": gate_time_1q,
            "gate_time_2q_ns": gate_time_2q,
            "noise_model_summary": str(noise_model),
        }
    except Exception as e:
        logger.error(f"Failed to create thermal noise model: {e}")
        return {"status": "error", "message": f"Failed to create noise model: {str(e)}"}


async def create_amplitude_damping_noise_model(
    damping_param: float = 0.01,
) -> Dict[str, Any]:
    """Create an amplitude damping noise model.

    Args:
        damping_param: Amplitude damping parameter (probability)

    Returns:
        Noise model information
    """
    try:
        noise_model = NoiseModel()

        # Add amplitude damping to all gates
        gates = ["u1", "u2", "u3", "rx", "ry", "rz", "h", "x", "y", "z", "cx"]
        error = amplitude_damping_error(damping_param)

        for gate in gates:
            if gate in ["cx", "cz"]:
                # Two-qubit gates
                error_2q = error.tensor(error)
                noise_model.add_all_qubit_quantum_error(error_2q, gate)
            else:
                # Single-qubit gates
                noise_model.add_all_qubit_quantum_error(error, gate)

        return {
            "status": "success",
            "message": "Created amplitude damping noise model",
            "damping_parameter": damping_param,
            "noise_model_summary": str(noise_model),
        }
    except Exception as e:
        logger.error(f"Failed to create amplitude damping noise model: {e}")
        return {"status": "error", "message": f"Failed to create noise model: {str(e)}"}


async def create_phase_damping_noise_model(
    damping_param: float = 0.01,
) -> Dict[str, Any]:
    """Create a phase damping noise model.

    Args:
        damping_param: Phase damping parameter (probability)

    Returns:
        Noise model information
    """
    try:
        noise_model = NoiseModel()

        # Add phase damping to all gates
        gates = ["u1", "u2", "u3", "rx", "ry", "rz", "h", "x", "y", "z", "cx"]
        error = phase_damping_error(damping_param)

        for gate in gates:
            if gate in ["cx", "cz"]:
                # Two-qubit gates
                error_2q = error.tensor(error)
                noise_model.add_all_qubit_quantum_error(error_2q, gate)
            else:
                # Single-qubit gates
                noise_model.add_all_qubit_quantum_error(error, gate)

        return {
            "status": "success",
            "message": "Created phase damping noise model",
            "damping_parameter": damping_param,
            "noise_model_summary": str(noise_model),
        }
    except Exception as e:
        logger.error(f"Failed to create phase damping noise model: {e}")
        return {"status": "error", "message": f"Failed to create noise model: {str(e)}"}


# ============================================================================
# Noisy Simulation
# ============================================================================


async def simulate_with_noise(
    circuit_qasm: str,
    noise_type: str = "depolarizing",
    single_qubit_error: float = 0.001,
    two_qubit_error: float = 0.01,
    shots: int = 1024,
) -> Dict[str, Any]:
    """Simulate circuit with noise model.

    Args:
        circuit_qasm: QASM representation of circuit
        noise_type: Type of noise (depolarizing, thermal, amplitude_damping, phase_damping)
        single_qubit_error: Single-qubit error rate
        two_qubit_error: Two-qubit error rate
        shots: Number of shots

    Returns:
        Noisy simulation results
    """
    try:
        circuit = qasm2.loads(circuit_qasm)

        # Add measurements if not present
        if circuit.num_clbits == 0:
            circuit.measure_all()

        # Create noise model based on type
        if noise_type == "depolarizing":
            noise_model = NoiseModel()
            error_1q = depolarizing_error(single_qubit_error, 1)
            error_2q = depolarizing_error(two_qubit_error, 2)

            for instr in circuit.data:
                gate_name = instr.operation.name
                num_qubits = len(instr.qubits)
                if num_qubits == 1:
                    noise_model.add_all_qubit_quantum_error(error_1q, gate_name)
                elif num_qubits == 2:
                    noise_model.add_all_qubit_quantum_error(error_2q, gate_name)

        elif noise_type == "thermal":
            noise_model = NoiseModel()
            t1 = 50000.0  # 50 µs
            t2 = 70000.0  # 70 µs
            error_1q = thermal_relaxation_error(t1, t2, 50.0)  # 50ns gate time
            error_2q = thermal_relaxation_error(t1, t2, 300.0).tensor(
                thermal_relaxation_error(t1, t2, 300.0)
            )

            for instr in circuit.data:
                gate_name = instr.operation.name
                num_qubits = len(instr.qubits)
                if num_qubits == 1:
                    noise_model.add_all_qubit_quantum_error(error_1q, gate_name)
                elif num_qubits == 2:
                    noise_model.add_all_qubit_quantum_error(error_2q, gate_name)

        elif noise_type == "amplitude_damping":
            noise_model = NoiseModel()
            error = amplitude_damping_error(single_qubit_error)
            for instr in circuit.data:
                gate_name = instr.operation.name
                num_qubits = len(instr.qubits)
                if num_qubits == 1:
                    noise_model.add_all_qubit_quantum_error(error, gate_name)
                elif num_qubits == 2:
                    noise_model.add_all_qubit_quantum_error(
                        error.tensor(error), gate_name
                    )

        elif noise_type == "phase_damping":
            noise_model = NoiseModel()
            error = phase_damping_error(single_qubit_error)
            for instr in circuit.data:
                gate_name = instr.operation.name
                num_qubits = len(instr.qubits)
                if num_qubits == 1:
                    noise_model.add_all_qubit_quantum_error(error, gate_name)
                elif num_qubits == 2:
                    noise_model.add_all_qubit_quantum_error(
                        error.tensor(error), gate_name
                    )
        else:
            return {
                "status": "error",
                "message": f"Unknown noise type: {noise_type}. Use: depolarizing, thermal, amplitude_damping, or phase_damping",
            }

        # Run noisy simulation
        simulator = AerSimulator(noise_model=noise_model)
        job = simulator.run(circuit, shots=shots)
        result = job.result()
        counts = result.get_counts()

        return {
            "status": "success",
            "message": f"Simulated with {noise_type} noise",
            "noise_type": noise_type,
            "shots": shots,
            "counts": dict(counts),
            "single_qubit_error": single_qubit_error,
            "two_qubit_error": two_qubit_error,
        }
    except Exception as e:
        logger.error(f"Failed to simulate with noise: {e}")
        return {
            "status": "error",
            "message": f"Failed to simulate with noise: {str(e)}",
        }


async def compare_ideal_vs_noisy(
    circuit_qasm: str,
    noise_type: str = "depolarizing",
    error_rate: float = 0.01,
    shots: int = 1024,
) -> Dict[str, Any]:
    """Compare ideal and noisy simulation results.

    Args:
        circuit_qasm: QASM representation of circuit
        noise_type: Type of noise model
        error_rate: Error rate for noise
        shots: Number of shots

    Returns:
        Comparison of ideal vs noisy results
    """
    try:
        circuit = qasm2.loads(circuit_qasm)

        # Add measurements if not present
        if circuit.num_clbits == 0:
            circuit.measure_all()

        # Ideal simulation
        ideal_sim = AerSimulator()
        ideal_job = ideal_sim.run(circuit, shots=shots)
        ideal_result = ideal_job.result()
        ideal_counts = dict(ideal_result.get_counts())

        # Noisy simulation
        noisy_result = await simulate_with_noise(
            circuit_qasm,
            noise_type=noise_type,
            single_qubit_error=error_rate,
            two_qubit_error=error_rate * 10,
            shots=shots,
        )

        if noisy_result["status"] == "error":
            return noisy_result

        noisy_counts = noisy_result["counts"]

        # Calculate total variation distance
        all_states = set(ideal_counts.keys()) | set(noisy_counts.keys())
        tvd = 0.0
        for state in all_states:
            p_ideal = ideal_counts.get(state, 0) / shots
            p_noisy = noisy_counts.get(state, 0) / shots
            tvd += abs(p_ideal - p_noisy)
        tvd /= 2.0

        return {
            "status": "success",
            "message": "Compared ideal vs noisy simulation",
            "ideal_counts": ideal_counts,
            "noisy_counts": noisy_counts,
            "total_variation_distance": tvd,
            "noise_type": noise_type,
            "error_rate": error_rate,
            "shots": shots,
        }
    except Exception as e:
        logger.error(f"Failed to compare simulations: {e}")
        return {
            "status": "error",
            "message": f"Failed to compare simulations: {str(e)}",
        }


# Assisted by watsonx Code Assistant
