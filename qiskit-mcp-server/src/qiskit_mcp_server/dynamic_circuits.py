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

"""Dynamic circuits with mid-circuit measurement and classical control."""

import logging
from typing import Any, Dict
import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, qasm2

logger = logging.getLogger(__name__)


# ============================================================================
# Mid-Circuit Measurement
# ============================================================================


async def add_mid_circuit_measurement(
    circuit_qasm: str,
    measure_qubit: int,
    classical_bit: int,
) -> Dict[str, Any]:
    """Add mid-circuit measurement to a circuit.

    Enables measurement before the end of the circuit for dynamic operations.

    Args:
        circuit_qasm: QASM representation of circuit
        measure_qubit: Qubit index to measure
        classical_bit: Classical bit to store measurement result

    Returns:
        Circuit with mid-circuit measurement added
    """
    try:
        circuit = qasm2.loads(circuit_qasm)

        # Ensure sufficient classical bits
        if classical_bit >= circuit.num_clbits:
            # Add more classical bits if needed
            num_new_bits = classical_bit - circuit.num_clbits + 1
            circuit.add_bits(
                [
                    ClassicalRegister(num_new_bits, f"c_mid_{i}")
                    for i in range(num_new_bits)
                ]
            )

        # Add mid-circuit measurement
        circuit.measure(measure_qubit, classical_bit)

        return {
            "status": "success",
            "message": f"Added mid-circuit measurement: qubit {measure_qubit} -> cbit {classical_bit}",
            "circuit": {
                "qasm": qasm2.dumps(circuit),
                "num_qubits": circuit.num_qubits,
                "num_clbits": circuit.num_clbits,
                "depth": circuit.depth(),
                "size": circuit.size(),
            },
            "measurement": {
                "qubit": measure_qubit,
                "classical_bit": classical_bit,
                "type": "mid-circuit",
            },
            "usage": "Mid-circuit measurements enable dynamic quantum circuits and feedback control",
        }

    except Exception as e:
        logger.error(f"Failed to add mid-circuit measurement: {e}")
        return {
            "status": "error",
            "message": f"Failed to add mid-circuit measurement: {str(e)}",
        }


async def add_conditional_gate(
    circuit_qasm: str,
    gate_name: str,
    target_qubit: int,
    classical_bit: int,
    condition_value: int = 1,
) -> Dict[str, Any]:
    """Add a gate conditioned on a classical bit value.

    Implements c_if() functionality for classical control.

    Args:
        circuit_qasm: QASM representation of circuit
        gate_name: Gate to apply (x, y, z, h, etc.)
        target_qubit: Qubit to apply gate to
        classical_bit: Classical bit to condition on
        condition_value: Classical bit value to trigger gate (0 or 1, default: 1)

    Returns:
        Circuit with conditional gate added
    """
    try:
        circuit = qasm2.loads(circuit_qasm)

        # Map gate names to operations
        gate_map = {
            "x": circuit.x,
            "y": circuit.y,
            "z": circuit.z,
            "h": circuit.h,
            "s": circuit.s,
            "t": circuit.t,
            "rx": circuit.rx,
            "ry": circuit.ry,
            "rz": circuit.rz,
        }

        if gate_name.lower() not in gate_map:
            return {
                "status": "error",
                "message": f"Gate '{gate_name}' not supported. Use: {list(gate_map.keys())}",
            }

        # Add conditional gate
        gate_func = gate_map[gate_name.lower()]

        # For rotation gates, use default angle (need to handle differently)
        if gate_name.lower() in ["rx", "ry", "rz"]:
            gate_func(np.pi, target_qubit).c_if(classical_bit, condition_value)
        else:
            gate_func(target_qubit).c_if(classical_bit, condition_value)

        return {
            "status": "success",
            "message": f"Added conditional {gate_name} gate on qubit {target_qubit}",
            "circuit": {
                "qasm": qasm2.dumps(circuit),
                "num_qubits": circuit.num_qubits,
                "num_clbits": circuit.num_clbits,
                "depth": circuit.depth(),
            },
            "conditional_operation": {
                "gate": gate_name,
                "target_qubit": target_qubit,
                "condition_bit": classical_bit,
                "condition_value": condition_value,
            },
            "usage": "Conditional gates enable feedback control and dynamic quantum algorithms",
        }

    except Exception as e:
        logger.error(f"Failed to add conditional gate: {e}")
        return {
            "status": "error",
            "message": f"Failed to add conditional gate: {str(e)}",
        }


async def add_conditional_reset(
    circuit_qasm: str,
    target_qubit: int,
    classical_bit: int,
    reset_on_value: int = 1,
) -> Dict[str, Any]:
    """Add a reset conditioned on measurement outcome.

    Resets qubit to |0> if classical bit equals specified value.

    Args:
        circuit_qasm: QASM representation of circuit
        target_qubit: Qubit to reset
        classical_bit: Classical bit to condition on
        reset_on_value: Classical bit value to trigger reset (default: 1)

    Returns:
        Circuit with conditional reset added
    """
    try:
        circuit = qasm2.loads(circuit_qasm)

        # Add conditional X gate (which acts as reset from |1> to |0>)
        circuit.x(target_qubit).c_if(classical_bit, reset_on_value)

        return {
            "status": "success",
            "message": f"Added conditional reset on qubit {target_qubit}",
            "circuit": {
                "qasm": qasm2.dumps(circuit),
                "num_qubits": circuit.num_qubits,
                "num_clbits": circuit.num_clbits,
                "depth": circuit.depth(),
            },
            "conditional_reset": {
                "target_qubit": target_qubit,
                "condition_bit": classical_bit,
                "reset_on_value": reset_on_value,
                "implementation": "X gate conditioned on measurement outcome",
            },
            "usage": "Conditional reset is essential for quantum error correction and state preparation",
        }

    except Exception as e:
        logger.error(f"Failed to add conditional reset: {e}")
        return {
            "status": "error",
            "message": f"Failed to add conditional reset: {str(e)}",
        }


# ============================================================================
# Dynamic Circuit Patterns
# ============================================================================


async def create_teleportation_circuit() -> Dict[str, Any]:
    """Create a quantum teleportation circuit with mid-circuit measurements.

    Demonstrates classical communication and conditional operations.

    Returns:
        Quantum teleportation circuit
    """
    try:
        qr = QuantumRegister(3, "q")
        cr = ClassicalRegister(3, "c")
        circuit = QuantumCircuit(qr, cr)

        # Prepare entangled pair (qubits 1 and 2)
        circuit.h(1)
        circuit.cx(1, 2)

        # Bell measurement (qubit 0 and qubit 1)
        circuit.cx(0, 1)
        circuit.h(0)

        # Mid-circuit measurements
        circuit.measure(0, 0)
        circuit.measure(1, 1)

        # Conditional operations based on measurements
        circuit.x(2).c_if(cr[1], 1)
        circuit.z(2).c_if(cr[0], 1)

        # Final measurement
        circuit.measure(2, 2)

        return {
            "status": "success",
            "message": "Created quantum teleportation circuit",
            "circuit": {
                "qasm": qasm2.dumps(circuit),
                "num_qubits": circuit.num_qubits,
                "num_clbits": circuit.num_clbits,
                "depth": circuit.depth(),
            },
            "algorithm": "Quantum Teleportation",
            "description": "Teleports quantum state from qubit 0 to qubit 2 using entanglement and classical communication",
            "mid_circuit_measurements": 2,
            "conditional_operations": 2,
        }

    except Exception as e:
        logger.error(f"Failed to create teleportation circuit: {e}")
        return {
            "status": "error",
            "message": f"Failed to create teleportation circuit: {str(e)}",
        }


async def create_repeat_until_success_circuit(
    operation_qasm: str,
    max_attempts: int = 3,
) -> Dict[str, Any]:
    """Create a repeat-until-success circuit pattern.

    Repeats an operation until measurement indicates success.

    Args:
        operation_qasm: QASM of operation to repeat
        max_attempts: Maximum number of attempts (default: 3)

    Returns:
        Repeat-until-success circuit structure
    """
    try:
        operation = qasm2.loads(operation_qasm)

        # Create circuit with extra qubits for ancilla
        qr = QuantumRegister(operation.num_qubits + 1, "q")
        cr = ClassicalRegister(max_attempts, "c")
        circuit = QuantumCircuit(qr, cr)

        # Note: Full repeat-until-success requires runtime classical control
        # This creates the structure that would be used

        for attempt in range(max_attempts):
            # Apply operation
            circuit.compose(operation, range(operation.num_qubits), inplace=True)

            # Measure success ancilla
            circuit.measure(operation.num_qubits, attempt)

            # In hardware with dynamic circuits, would check and break here
            circuit.barrier()

        return {
            "status": "success",
            "message": f"Created repeat-until-success circuit structure ({max_attempts} attempts)",
            "circuit": {
                "qasm": qasm2.dumps(circuit),
                "num_qubits": circuit.num_qubits,
                "num_clbits": circuit.num_clbits,
                "depth": circuit.depth(),
            },
            "max_attempts": max_attempts,
            "note": "Full repeat-until-success requires runtime classical control flow",
            "usage": "RUS circuits enable probabilistic gate implementations with post-selection",
        }

    except Exception as e:
        logger.error(f"Failed to create RUS circuit: {e}")
        return {"status": "error", "message": f"Failed to create RUS circuit: {str(e)}"}


async def analyze_dynamic_circuit(circuit_qasm: str) -> Dict[str, Any]:
    """Analyze a circuit for dynamic features.

    Identifies mid-circuit measurements, conditional operations, and resets.

    Args:
        circuit_qasm: QASM representation of circuit

    Returns:
        Analysis of dynamic circuit features
    """
    try:
        circuit = qasm2.loads(circuit_qasm)

        # Count different operation types
        measurements = 0
        conditional_ops = 0
        resets = 0
        mid_circuit_measurements = 0

        measurement_positions = []

        for i, instruction in enumerate(circuit.data):
            op_name = instruction.operation.name.lower()

            if op_name == "measure":
                measurements += 1
                measurement_positions.append(i)
                # Check if it's mid-circuit (not at the end)
                if i < len(circuit.data) - circuit.num_qubits:
                    mid_circuit_measurements += 1

            if (
                hasattr(instruction.operation, "condition")
                and instruction.operation.condition is not None
            ):
                conditional_ops += 1

            if op_name == "reset":
                resets += 1

        has_dynamic_features = (mid_circuit_measurements > 0) or (conditional_ops > 0)

        return {
            "status": "success",
            "is_dynamic": has_dynamic_features,
            "num_qubits": circuit.num_qubits,
            "num_clbits": circuit.num_clbits,
            "total_measurements": measurements,
            "mid_circuit_measurements": mid_circuit_measurements,
            "conditional_operations": conditional_ops,
            "resets": resets,
            "circuit_depth": circuit.depth(),
            "circuit_size": circuit.size(),
            "measurement_positions": measurement_positions,
            "dynamic_features": {
                "has_mid_circuit_measurement": mid_circuit_measurements > 0,
                "has_conditional_ops": conditional_ops > 0,
                "has_resets": resets > 0,
            },
        }

    except Exception as e:
        logger.error(f"Failed to analyze dynamic circuit: {e}")
        return {
            "status": "error",
            "message": f"Failed to analyze dynamic circuit: {str(e)}",
        }


# Assisted by watsonx Code Assistant
