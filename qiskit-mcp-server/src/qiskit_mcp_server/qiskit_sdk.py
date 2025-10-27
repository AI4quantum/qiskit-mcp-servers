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

"""Core Qiskit SDK functions for the MCP server."""

import logging
import re
from typing import Any, Dict, Optional

from qiskit import QuantumCircuit, qasm2, __version__ as qiskit_version
from qiskit import transpile
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Statevector

logger = logging.getLogger(__name__)


def _extract_qubit_index(qubit_str: str) -> int:
    """Extract qubit index from various formats.

    Supports:
    - Plain integers: "0", "1"
    - QASM format: "q[0]", "q[1]"
    - With commas: "q[0],"

    Args:
        qubit_str: String representation of a qubit

    Returns:
        Integer qubit index
    """
    # Remove commas and whitespace
    qubit_str = qubit_str.strip().rstrip(',')

    # Try QASM format: q[n]
    match = re.match(r'q\[(\d+)\]', qubit_str)
    if match:
        return int(match.group(1))

    # Try plain integer
    try:
        return int(qubit_str)
    except ValueError:
        raise ValueError(f"Cannot parse qubit index from: {qubit_str}")


def parse_gate_string(gate_str: str) -> list:
    """Parse a gate string into individual gate operations.

    Args:
        gate_str: String containing gate operations

    Returns:
        List of (gate_name, params, qubit_indices) tuples
        For gates without parameters, params is an empty list
    """
    gates = []
    for instruction in gate_str.split(";"):
        instruction = instruction.strip()
        if not instruction:
            continue

        # Check for invalid parametric format like "ry(0)" or "rz(θ[0])" before splitting
        if '(' in instruction and ')' in instruction and not instruction.startswith('measure'):
            logger.warning(
                f"Skipping parametric gate format '{instruction}'. "
                "Parametric circuits should be bound to specific values before adding gates. "
                "Use circuit.assign_parameters() or provide numeric angle values."
            )
            continue

        parts = instruction.split()
        if len(parts) < 2:
            logger.warning(f"Invalid gate instruction: {instruction}")
            continue

        gate_name = parts[0].lower()

        # Rotation gates that require angle parameters
        rotation_gates = ["rx", "ry", "rz", "p", "u"]

        try:
            if gate_name in rotation_gates:
                # For rotation gates: rx <angle> <qubit> or u <theta> <phi> <lambda> <qubit>
                if gate_name == "u" and len(parts) >= 5:
                    # U gate: u theta phi lambda qubit
                    params = [float(parts[1]), float(parts[2]), float(parts[3])]
                    qubits = [_extract_qubit_index(parts[4])]
                elif len(parts) >= 3:
                    # Single rotation: rx/ry/rz/p angle qubit
                    params = [float(parts[1])]
                    qubits = [_extract_qubit_index(parts[2])]
                else:
                    logger.warning(f"Invalid rotation gate instruction: {instruction}")
                    continue
                gates.append((gate_name, params, qubits))
            else:
                # Non-parameterized gates: just parse qubit indices
                qubits = [_extract_qubit_index(q) for q in parts[1:]]
                gates.append((gate_name, [], qubits))
        except (ValueError, IndexError) as e:
            logger.warning(f"Invalid gate instruction '{instruction}': {e}")

    return gates


async def create_quantum_circuit(
    num_qubits: int, num_classical_bits: int = 0, name: str = "circuit"
) -> Dict[str, Any]:
    """Create a new quantum circuit.

    Args:
        num_qubits: Number of quantum bits
        num_classical_bits: Number of classical bits
        name: Circuit name

    Returns:
        Circuit information including QASM representation
    """
    try:
        if num_qubits <= 0:
            return {
                "status": "error",
                "message": "Number of qubits must be positive",
            }

        circuit = QuantumCircuit(num_qubits, num_classical_bits, name=name)

        return {
            "status": "success",
            "message": f"Created circuit with {num_qubits} qubits and {num_classical_bits} classical bits",
            "circuit": {
                "name": circuit.name,
                "num_qubits": circuit.num_qubits,
                "num_clbits": circuit.num_clbits,
                "qasm": qasm2.dumps(circuit),
            },
        }
    except Exception as e:
        logger.error(f"Failed to create circuit: {e}")
        return {"status": "error", "message": f"Failed to create circuit: {str(e)}"}


async def add_gates_to_circuit(circuit_qasm: str, gates: str) -> Dict[str, Any]:
    """Add gates to an existing circuit.

    Args:
        circuit_qasm: QASM representation of the circuit
        gates: Gates to add (e.g., "h 0; cx 0 1")

    Returns:
        Updated circuit information
    """
    try:
        # Parse the QASM string to create a circuit
        circuit = qasm2.loads(circuit_qasm)

        # Parse and add gates
        parsed_gates = parse_gate_string(gates)

        for gate_info in parsed_gates:
            gate_name, params, qubits = gate_info

            # Rotation gates (parameterized)
            if gate_name == "rx":
                circuit.rx(params[0], qubits[0])
            elif gate_name == "ry":
                circuit.ry(params[0], qubits[0])
            elif gate_name == "rz":
                circuit.rz(params[0], qubits[0])
            elif gate_name == "p":
                circuit.p(params[0], qubits[0])
            elif gate_name == "u":
                circuit.u(params[0], params[1], params[2], qubits[0])
            # Standard single-qubit gates
            elif gate_name == "h":
                circuit.h(qubits[0])
            elif gate_name == "x":
                circuit.x(qubits[0])
            elif gate_name == "y":
                circuit.y(qubits[0])
            elif gate_name == "z":
                circuit.z(qubits[0])
            elif gate_name == "s":
                circuit.s(qubits[0])
            elif gate_name == "t":
                circuit.t(qubits[0])
            elif gate_name == "sdg":
                circuit.sdg(qubits[0])
            elif gate_name == "tdg":
                circuit.tdg(qubits[0])
            elif gate_name == "sx":
                circuit.sx(qubits[0])
            elif gate_name == "sxdg":
                circuit.sxdg(qubits[0])
            # Two-qubit gates
            elif gate_name == "cx" or gate_name == "cnot":
                if len(qubits) >= 2:
                    circuit.cx(qubits[0], qubits[1])
            elif gate_name == "cz":
                if len(qubits) >= 2:
                    circuit.cz(qubits[0], qubits[1])
            elif gate_name == "cy":
                if len(qubits) >= 2:
                    circuit.cy(qubits[0], qubits[1])
            elif gate_name == "swap":
                if len(qubits) >= 2:
                    circuit.swap(qubits[0], qubits[1])
            elif gate_name == "iswap":
                if len(qubits) >= 2:
                    circuit.iswap(qubits[0], qubits[1])
            elif gate_name == "ecr":
                if len(qubits) >= 2:
                    circuit.ecr(qubits[0], qubits[1])
            # Measurement
            elif gate_name == "measure":
                if len(qubits) >= 2:
                    circuit.measure(qubits[0], qubits[1])
                elif len(qubits) == 1:
                    # Measure qubit to same index classical bit
                    circuit.measure(qubits[0], qubits[0])
            # Barrier and reset
            elif gate_name == "barrier":
                if qubits:
                    circuit.barrier(qubits)
                else:
                    circuit.barrier()
            elif gate_name == "reset":
                circuit.reset(qubits[0])
            else:
                logger.warning(f"Unknown gate: {gate_name}")

        return {
            "status": "success",
            "message": f"Added {len(parsed_gates)} gate operation(s)",
            "circuit": {
                "name": circuit.name,
                "num_qubits": circuit.num_qubits,
                "num_clbits": circuit.num_clbits,
                "depth": circuit.depth(),
                "qasm": qasm2.dumps(circuit),
            },
        }
    except Exception as e:
        logger.error(f"Failed to add gates: {e}")
        return {"status": "error", "message": f"Failed to add gates: {str(e)}"}


async def transpile_circuit(
    circuit_qasm: str, optimization_level: int = 1, basis_gates: str = ""
) -> Dict[str, Any]:
    """Transpile a circuit.

    Args:
        circuit_qasm: QASM representation of the circuit
        optimization_level: Optimization level (0-3)
        basis_gates: Comma-separated basis gates

    Returns:
        Transpiled circuit information
    """
    try:
        circuit = qasm2.loads(circuit_qasm)

        # Parse basis gates if provided
        basis_gates_list = None
        if basis_gates:
            basis_gates_list = [g.strip() for g in basis_gates.split(",")]

        # Validate optimization level
        if optimization_level not in [0, 1, 2, 3]:
            optimization_level = 1

        # Transpile the circuit
        transpiled = transpile(
            circuit,
            optimization_level=optimization_level,
            basis_gates=basis_gates_list,
        )

        return {
            "status": "success",
            "message": f"Circuit transpiled with optimization level {optimization_level}",
            "original_depth": circuit.depth(),
            "transpiled_depth": transpiled.depth(),
            "circuit": {
                "name": transpiled.name,
                "num_qubits": transpiled.num_qubits,
                "num_clbits": transpiled.num_clbits,
                "depth": transpiled.depth(),
                "qasm": qasm2.dumps(transpiled),
            },
        }
    except Exception as e:
        logger.error(f"Failed to transpile circuit: {e}")
        return {"status": "error", "message": f"Failed to transpile circuit: {str(e)}"}


async def get_circuit_depth(circuit_qasm: str) -> Dict[str, Any]:
    """Get circuit depth.

    Args:
        circuit_qasm: QASM representation of the circuit

    Returns:
        Circuit depth information
    """
    try:
        circuit = qasm2.loads(circuit_qasm)

        return {
            "status": "success",
            "depth": circuit.depth(),
            "num_qubits": circuit.num_qubits,
            "size": circuit.size(),
        }
    except Exception as e:
        logger.error(f"Failed to get circuit depth: {e}")
        return {"status": "error", "message": f"Failed to get circuit depth: {str(e)}"}


async def get_circuit_qasm(circuit_qasm: str) -> Dict[str, Any]:
    """Get QASM representation of a circuit.

    Args:
        circuit_qasm: QASM representation of the circuit

    Returns:
        Formatted QASM string
    """
    try:
        circuit = qasm2.loads(circuit_qasm)

        return {
            "status": "success",
            "qasm": qasm2.dumps(circuit),
            "circuit_name": circuit.name,
        }
    except Exception as e:
        logger.error(f"Failed to get QASM: {e}")
        return {"status": "error", "message": f"Failed to get QASM: {str(e)}"}


async def get_statevector(circuit_qasm: str) -> Dict[str, Any]:
    """Get the statevector from a circuit simulation.

    Args:
        circuit_qasm: QASM representation of the circuit

    Returns:
        Statevector information
    """
    try:
        circuit = qasm2.loads(circuit_qasm)

        # Remove any measurements for statevector simulation
        circuit_copy = circuit.copy()
        circuit_copy.remove_final_measurements(inplace=True)

        # Calculate statevector
        statevector = Statevector.from_instruction(circuit_copy)

        return {
            "status": "success",
            "statevector": str(statevector),
            "probabilities": statevector.probabilities().tolist(),
            "num_qubits": circuit.num_qubits,
        }
    except Exception as e:
        logger.error(f"Failed to get statevector: {e}")
        return {"status": "error", "message": f"Failed to get statevector: {str(e)}"}


async def visualize_circuit(
    circuit_qasm: str, output_format: str = "text"
) -> Dict[str, Any]:
    """Visualize a circuit.

    Args:
        circuit_qasm: QASM representation of the circuit
        output_format: "text" for ASCII art

    Returns:
        Circuit visualization
    """
    try:
        circuit = qasm2.loads(circuit_qasm)

        if output_format == "text":
            # Get text drawing
            visualization = str(circuit.draw(output="text"))

            return {
                "status": "success",
                "format": "text",
                "visualization": visualization,
            }
        elif output_format == "mpl":
            return {
                "status": "success",
                "format": "mpl",
                "message": "Matplotlib visualization not supported in MCP server. Use text format instead.",
                "visualization": str(circuit.draw(output="text")),
            }
        else:
            return {
                "status": "error",
                "message": f"Unsupported output format: {output_format}. Use 'text'.",
            }
    except Exception as e:
        logger.error(f"Failed to visualize circuit: {e}")
        return {"status": "error", "message": f"Failed to visualize circuit: {str(e)}"}


async def create_random_circuit(
    num_qubits: int, depth: int, measure: bool = False, seed: Optional[int] = None
) -> Dict[str, Any]:
    """Create a random circuit.

    Args:
        num_qubits: Number of qubits
        depth: Circuit depth
        measure: Add measurements
        seed: Random seed

    Returns:
        Random circuit information
    """
    try:
        if num_qubits <= 0 or depth <= 0:
            return {
                "status": "error",
                "message": "Number of qubits and depth must be positive",
            }

        circuit = random_circuit(
            num_qubits=num_qubits,
            depth=depth,
            measure=measure,
            seed=seed,
        )

        return {
            "status": "success",
            "message": f"Created random circuit with {num_qubits} qubits and depth {depth}",
            "circuit": {
                "name": circuit.name,
                "num_qubits": circuit.num_qubits,
                "num_clbits": circuit.num_clbits,
                "depth": circuit.depth(),
                "size": circuit.size(),
                "qasm": qasm2.dumps(circuit),
            },
        }
    except Exception as e:
        logger.error(f"Failed to create random circuit: {e}")
        return {
            "status": "error",
            "message": f"Failed to create random circuit: {str(e)}",
        }


async def get_qiskit_version() -> Dict[str, Any]:
    """Get Qiskit version information.

    Returns:
        Version information
    """
    try:
        return {
            "status": "success",
            "version": qiskit_version,
            "sdk": "Qiskit",
        }
    except Exception as e:
        logger.error(f"Failed to get version: {e}")
        return {"status": "error", "message": f"Failed to get version: {str(e)}"}


# Assisted by watsonx Code Assistant
