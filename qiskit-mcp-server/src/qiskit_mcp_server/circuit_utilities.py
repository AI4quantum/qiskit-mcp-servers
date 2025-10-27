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

"""Circuit utilities including manipulation methods, QASM file I/O, and converters."""

import logging
from typing import Any, Dict
import os

from qiskit import QuantumCircuit, qasm2, qasm3
from qiskit.converters import circuit_to_dag, dag_to_circuit

logger = logging.getLogger(__name__)


# ============================================================================
# Circuit Manipulation Methods
# ============================================================================


async def circuit_inverse(circuit_qasm: str) -> Dict[str, Any]:
    """Get the inverse of a quantum circuit.

    Args:
        circuit_qasm: QASM representation of circuit

    Returns:
        Inverse circuit information
    """
    try:
        circuit = qasm2.loads(circuit_qasm)

        # Remove measurements before inverting
        circuit.remove_final_measurements(inplace=True)

        # Get inverse
        inverse_circuit = circuit.inverse()

        return {
            "status": "success",
            "message": f"Created inverse of circuit with {circuit.num_qubits} qubits",
            "circuit": {
                "name": f"{circuit.name}_inverse" if circuit.name else "inverse",
                "num_qubits": inverse_circuit.num_qubits,
                "num_clbits": inverse_circuit.num_clbits,
                "depth": inverse_circuit.depth(),
                "size": inverse_circuit.size(),
                "qasm": qasm2.dumps(inverse_circuit),
            },
        }
    except Exception as e:
        logger.error(f"Failed to create inverse circuit: {e}")
        return {
            "status": "error",
            "message": f"Failed to create inverse circuit: {str(e)}",
        }


async def circuit_copy(circuit_qasm: str, name: str = None) -> Dict[str, Any]:
    """Create a deep copy of a quantum circuit.

    Args:
        circuit_qasm: QASM representation of circuit
        name: Optional name for the copied circuit

    Returns:
        Copied circuit information
    """
    try:
        circuit = qasm2.loads(circuit_qasm)

        # Create deep copy
        copied_circuit = circuit.copy(name=name)

        return {
            "status": "success",
            "message": f"Created copy of circuit with {circuit.num_qubits} qubits",
            "circuit": {
                "name": copied_circuit.name,
                "num_qubits": copied_circuit.num_qubits,
                "num_clbits": copied_circuit.num_clbits,
                "depth": copied_circuit.depth(),
                "size": copied_circuit.size(),
                "qasm": qasm2.dumps(copied_circuit),
            },
        }
    except Exception as e:
        logger.error(f"Failed to copy circuit: {e}")
        return {"status": "error", "message": f"Failed to copy circuit: {str(e)}"}


async def circuit_reverse_bits(circuit_qasm: str) -> Dict[str, Any]:
    """Reverse the order of bits in a circuit.

    Args:
        circuit_qasm: QASM representation of circuit

    Returns:
        Circuit with reversed bit order
    """
    try:
        circuit = qasm2.loads(circuit_qasm)

        # Reverse bits
        reversed_circuit = circuit.reverse_bits()

        return {
            "status": "success",
            "message": f"Reversed bit order in circuit with {circuit.num_qubits} qubits",
            "circuit": {
                "name": f"{circuit.name}_reversed" if circuit.name else "reversed",
                "num_qubits": reversed_circuit.num_qubits,
                "num_clbits": reversed_circuit.num_clbits,
                "depth": reversed_circuit.depth(),
                "size": reversed_circuit.size(),
                "qasm": qasm2.dumps(reversed_circuit),
            },
        }
    except Exception as e:
        logger.error(f"Failed to reverse circuit bits: {e}")
        return {
            "status": "error",
            "message": f"Failed to reverse circuit bits: {str(e)}",
        }


async def circuit_to_gate(circuit_qasm: str, label: str = None) -> Dict[str, Any]:
    """Convert a quantum circuit to a Gate object.

    Args:
        circuit_qasm: QASM representation of circuit
        label: Optional label for the gate

    Returns:
        Information about the created gate
    """
    try:
        circuit = qasm2.loads(circuit_qasm)

        # Remove measurements
        circuit.remove_final_measurements(inplace=True)

        # Convert to gate
        gate = circuit.to_gate(label=label)

        # Create a new circuit to demonstrate usage
        demo_circuit = QuantumCircuit(circuit.num_qubits)
        demo_circuit.append(gate, range(circuit.num_qubits))

        return {
            "status": "success",
            "message": f"Converted circuit to gate '{gate.label if gate.label else gate.name}'",
            "gate": {
                "name": gate.name,
                "label": gate.label,
                "num_qubits": gate.num_qubits,
                "num_params": len(gate.params),
            },
            "usage_example": {
                "qasm": qasm2.dumps(demo_circuit),
                "description": "Circuit showing how to use the created gate",
            },
        }
    except Exception as e:
        logger.error(f"Failed to convert circuit to gate: {e}")
        return {
            "status": "error",
            "message": f"Failed to convert circuit to gate: {str(e)}",
        }


async def circuit_to_instruction(
    circuit_qasm: str, label: str = None
) -> Dict[str, Any]:
    """Convert a quantum circuit to an Instruction object.

    Args:
        circuit_qasm: QASM representation of circuit
        label: Optional label for the instruction

    Returns:
        Information about the created instruction
    """
    try:
        circuit = qasm2.loads(circuit_qasm)

        # Convert to instruction
        instruction = circuit.to_instruction(label=label)

        # Create a new circuit to demonstrate usage
        demo_circuit = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
        demo_circuit.append(
            instruction, range(circuit.num_qubits), range(circuit.num_clbits)
        )

        return {
            "status": "success",
            "message": f"Converted circuit to instruction '{instruction.label if instruction.label else instruction.name}'",
            "instruction": {
                "name": instruction.name,
                "label": instruction.label,
                "num_qubits": instruction.num_qubits,
                "num_clbits": instruction.num_clbits,
            },
            "usage_example": {
                "qasm": qasm2.dumps(demo_circuit),
                "description": "Circuit showing how to use the created instruction",
            },
        }
    except Exception as e:
        logger.error(f"Failed to convert circuit to instruction: {e}")
        return {
            "status": "error",
            "message": f"Failed to convert circuit to instruction: {str(e)}",
        }


# ============================================================================
# QASM File I/O
# ============================================================================


async def load_qasm2_file(file_path: str) -> Dict[str, Any]:
    """Load a quantum circuit from a QASM 2.0 file.

    Args:
        file_path: Path to the QASM file

    Returns:
        Loaded circuit information
    """
    try:
        if not os.path.exists(file_path):
            return {
                "status": "error",
                "message": f"File not found: {file_path}",
            }

        circuit = qasm2.load(file_path)

        return {
            "status": "success",
            "message": f"Loaded circuit from {file_path}",
            "file_path": file_path,
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
        logger.error(f"Failed to load QASM 2.0 file: {e}")
        return {"status": "error", "message": f"Failed to load QASM 2.0 file: {str(e)}"}


async def save_qasm2_file(circuit_qasm: str, file_path: str) -> Dict[str, Any]:
    """Save a quantum circuit to a QASM 2.0 file.

    Args:
        circuit_qasm: QASM representation of circuit
        file_path: Path where the file should be saved

    Returns:
        Save operation result
    """
    try:
        circuit = qasm2.loads(circuit_qasm)

        # Ensure directory exists
        os.makedirs(
            os.path.dirname(file_path) if os.path.dirname(file_path) else ".",
            exist_ok=True,
        )

        # Save to file
        qasm2.dump(circuit, file_path)

        # Get file size
        file_size = os.path.getsize(file_path)

        return {
            "status": "success",
            "message": f"Saved circuit to {file_path}",
            "file_path": file_path,
            "file_size_bytes": file_size,
            "circuit_info": {
                "num_qubits": circuit.num_qubits,
                "num_clbits": circuit.num_clbits,
                "depth": circuit.depth(),
                "size": circuit.size(),
            },
        }
    except Exception as e:
        logger.error(f"Failed to save QASM 2.0 file: {e}")
        return {"status": "error", "message": f"Failed to save QASM 2.0 file: {str(e)}"}


async def load_qasm3_file(file_path: str) -> Dict[str, Any]:
    """Load a quantum circuit from a QASM 3.0 file.

    Args:
        file_path: Path to the QASM 3.0 file

    Returns:
        Loaded circuit information
    """
    try:
        if not os.path.exists(file_path):
            return {
                "status": "error",
                "message": f"File not found: {file_path}",
            }

        circuit = qasm3.load(file_path)

        return {
            "status": "success",
            "message": f"Loaded circuit from {file_path}",
            "file_path": file_path,
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
        logger.error(f"Failed to load QASM 3.0 file: {e}")
        return {"status": "error", "message": f"Failed to load QASM 3.0 file: {str(e)}"}


async def save_qasm3_file(circuit_qasm: str, file_path: str) -> Dict[str, Any]:
    """Save a quantum circuit to a QASM 3.0 file.

    Args:
        circuit_qasm: QASM representation of circuit
        file_path: Path where the file should be saved

    Returns:
        Save operation result
    """
    try:
        circuit = qasm2.loads(circuit_qasm)

        # Ensure directory exists
        os.makedirs(
            os.path.dirname(file_path) if os.path.dirname(file_path) else ".",
            exist_ok=True,
        )

        # Save to file
        with open(file_path, "w") as f:
            qasm3.dump(circuit, f)

        # Get file size
        file_size = os.path.getsize(file_path)

        return {
            "status": "success",
            "message": f"Saved circuit to {file_path} (QASM 3.0 format)",
            "file_path": file_path,
            "file_size_bytes": file_size,
            "circuit_info": {
                "num_qubits": circuit.num_qubits,
                "num_clbits": circuit.num_clbits,
                "depth": circuit.depth(),
                "size": circuit.size(),
            },
        }
    except Exception as e:
        logger.error(f"Failed to save QASM 3.0 file: {e}")
        return {"status": "error", "message": f"Failed to save QASM 3.0 file: {str(e)}"}


# ============================================================================
# Circuit Converters
# ============================================================================


async def convert_circuit_to_dag(circuit_qasm: str) -> Dict[str, Any]:
    """Convert a quantum circuit to a Directed Acyclic Graph (DAG) representation.

    Args:
        circuit_qasm: QASM representation of circuit

    Returns:
        DAG information
    """
    try:
        circuit = qasm2.loads(circuit_qasm)

        # Convert to DAG
        dag = circuit_to_dag(circuit)

        # Collect DAG statistics
        num_qubits = len(dag.qubits)
        num_clbits = len(dag.clbits)

        # Get gate operations (excluding input/output nodes)
        num_ops = dag.size()
        depth = dag.depth()

        # Count gate types
        op_counts = {}
        for node in dag.op_nodes():
            op_name = node.op.name
            op_counts[op_name] = op_counts.get(op_name, 0) + 1

        return {
            "status": "success",
            "message": f"Converted circuit to DAG with {num_ops} operations",
            "dag": {
                "num_qubits": num_qubits,
                "num_clbits": num_clbits,
                "num_operations": num_ops,
                "depth": depth,
                "operation_counts": op_counts,
                "num_tensor_factors": dag.num_tensor_factors(),
            },
            "note": "DAG representation is internal. Use convert_dag_to_circuit to get back to circuit format.",
        }
    except Exception as e:
        logger.error(f"Failed to convert circuit to DAG: {e}")
        return {
            "status": "error",
            "message": f"Failed to convert circuit to DAG: {str(e)}",
        }


async def convert_dag_to_circuit_wrapper(circuit_qasm: str) -> Dict[str, Any]:
    """Convert a DAG back to a quantum circuit (demonstration using circuit->DAG->circuit).

    Args:
        circuit_qasm: QASM representation of original circuit

    Returns:
        Reconstructed circuit information
    """
    try:
        # Start with circuit
        original_circuit = qasm2.loads(circuit_qasm)

        # Convert to DAG
        dag = circuit_to_dag(original_circuit)

        # Convert back to circuit
        reconstructed_circuit = dag_to_circuit(dag)

        return {
            "status": "success",
            "message": "Converted circuit -> DAG -> circuit successfully",
            "original": {
                "num_qubits": original_circuit.num_qubits,
                "depth": original_circuit.depth(),
                "size": original_circuit.size(),
            },
            "reconstructed": {
                "name": reconstructed_circuit.name,
                "num_qubits": reconstructed_circuit.num_qubits,
                "num_clbits": reconstructed_circuit.num_clbits,
                "depth": reconstructed_circuit.depth(),
                "size": reconstructed_circuit.size(),
                "qasm": qasm2.dumps(reconstructed_circuit),
            },
            "note": "DAG conversion preserves circuit structure and semantics",
        }
    except Exception as e:
        logger.error(f"Failed to convert DAG to circuit: {e}")
        return {
            "status": "error",
            "message": f"Failed to convert DAG to circuit: {str(e)}",
        }


async def decompose_circuit(
    circuit_qasm: str, gates_to_decompose: str = None, reps: int = 1
) -> Dict[str, Any]:
    """Decompose a circuit by expanding composite gates.

    Args:
        circuit_qasm: QASM representation of circuit
        gates_to_decompose: Comma-separated list of gate names to decompose (None = all)
        reps: Number of decomposition repetitions

    Returns:
        Decomposed circuit information
    """
    try:
        circuit = qasm2.loads(circuit_qasm)

        original_size = circuit.size()
        original_depth = circuit.depth()

        # Parse gates to decompose
        if gates_to_decompose:
            gates_list = [g.strip() for g in gates_to_decompose.split(",")]
            decomposed = circuit.decompose(gates_to_decompose=gates_list, reps=reps)
        else:
            decomposed = circuit.decompose(reps=reps)

        return {
            "status": "success",
            "message": f"Decomposed circuit {reps} time(s)",
            "original": {
                "depth": original_depth,
                "size": original_size,
            },
            "decomposed": {
                "name": f"{circuit.name}_decomposed" if circuit.name else "decomposed",
                "num_qubits": decomposed.num_qubits,
                "num_clbits": decomposed.num_clbits,
                "depth": decomposed.depth(),
                "size": decomposed.size(),
                "qasm": qasm2.dumps(decomposed),
            },
            "increase": {
                "depth": decomposed.depth() - original_depth,
                "size": decomposed.size() - original_size,
            },
        }
    except Exception as e:
        logger.error(f"Failed to decompose circuit: {e}")
        return {"status": "error", "message": f"Failed to decompose circuit: {str(e)}"}


# Assisted by watsonx Code Assistant
