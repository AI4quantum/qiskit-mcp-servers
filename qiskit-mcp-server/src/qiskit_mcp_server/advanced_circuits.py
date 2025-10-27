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

"""Advanced circuit operations including composition, library circuits, and analysis."""

import logging
import json
from typing import Any, Dict, Optional
import io
import base64

from qiskit import QuantumCircuit, qasm2, qasm3
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import (
    QFT,
    GroverOperator,
    PhaseOracle,
    EfficientSU2,
    RealAmplitudes,
    TwoLocal,
    PauliEvolutionGate,
    HGate,
    XGate,
    YGate,
    ZGate,
)
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator

logger = logging.getLogger(__name__)


def _load_qasm(qasm_str: str) -> QuantumCircuit:
    """Load QASM string, automatically detecting version.

    Args:
        qasm_str: QASM string (version 2.0 or 3.0)

    Returns:
        QuantumCircuit loaded from QASM

    Raises:
        Exception: If QASM cannot be parsed
    """
    qasm_str = qasm_str.strip()

    # Detect QASM version
    if qasm_str.startswith('OPENQASM 3'):
        # Try QASM 3.0 parser
        try:
            return qasm3.loads(qasm_str)
        except Exception as e:
            error_msg = str(e)
            if 'qiskit_qasm3_import' in error_msg:
                raise ValueError(
                    "QASM 3.0 import requires the 'qiskit_qasm3_import' package. "
                    "Install it with: pip install qiskit_qasm3_import"
                )
            raise ValueError(f"Failed to parse QASM 3.0: {e}")
    else:
        # Default to QASM 2.0 parser
        try:
            return qasm2.loads(qasm_str)
        except Exception as e:
            # If QASM 2.0 fails and string contains QASM 3.0 keywords, suggest version mismatch
            qasm3_keywords = ['for', 'while', 'if', 'include "stdgates.inc"', 'gate']
            if any(keyword in qasm_str for keyword in qasm3_keywords):
                raise ValueError(
                    f"Failed to parse as QASM 2.0. This might be QASM 3.0 content. "
                    f"Original error: {e}"
                )
            raise ValueError(f"Failed to parse QASM: {e}")


# ============================================================================
# Circuit Composition & Advanced Features
# ============================================================================


async def compose_circuits(
    circuit1_qasm: str, circuit2_qasm: str, inplace: bool = False
) -> Dict[str, Any]:
    """Compose two quantum circuits.

    Args:
        circuit1_qasm: QASM representation of first circuit
        circuit2_qasm: QASM representation of second circuit
        inplace: If True, modify first circuit; if False, return new circuit

    Returns:
        Composed circuit information
    """
    try:
        circuit1 = qasm2.loads(circuit1_qasm)
        circuit2 = qasm2.loads(circuit2_qasm)

        if circuit1.num_qubits != circuit2.num_qubits:
            return {
                "status": "error",
                "message": f"Circuits must have same number of qubits. Circuit1: {circuit1.num_qubits}, Circuit2: {circuit2.num_qubits}",
            }

        if inplace:
            circuit1.compose(circuit2, inplace=True)
            result_circuit = circuit1
        else:
            result_circuit = circuit1.compose(circuit2)

        return {
            "status": "success",
            "message": f"Composed circuits with {result_circuit.num_qubits} qubits",
            "circuit": {
                "name": result_circuit.name,
                "num_qubits": result_circuit.num_qubits,
                "num_clbits": result_circuit.num_clbits,
                "depth": result_circuit.depth(),
                "size": result_circuit.size(),
                "qasm": qasm2.dumps(result_circuit),
            },
        }
    except Exception as e:
        logger.error(f"Failed to compose circuits: {e}")
        return {"status": "error", "message": f"Failed to compose circuits: {str(e)}"}


async def tensor_circuits(circuit1_qasm: str, circuit2_qasm: str) -> Dict[str, Any]:
    """Tensor product of two quantum circuits (place side by side).

    Args:
        circuit1_qasm: QASM representation of first circuit
        circuit2_qasm: QASM representation of second circuit

    Returns:
        Tensored circuit information
    """
    try:
        circuit1 = qasm2.loads(circuit1_qasm)
        circuit2 = qasm2.loads(circuit2_qasm)

        result_circuit = circuit1.tensor(circuit2)

        return {
            "status": "success",
            "message": f"Tensored circuits: {circuit1.num_qubits} + {circuit2.num_qubits} = {result_circuit.num_qubits} qubits",
            "circuit": {
                "name": result_circuit.name,
                "num_qubits": result_circuit.num_qubits,
                "num_clbits": result_circuit.num_clbits,
                "depth": result_circuit.depth(),
                "size": result_circuit.size(),
                "qasm": qasm2.dumps(result_circuit),
            },
        }
    except Exception as e:
        logger.error(f"Failed to tensor circuits: {e}")
        return {"status": "error", "message": f"Failed to tensor circuits: {str(e)}"}


async def create_parametric_circuit(
    num_qubits: int, parameter_names: str, num_classical_bits: int = 0
) -> Dict[str, Any]:
    """Create a circuit with named parameters.

    Args:
        num_qubits: Number of qubits
        parameter_names: Comma-separated parameter names (e.g., "theta,phi,lambda")
        num_classical_bits: Number of classical bits

    Returns:
        Circuit with parameters information
    """
    try:
        circuit = QuantumCircuit(num_qubits, num_classical_bits)

        # Create parameters
        param_list = [name.strip() for name in parameter_names.split(",")]
        _ = [Parameter(name) for name in param_list]

        return {
            "status": "success",
            "message": f"Created parametric circuit with {num_qubits} qubits and parameters: {param_list}",
            "circuit": {
                "name": circuit.name,
                "num_qubits": circuit.num_qubits,
                "num_clbits": circuit.num_clbits,
                "parameters": param_list,
                "qasm": qasm2.dumps(circuit),
            },
            "parameters": {name: f"Parameter({name})" for name in param_list},
        }
    except Exception as e:
        logger.error(f"Failed to create parametric circuit: {e}")
        return {
            "status": "error",
            "message": f"Failed to create parametric circuit: {str(e)}",
        }


async def bind_parameters(circuit_qasm: str, parameter_values: str) -> Dict[str, Any]:
    """Bind parameter values to a parametric circuit.

    Args:
        circuit_qasm: QASM representation of parametric circuit
        parameter_values: JSON dict of parameter names to values, e.g., '{"theta": 1.57, "phi": 3.14}'

    Returns:
        Bound circuit information
    """
    try:
        circuit = qasm2.loads(circuit_qasm)

        # Parse parameter values
        param_dict = json.loads(parameter_values)

        # Bind parameters
        bound_circuit = circuit.assign_parameters(param_dict)

        return {
            "status": "success",
            "message": f"Bound {len(param_dict)} parameters",
            "circuit": {
                "name": bound_circuit.name,
                "num_qubits": bound_circuit.num_qubits,
                "num_clbits": bound_circuit.num_clbits,
                "depth": bound_circuit.depth(),
                "qasm": qasm2.dumps(bound_circuit),
            },
            "bound_parameters": param_dict,
        }
    except Exception as e:
        logger.error(f"Failed to bind parameters: {e}")
        return {"status": "error", "message": f"Failed to bind parameters: {str(e)}"}


async def decompose_circuit(circuit_qasm: str, reps: int = 1) -> Dict[str, Any]:
    """Decompose circuit gates into basis gates.

    Args:
        circuit_qasm: QASM representation of circuit
        reps: Number of decomposition repetitions

    Returns:
        Decomposed circuit information
    """
    try:
        circuit = qasm2.loads(circuit_qasm)

        decomposed = circuit.decompose(reps=reps)

        return {
            "status": "success",
            "message": f"Decomposed circuit {reps} time(s)",
            "original_depth": circuit.depth(),
            "decomposed_depth": decomposed.depth(),
            "original_size": circuit.size(),
            "decomposed_size": decomposed.size(),
            "circuit": {
                "name": decomposed.name,
                "num_qubits": decomposed.num_qubits,
                "num_clbits": decomposed.num_clbits,
                "depth": decomposed.depth(),
                "size": decomposed.size(),
                "qasm": qasm2.dumps(decomposed),
            },
        }
    except Exception as e:
        logger.error(f"Failed to decompose circuit: {e}")
        return {"status": "error", "message": f"Failed to decompose circuit: {str(e)}"}


# ============================================================================
# Advanced Gate Operations
# ============================================================================


async def add_controlled_gate(
    circuit_qasm: str, gate_name: str, control_qubits: str, target_qubits: str
) -> Dict[str, Any]:
    """Add a controlled version of a gate to the circuit.

    Args:
        circuit_qasm: QASM representation of circuit
        gate_name: Name of gate to control (h, x, y, z, s, t, rx, ry, rz)
        control_qubits: Comma-separated control qubit indices
        target_qubits: Comma-separated target qubit indices

    Returns:
        Updated circuit information
    """
    try:
        circuit = qasm2.loads(circuit_qasm)

        controls = [int(q.strip()) for q in control_qubits.split(",")]
        targets = [int(q.strip()) for q in target_qubits.split(",")]

        # Create controlled gate
        if gate_name.lower() == "x":
            if len(controls) == 1 and len(targets) == 1:
                circuit.cx(controls[0], targets[0])
            elif len(controls) == 2 and len(targets) == 1:
                circuit.ccx(controls[0], controls[1], targets[0])
            else:
                circuit.mcx(controls, targets[0])
        elif gate_name.lower() == "z":
            if len(controls) == 1 and len(targets) == 1:
                circuit.cz(controls[0], targets[0])
            elif len(controls) == 2 and len(targets) == 1:
                circuit.ccz(controls[0], controls[1], targets[0])
            else:
                circuit.mcz(controls, targets[0])
        elif gate_name.lower() == "h":
            circuit.ch(controls[0], targets[0])
        elif gate_name.lower() == "y":
            circuit.cy(controls[0], targets[0])
        elif gate_name.lower() == "swap":
            circuit.cswap(controls[0], targets[0], targets[1])
        else:
            return {
                "status": "error",
                "message": f"Controlled {gate_name} not supported. Use x, z, h, y, or swap.",
            }

        return {
            "status": "success",
            "message": f"Added controlled {gate_name} gate",
            "circuit": {
                "name": circuit.name,
                "num_qubits": circuit.num_qubits,
                "num_clbits": circuit.num_clbits,
                "depth": circuit.depth(),
                "qasm": qasm2.dumps(circuit),
            },
        }
    except Exception as e:
        logger.error(f"Failed to add controlled gate: {e}")
        return {
            "status": "error",
            "message": f"Failed to add controlled gate: {str(e)}",
        }


async def add_power_gate(
    circuit_qasm: str, gate_name: str, power: float, qubit: int
) -> Dict[str, Any]:
    """Add a gate raised to a power.

    Args:
        circuit_qasm: QASM representation of circuit
        gate_name: Name of gate (x, y, z, h, s, t)
        power: Power to raise gate to
        qubit: Target qubit index

    Returns:
        Updated circuit information
    """
    try:
        circuit = qasm2.loads(circuit_qasm)

        # Map gate names to classes
        gate_map = {
            "x": XGate(),
            "y": YGate(),
            "z": ZGate(),
            "h": HGate(),
        }

        if gate_name.lower() not in gate_map:
            return {
                "status": "error",
                "message": f"Gate {gate_name} not supported for power operation. Use x, y, z, or h.",
            }

        gate = gate_map[gate_name.lower()]
        powered_gate = gate.power(power)
        circuit.append(powered_gate, [qubit])

        return {
            "status": "success",
            "message": f"Added {gate_name}^{power} gate to qubit {qubit}",
            "circuit": {
                "name": circuit.name,
                "num_qubits": circuit.num_qubits,
                "num_clbits": circuit.num_clbits,
                "depth": circuit.depth(),
                "qasm": qasm2.dumps(circuit),
            },
        }
    except Exception as e:
        logger.error(f"Failed to add power gate: {e}")
        return {"status": "error", "message": f"Failed to add power gate: {str(e)}"}


# ============================================================================
# Circuit Library
# ============================================================================


async def create_qft_circuit(
    num_qubits: int, inverse: bool = False, do_swaps: bool = True
) -> Dict[str, Any]:
    """Create a Quantum Fourier Transform circuit.

    Args:
        num_qubits: Number of qubits
        inverse: If True, create inverse QFT
        do_swaps: If True, include swap gates

    Returns:
        QFT circuit information
    """
    try:
        qft = QFT(num_qubits=num_qubits, inverse=inverse, do_swaps=do_swaps)
        circuit = QuantumCircuit(num_qubits)
        circuit.append(qft, range(num_qubits))

        return {
            "status": "success",
            "message": f"Created {'inverse ' if inverse else ''}QFT circuit with {num_qubits} qubits",
            "circuit": {
                "name": f"{'IQFT' if inverse else 'QFT'}_{num_qubits}",
                "num_qubits": circuit.num_qubits,
                "num_clbits": circuit.num_clbits,
                "depth": circuit.depth(),
                "size": circuit.size(),
                "qasm": qasm2.dumps(circuit),
            },
        }
    except Exception as e:
        logger.error(f"Failed to create QFT circuit: {e}")
        return {"status": "error", "message": f"Failed to create QFT circuit: {str(e)}"}


async def create_grover_operator(num_qubits: int, oracle_qasm: str) -> Dict[str, Any]:
    """Create a Grover operator circuit.

    Args:
        num_qubits: Number of qubits
        oracle_qasm: QASM representation of oracle circuit

    Returns:
        Grover operator circuit information
    """
    try:
        oracle_circuit = qasm2.loads(oracle_qasm)

        # Create Grover operator
        grover_op = GroverOperator(oracle_circuit)

        circuit = QuantumCircuit(num_qubits)
        circuit.append(grover_op, range(num_qubits))

        return {
            "status": "success",
            "message": f"Created Grover operator with {num_qubits} qubits",
            "circuit": {
                "name": "GroverOperator",
                "num_qubits": circuit.num_qubits,
                "num_clbits": circuit.num_clbits,
                "depth": circuit.depth(),
                "size": circuit.size(),
                "qasm": qasm2.dumps(circuit),
            },
        }
    except Exception as e:
        logger.error(f"Failed to create Grover operator: {e}")
        return {
            "status": "error",
            "message": f"Failed to create Grover operator: {str(e)}",
        }


async def create_efficient_su2(
    num_qubits: int, reps: int = 3, entanglement: str = "full"
) -> Dict[str, Any]:
    """Create an EfficientSU2 variational circuit.

    Args:
        num_qubits: Number of qubits
        reps: Number of repetitions
        entanglement: Entanglement strategy (full, linear, circular)

    Returns:
        EfficientSU2 circuit information
    """
    try:
        circuit = EfficientSU2(
            num_qubits=num_qubits, reps=reps, entanglement=entanglement
        )

        return {
            "status": "success",
            "message": f"Created EfficientSU2 circuit with {num_qubits} qubits and {reps} repetitions",
            "circuit": {
                "name": "EfficientSU2",
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
        logger.error(f"Failed to create EfficientSU2: {e}")
        return {
            "status": "error",
            "message": f"Failed to create EfficientSU2: {str(e)}",
        }


async def create_real_amplitudes(
    num_qubits: int, reps: int = 3, entanglement: str = "full"
) -> Dict[str, Any]:
    """Create a RealAmplitudes variational circuit.

    Args:
        num_qubits: Number of qubits
        reps: Number of repetitions
        entanglement: Entanglement strategy (full, linear, circular)

    Returns:
        RealAmplitudes circuit information
    """
    try:
        circuit = RealAmplitudes(
            num_qubits=num_qubits, reps=reps, entanglement=entanglement
        )

        return {
            "status": "success",
            "message": f"Created RealAmplitudes circuit with {num_qubits} qubits and {reps} repetitions",
            "circuit": {
                "name": "RealAmplitudes",
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
        logger.error(f"Failed to create RealAmplitudes: {e}")
        return {
            "status": "error",
            "message": f"Failed to create RealAmplitudes: {str(e)}",
        }


# ============================================================================
# Simulation Backends
# ============================================================================


async def simulate_with_aer(
    circuit_qasm: str, shots: int = 1024, backend_name: str = "aer_simulator"
) -> Dict[str, Any]:
    """Simulate circuit using Aer simulator.

    Args:
        circuit_qasm: QASM representation of circuit
        shots: Number of shots
        backend_name: Aer backend name (aer_simulator, aer_simulator_statevector)

    Returns:
        Simulation results
    """
    try:
        circuit = qasm2.loads(circuit_qasm)

        # Add measurements if not present
        if circuit.num_clbits == 0:
            circuit.measure_all()

        # Create Aer simulator
        simulator = AerSimulator(method="automatic")

        # Run simulation
        job = simulator.run(circuit, shots=shots)
        result = job.result()
        counts = result.get_counts()

        return {
            "status": "success",
            "message": f"Simulated with {shots} shots using {backend_name}",
            "backend": backend_name,
            "shots": shots,
            "counts": dict(counts),
            "success": result.success,
        }
    except Exception as e:
        logger.error(f"Failed to simulate with Aer: {e}")
        return {"status": "error", "message": f"Failed to simulate with Aer: {str(e)}"}


async def get_unitary_matrix(circuit_qasm: str) -> Dict[str, Any]:
    """Get the unitary matrix of a circuit.

    Args:
        circuit_qasm: QASM representation of circuit

    Returns:
        Unitary matrix information
    """
    try:
        circuit = qasm2.loads(circuit_qasm)

        # Remove measurements
        circuit_copy = circuit.copy()
        circuit_copy.remove_final_measurements(inplace=True)

        # Get unitary
        unitary = Operator(circuit_copy)

        return {
            "status": "success",
            "message": f"Computed unitary matrix for {circuit.num_qubits}-qubit circuit",
            "num_qubits": circuit.num_qubits,
            "dimension": 2**circuit.num_qubits,
            "unitary": str(unitary),
            "is_unitary": unitary.is_unitary(),
        }
    except Exception as e:
        logger.error(f"Failed to get unitary matrix: {e}")
        return {"status": "error", "message": f"Failed to get unitary matrix: {str(e)}"}


# ============================================================================
# Circuit Analysis
# ============================================================================


async def analyze_circuit(circuit_qasm: str) -> Dict[str, Any]:
    """Perform comprehensive circuit analysis.

    Args:
        circuit_qasm: QASM representation of circuit

    Returns:
        Circuit analysis information
    """
    try:
        circuit = _load_qasm(circuit_qasm)

        # Count gates by type
        gate_counts: dict[str, int] = {}
        for instruction in circuit.data:
            gate_name = instruction.operation.name
            gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1

        # Get two-qubit gate count
        two_qubit_gates = sum(1 for inst in circuit.data if len(inst.qubits) == 2)

        return {
            "status": "success",
            "circuit_name": circuit.name,
            "num_qubits": circuit.num_qubits,
            "num_clbits": circuit.num_clbits,
            "depth": circuit.depth(),
            "size": circuit.size(),
            "width": circuit.width(),
            "num_parameters": circuit.num_parameters,
            "gate_counts": gate_counts,
            "two_qubit_gate_count": two_qubit_gates,
            "num_nonlocal_gates": circuit.num_nonlocal_gates(),
            "num_connected_components": circuit.num_connected_components(),
        }
    except Exception as e:
        logger.error(f"Failed to analyze circuit: {e}")
        return {"status": "error", "message": f"Failed to analyze circuit: {str(e)}"}


async def get_circuit_instructions(circuit_qasm: str) -> Dict[str, Any]:
    """Get detailed list of circuit instructions.

    Args:
        circuit_qasm: QASM representation of circuit

    Returns:
        List of circuit instructions
    """
    try:
        circuit = _load_qasm(circuit_qasm)

        instructions = []
        for i, instruction in enumerate(circuit.data):
            inst_info = {
                "index": i,
                "gate": instruction.operation.name,
                "qubits": [circuit.find_bit(q).index for q in instruction.qubits],
                "clbits": [circuit.find_bit(c).index for c in instruction.clbits],
                "params": [
                    float(p) if hasattr(p, "__float__") else str(p)
                    for p in instruction.operation.params
                ],
            }
            instructions.append(inst_info)

        return {
            "status": "success",
            "num_instructions": len(instructions),
            "instructions": instructions,
        }
    except Exception as e:
        logger.error(f"Failed to get circuit instructions: {e}")
        return {
            "status": "error",
            "message": f"Failed to get circuit instructions: {str(e)}",
        }


# ============================================================================
# OpenQASM 3 Support
# ============================================================================


async def convert_to_qasm3(circuit_qasm: str) -> Dict[str, Any]:
    """Convert circuit to OpenQASM 3.0 format.

    Args:
        circuit_qasm: QASM 2.0 representation of circuit

    Returns:
        QASM 3.0 representation
    """
    try:
        circuit = qasm2.loads(circuit_qasm)

        # Convert to QASM 3
        qasm3_str = qasm3.dumps(circuit)

        return {
            "status": "success",
            "message": "Converted circuit to OpenQASM 3.0",
            "qasm3": qasm3_str,
            "qasm2": qasm2.dumps(circuit),
        }
    except Exception as e:
        logger.error(f"Failed to convert to QASM3: {e}")
        return {"status": "error", "message": f"Failed to convert to QASM3: {str(e)}"}


async def load_qasm3_circuit(qasm3_str: str) -> Dict[str, Any]:
    """Load circuit from OpenQASM 3.0 format.

    Args:
        qasm3_str: QASM 3.0 representation

    Returns:
        Circuit information
    """
    try:
        circuit = qasm3.loads(qasm3_str)

        return {
            "status": "success",
            "message": "Loaded circuit from OpenQASM 3.0",
            "circuit": {
                "name": circuit.name,
                "num_qubits": circuit.num_qubits,
                "num_clbits": circuit.num_clbits,
                "depth": circuit.depth(),
                "size": circuit.size(),
                "qasm2": qasm2.dumps(circuit),
                "qasm3": qasm3.dumps(circuit),
            },
        }
    except Exception as e:
        logger.error(f"Failed to load QASM3: {e}")
        return {"status": "error", "message": f"Failed to load QASM3: {str(e)}"}


# ============================================================================
# Circuit Drawing & Visualization
# ============================================================================


async def draw_circuit_text(circuit_qasm: str, fold: int = -1) -> Dict[str, Any]:
    """Draw circuit as text/ASCII art.

    Args:
        circuit_qasm: QASM representation of circuit
        fold: Column to fold the circuit at (-1 for no folding)

    Returns:
        Text visualization of circuit
    """
    try:
        circuit = qasm2.loads(circuit_qasm)

        if fold > 0:
            text_drawing = str(circuit.draw(output="text", fold=fold))
        else:
            text_drawing = str(circuit.draw(output="text"))

        return {
            "status": "success",
            "format": "text",
            "drawing": text_drawing,
            "circuit_info": {
                "num_qubits": circuit.num_qubits,
                "depth": circuit.depth(),
                "size": circuit.size(),
            },
        }
    except Exception as e:
        logger.error(f"Failed to draw circuit: {e}")
        return {"status": "error", "message": f"Failed to draw circuit: {str(e)}"}


async def draw_circuit_matplotlib(
    circuit_qasm: str, style: str = "default"
) -> Dict[str, Any]:
    """Draw circuit using matplotlib (returns base64 encoded image).

    Args:
        circuit_qasm: QASM representation of circuit
        style: Drawing style (default, iqp, clifford, textbook)

    Returns:
        Base64 encoded PNG image
    """
    try:
        import matplotlib.pyplot as plt

        circuit = qasm2.loads(circuit_qasm)

        # Create figure
        fig = circuit.draw(output="mpl", style=style)

        # Save to base64
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)

        return {
            "status": "success",
            "format": "matplotlib",
            "style": style,
            "image_base64": img_base64,
            "circuit_info": {
                "num_qubits": circuit.num_qubits,
                "depth": circuit.depth(),
                "size": circuit.size(),
            },
        }
    except ImportError:
        return {
            "status": "error",
            "message": "Matplotlib not available. Install with: pip install matplotlib",
        }
    except Exception as e:
        logger.error(f"Failed to draw circuit with matplotlib: {e}")
        return {"status": "error", "message": f"Failed to draw circuit: {str(e)}"}


# ============================================================================
# Missing Circuit Library Functions
# ============================================================================


async def create_pauli_evolution_circuit(
    pauli_string: str, time: float, num_qubits: Optional[int] = None
) -> Dict[str, Any]:
    """Create a circuit for Pauli operator time evolution.

    Implements exp(-i * time * PauliOp) for Hamiltonian simulation.

    Args:
        pauli_string: Pauli string (e.g., "XXYZI", "XYZ")
        time: Evolution time
        num_qubits: Number of qubits (inferred from string if not provided)

    Returns:
        Circuit with Pauli evolution gate
    """
    try:
        from qiskit.quantum_info import SparsePauliOp

        # Infer num_qubits if not provided
        if num_qubits is None:
            num_qubits = len(pauli_string)

        # Pad pauli_string with 'I' if shorter than num_qubits
        if len(pauli_string) < num_qubits:
            pauli_string = pauli_string + "I" * (num_qubits - len(pauli_string))
        elif len(pauli_string) > num_qubits:
            return {
                "status": "error",
                "message": f"Pauli string length ({len(pauli_string)}) cannot exceed num_qubits ({num_qubits})",
            }

        # Create Pauli operator
        pauli_op = SparsePauliOp(pauli_string)

        # Create circuit with evolution gate
        circuit = QuantumCircuit(num_qubits)
        evolution_gate = PauliEvolutionGate(pauli_op, time=time)
        circuit.append(evolution_gate, range(num_qubits))

        return {
            "status": "success",
            "message": f"Created Pauli evolution circuit for {pauli_string}",
            "pauli_string": pauli_string,
            "time": time,
            "circuit": {
                "name": f"pauli_evolution_{pauli_string}",
                "num_qubits": circuit.num_qubits,
                "depth": circuit.depth(),
                "size": circuit.size(),
                "qasm": qasm2.dumps(circuit),
            },
            "usage": "Used for Hamiltonian simulation, VQE, and Trotter evolution",
        }
    except Exception as e:
        logger.error(f"Failed to create Pauli evolution circuit: {e}")
        return {
            "status": "error",
            "message": f"Failed to create Pauli evolution circuit: {str(e)}",
        }


async def create_phase_oracle_circuit(
    expression: str, num_qubits: int
) -> Dict[str, Any]:
    """Create a phase oracle from a boolean expression.

    Args:
        expression: Boolean expression (e.g., "(x0 & x1) | ~x2")
        num_qubits: Number of qubits

    Returns:
        Circuit with phase oracle
    """
    try:
        # Create phase oracle (mct_mode removed in Qiskit 2.2+)
        oracle = PhaseOracle(expression)

        # Create circuit
        circuit = QuantumCircuit(num_qubits)
        circuit.compose(oracle, inplace=True)

        return {
            "status": "success",
            "message": f"Created phase oracle for expression: {expression}",
            "expression": expression,
            "circuit": {
                "name": "phase_oracle",
                "num_qubits": circuit.num_qubits,
                "depth": circuit.depth(),
                "size": circuit.size(),
                "qasm": qasm2.dumps(circuit),
            },
            "usage": "Used in Grover's algorithm and amplitude amplification",
        }
    except Exception as e:
        logger.error(f"Failed to create phase oracle: {e}")
        return {
            "status": "error",
            "message": f"Failed to create phase oracle: {str(e)}",
        }


async def create_general_two_local_circuit(
    num_qubits: int,
    rotation_blocks: str = "ry",
    entanglement_blocks: str = "cx",
    entanglement: str = "full",
    reps: int = 3,
    insert_barriers: bool = False,
) -> Dict[str, Any]:
    """Create a general TwoLocal variational circuit.

    More flexible than EfficientSU2 - allows custom rotation and entanglement gates.

    Args:
        num_qubits: Number of qubits
        rotation_blocks: Rotation gate(s) - "ry", "rz", "rx", or combinations like "ry,rz"
        entanglement_blocks: Entanglement gate - "cx", "cz", "cry", "crz"
        entanglement: Entanglement pattern - "full", "linear", "circular", "sca"
        reps: Number of repetitions
        insert_barriers: Whether to insert barriers between layers

    Returns:
        TwoLocal circuit information
    """
    try:
        # Parse rotation blocks
        if "," in rotation_blocks:
            rot_blocks = [g.strip() for g in rotation_blocks.split(",")]
        else:
            rot_blocks = rotation_blocks

        # Create TwoLocal circuit
        circuit = TwoLocal(
            num_qubits=num_qubits,
            rotation_blocks=rot_blocks,
            entanglement_blocks=entanglement_blocks,
            entanglement=entanglement,
            reps=reps,
            insert_barriers=insert_barriers,
        )

        # For parametric circuits, bind to dummy values to generate QASM
        import numpy as np

        if circuit.num_parameters > 0:
            param_values = np.zeros(circuit.num_parameters)
            bound_circuit = circuit.assign_parameters(param_values)
            qasm_str = qasm2.dumps(bound_circuit)
        else:
            qasm_str = qasm2.dumps(circuit)

        return {
            "status": "success",
            "message": f"Created TwoLocal circuit with {num_qubits} qubits and {reps} repetitions",
            "parameters": {
                "num_qubits": num_qubits,
                "rotation_blocks": rotation_blocks,
                "entanglement_blocks": entanglement_blocks,
                "entanglement": entanglement,
                "reps": reps,
                "insert_barriers": insert_barriers,
            },
            "circuit": {
                "name": "two_local",
                "num_qubits": circuit.num_qubits,
                "num_parameters": circuit.num_parameters,
                "depth": circuit.depth(),
                "size": circuit.size(),
                "qasm": qasm_str,
            },
            "usage": "General-purpose variational ansatz for VQE and QAOA",
            "note": "QASM generated with parameters bound to 0"
            if circuit.num_parameters > 0
            else None,
        }
    except Exception as e:
        logger.error(f"Failed to create TwoLocal circuit: {e}")
        return {
            "status": "error",
            "message": f"Failed to create TwoLocal circuit: {str(e)}",
        }


async def create_parametric_circuit_with_vector(
    num_qubits: int, num_parameters: int, structure: str = "ry_cx"
) -> Dict[str, Any]:
    """Create a parametric circuit using ParameterVector for convenient parameter management.

    Args:
        num_qubits: Number of qubits
        num_parameters: Number of parameters
        structure: Circuit structure - "ry_cx", "rz_cz", "rx_crz"

    Returns:
        Parametric circuit with parameter vector
    """
    try:
        # Create parameter vector
        params = ParameterVector("θ", num_parameters)

        circuit = QuantumCircuit(num_qubits)

        # Build circuit based on structure
        param_idx = 0
        if structure == "ry_cx":
            for i in range(num_qubits):
                if param_idx < num_parameters:
                    circuit.ry(params[param_idx], i)
                    param_idx += 1
            for i in range(num_qubits - 1):
                circuit.cx(i, i + 1)
        elif structure == "rz_cz":
            for i in range(num_qubits):
                if param_idx < num_parameters:
                    circuit.rz(params[param_idx], i)
                    param_idx += 1
            for i in range(num_qubits - 1):
                circuit.cz(i, i + 1)
        elif structure == "rx_crz":
            for i in range(num_qubits):
                if param_idx < num_parameters:
                    circuit.rx(params[param_idx], i)
                    param_idx += 1
            for i in range(num_qubits - 1):
                if param_idx < num_parameters:
                    circuit.crz(params[param_idx], i, i + 1)
                    param_idx += 1
        else:
            return {
                "status": "error",
                "message": f"Unknown structure: {structure}. Use 'ry_cx', 'rz_cz', or 'rx_crz'",
            }

        # Get parameter names
        param_names = [str(p) for p in params]

        # For parametric circuits, bind to dummy values to generate QASM
        import numpy as np

        if circuit.num_parameters > 0:
            param_values = np.zeros(circuit.num_parameters)
            bound_circuit = circuit.assign_parameters(param_values)
            qasm_str = qasm2.dumps(bound_circuit)
        else:
            qasm_str = qasm2.dumps(circuit)

        return {
            "status": "success",
            "message": f"Created parametric circuit with ParameterVector of size {num_parameters}",
            "parameters": {
                "num_qubits": num_qubits,
                "num_parameters": num_parameters,
                "structure": structure,
                "parameter_vector_name": "θ",
                "parameter_names": param_names,
            },
            "circuit": {
                "name": "parametric_vector",
                "num_qubits": circuit.num_qubits,
                "num_parameters": circuit.num_parameters,
                "depth": circuit.depth(),
                "size": circuit.size(),
                "qasm": qasm_str,
            },
            "usage": "Convenient parameter management for variational algorithms",
            "note": "QASM generated with parameters bound to 0"
            if circuit.num_parameters > 0
            else None,
        }
    except Exception as e:
        logger.error(f"Failed to create parametric circuit with vector: {e}")
        return {
            "status": "error",
            "message": f"Failed to create parametric circuit with vector: {str(e)}",
        }


# Assisted by watsonx Code Assistant
