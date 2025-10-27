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

"""PassManager and transpiler pass functions."""

import logging
import json
from typing import Any, Dict, Optional

from qiskit import qasm2, transpile
from qiskit.transpiler import PassManager, CouplingMap
from qiskit.transpiler.passes import (
    Optimize1qGates,
    InverseCancellation,
    CommutativeCancellation,
    RemoveDiagonalGatesBeforeMeasure,
    Depth,
    Size,
    Width,
    CountOps,
    NumTensorFactors,
)
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

logger = logging.getLogger(__name__)


async def run_preset_pass_manager(
    circuit_qasm: str,
    optimization_level: int = 1,
    backend_name: Optional[str] = None,
    coupling_map_json: Optional[str] = None,
    basis_gates: Optional[str] = None,
    seed_transpiler: Optional[int] = None,
) -> Dict[str, Any]:
    """Run a preset pass manager on a circuit.

    Args:
        circuit_qasm: QASM representation of the circuit
        optimization_level: Optimization level (0-3)
        backend_name: Optional backend name for automatic configuration
        coupling_map_json: JSON array of edges or "linear:N" or "grid:NxM"
        basis_gates: Comma-separated basis gates
        seed_transpiler: Random seed for reproducibility

    Returns:
        Transpiled circuit with pass manager details
    """
    try:
        circuit = qasm2.loads(circuit_qasm)

        # Get backend if specified
        backend = None
        if backend_name:
            try:
                from qiskit_ibm_runtime import QiskitRuntimeService

                service = QiskitRuntimeService()
                backend = service.backend(backend_name)
            except ImportError:
                return {
                    "status": "error",
                    "message": "qiskit-ibm-runtime not installed. Install with: pip install qiskit-ibm-runtime",
                }

        # Parse coupling map
        coupling_map = None
        if coupling_map_json:
            if coupling_map_json.startswith("linear:"):
                num_qubits = int(coupling_map_json.split(":")[1])
                coupling_map = CouplingMap.from_line(num_qubits)
            elif coupling_map_json.startswith("grid:"):
                dims = coupling_map_json.split(":")[1].split("x")
                rows, cols = int(dims[0]), int(dims[1])
                coupling_map = CouplingMap.from_grid(rows, cols)
            else:
                edges = json.loads(coupling_map_json)
                coupling_map = CouplingMap(edges)

        # Parse basis gates
        basis_gates_list = None
        if basis_gates:
            basis_gates_list = [g.strip() for g in basis_gates.split(",")]

        # Generate preset pass manager
        pm = generate_preset_pass_manager(
            optimization_level=optimization_level,
            backend=backend,
            coupling_map=coupling_map,
            basis_gates=basis_gates_list,
            seed_transpiler=seed_transpiler,
        )

        # Run pass manager
        transpiled = pm.run(circuit)

        return {
            "status": "success",
            "message": f"Ran preset pass manager with optimization level {optimization_level}",
            "configuration": {
                "optimization_level": optimization_level,
                "backend": backend_name,
                "coupling_map": coupling_map_json,
                "basis_gates": basis_gates_list,
                "seed_transpiler": seed_transpiler,
            },
            "original": {
                "depth": circuit.depth(),
                "size": circuit.size(),
                "num_qubits": circuit.num_qubits,
            },
            "transpiled": {
                "depth": transpiled.depth(),
                "size": transpiled.size(),
                "num_qubits": transpiled.num_qubits,
                "qasm": qasm2.dumps(transpiled),
            },
            "improvement": {
                "depth_reduction": circuit.depth() - transpiled.depth(),
                "size_change": transpiled.size() - circuit.size(),
            },
        }
    except Exception as e:
        logger.error(f"Failed to run preset pass manager: {e}")
        return {
            "status": "error",
            "message": f"Failed to run preset pass manager: {str(e)}",
        }


async def run_optimization_passes(
    circuit_qasm: str, iterations: int = 2
) -> Dict[str, Any]:
    """Run optimization passes on a circuit.

    Args:
        circuit_qasm: QASM representation of the circuit
        iterations: Number of optimization iterations

    Returns:
        Optimized circuit information
    """
    try:
        circuit = qasm2.loads(circuit_qasm)

        # Build optimization pipeline
        pm = PassManager()
        for _ in range(iterations):
            pm.append(Optimize1qGates())
            pm.append(InverseCancellation())
            pm.append(CommutativeCancellation())

        pm.append(RemoveDiagonalGatesBeforeMeasure())

        # Get original gate counts
        original_gates: dict[str, int] = {}
        for instruction in circuit.data:
            gate_name = instruction.operation.name
            original_gates[gate_name] = original_gates.get(gate_name, 0) + 1

        # Run optimization
        optimized = pm.run(circuit)

        # Get optimized gate counts
        optimized_gates: dict[str, int] = {}
        for instruction in optimized.data:
            gate_name = instruction.operation.name
            optimized_gates[gate_name] = optimized_gates.get(gate_name, 0) + 1

        return {
            "status": "success",
            "message": f"Ran {iterations} optimization iterations",
            "passes_applied": [
                "Optimize1qGates",
                "InverseCancellation",
                "CommutativeCancellation",
                "RemoveDiagonalGatesBeforeMeasure",
            ],
            "iterations": iterations,
            "original": {
                "depth": circuit.depth(),
                "size": circuit.size(),
                "gate_counts": original_gates,
            },
            "optimized": {
                "depth": optimized.depth(),
                "size": optimized.size(),
                "gate_counts": optimized_gates,
                "qasm": qasm2.dumps(optimized),
            },
            "improvement": {
                "depth_reduction": circuit.depth() - optimized.depth(),
                "depth_reduction_percent": round(
                    (circuit.depth() - optimized.depth()) / circuit.depth() * 100, 2
                )
                if circuit.depth() > 0
                else 0,
                "size_reduction": circuit.size() - optimized.size(),
                "gate_reduction": circuit.size() - optimized.size(),
            },
        }
    except Exception as e:
        logger.error(f"Failed to run optimization passes: {e}")
        return {
            "status": "error",
            "message": f"Failed to run optimization passes: {str(e)}",
        }


async def run_analysis_passes(circuit_qasm: str) -> Dict[str, Any]:
    """Run analysis passes on a circuit to gather metrics.

    Args:
        circuit_qasm: QASM representation of the circuit

    Returns:
        Circuit analysis information
    """
    try:
        circuit = qasm2.loads(circuit_qasm)

        # Build analysis pipeline
        pm = PassManager()
        pm.append(Depth())
        pm.append(Size())
        pm.append(Width())
        pm.append(CountOps())
        pm.append(NumTensorFactors())

        # Run analysis
        pm.run(circuit)

        # Get property set (analysis results)
        property_set = pm.property_set

        # Count gate types
        gate_counts: dict[str, int] = {}
        for instruction in circuit.data:
            gate_name = instruction.operation.name
            gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1

        # Count qubit usage
        qubit_usage = [0] * circuit.num_qubits
        for instruction in circuit.data:
            for qubit in instruction.qubits:
                qubit_usage[circuit.find_bit(qubit).index] += 1

        return {
            "status": "success",
            "message": "Analyzed circuit with multiple passes",
            "passes_applied": [
                "Depth",
                "Size",
                "Width",
                "CountOps",
                "NumTensorFactors",
            ],
            "analysis": {
                "depth": circuit.depth(),
                "size": circuit.size(),
                "width": circuit.num_qubits,
                "num_qubits": circuit.num_qubits,
                "num_clbits": circuit.num_clbits,
                "gate_counts": gate_counts,
                "qubit_usage": qubit_usage,
                "num_tensor_factors": property_set.get("num_tensor_factors", None),
            },
            "statistics": {
                "total_gates": circuit.size(),
                "unique_gate_types": len(gate_counts),
                "avg_qubit_usage": sum(qubit_usage) / len(qubit_usage)
                if qubit_usage
                else 0,
                "max_qubit_usage": max(qubit_usage) if qubit_usage else 0,
                "min_qubit_usage": min(qubit_usage) if qubit_usage else 0,
            },
        }
    except Exception as e:
        logger.error(f"Failed to run analysis passes: {e}")
        return {
            "status": "error",
            "message": f"Failed to run analysis passes: {str(e)}",
        }


async def run_unroll_passes(circuit_qasm: str, basis_gates: str) -> Dict[str, Any]:
    """Run unrolling passes to decompose gates to a basis set.

    Args:
        circuit_qasm: QASM representation of the circuit
        basis_gates: Comma-separated basis gates

    Returns:
        Unrolled circuit information
    """
    try:
        circuit = qasm2.loads(circuit_qasm)

        # Parse basis gates
        basis_gates_list = [g.strip() for g in basis_gates.split(",")]

        # Get original gate counts
        original_gates: dict[str, int] = {}
        for instruction in circuit.data:
            gate_name = instruction.operation.name
            original_gates[gate_name] = original_gates.get(gate_name, 0) + 1

        # Run unrolling using transpile
        unrolled = transpile(
            circuit, basis_gates=basis_gates_list, optimization_level=0
        )

        # Get unrolled gate counts
        unrolled_gates: dict[str, int] = {}
        for instruction in unrolled.data:
            gate_name = instruction.operation.name
            unrolled_gates[gate_name] = unrolled_gates.get(gate_name, 0) + 1

        # Check if all gates are in basis
        all_in_basis = all(g in basis_gates_list for g in unrolled_gates.keys())

        return {
            "status": "success",
            "message": f"Unrolled circuit to basis gates: {basis_gates}",
            "passes_applied": ["transpile with basis_gates"],
            "basis_gates": basis_gates_list,
            "original": {
                "depth": circuit.depth(),
                "size": circuit.size(),
                "gate_counts": original_gates,
            },
            "unrolled": {
                "depth": unrolled.depth(),
                "size": unrolled.size(),
                "gate_counts": unrolled_gates,
                "qasm": qasm2.dumps(unrolled),
                "all_gates_in_basis": all_in_basis,
            },
            "transformation": {
                "depth_change": unrolled.depth() - circuit.depth(),
                "size_change": unrolled.size() - circuit.size(),
                "gate_expansion": unrolled.size() - circuit.size(),
            },
        }
    except Exception as e:
        logger.error(f"Failed to run unroll passes: {e}")
        return {"status": "error", "message": f"Failed to run unroll passes: {str(e)}"}


async def run_combined_passes(
    circuit_qasm: str,
    basis_gates: str,
    optimization_level: int = 2,
    coupling_map_json: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a combined pipeline of unrolling and optimization passes.

    Args:
        circuit_qasm: QASM representation of the circuit
        basis_gates: Comma-separated basis gates
        optimization_level: Number of optimization iterations (0-3)
        coupling_map_json: Optional coupling map JSON

    Returns:
        Processed circuit information
    """
    try:
        circuit = qasm2.loads(circuit_qasm)

        # Parse basis gates
        basis_gates_list = [g.strip() for g in basis_gates.split(",")]

        # Parse coupling map
        if coupling_map_json:
            if coupling_map_json.startswith("linear:"):
                num_qubits = int(coupling_map_json.split(":")[1])
                _ = CouplingMap.from_line(num_qubits)
            elif coupling_map_json.startswith("grid:"):
                dims = coupling_map_json.split(":")[1].split("x")
                rows, cols = int(dims[0]), int(dims[1])
                _ = CouplingMap.from_grid(rows, cols)
            else:
                edges = json.loads(coupling_map_json)
                _ = CouplingMap(edges)

        # Stage 1: Unroll to basis using transpile
        circuit = transpile(circuit, basis_gates=basis_gates_list, optimization_level=0)

        # Build optimization pipeline
        pm = PassManager()

        # Stage 2: Optimize (multiple iterations based on level)
        for _ in range(max(1, optimization_level)):
            pm.append(Optimize1qGates())
            pm.append(InverseCancellation())
            if optimization_level > 1:
                pm.append(CommutativeCancellation())

        # Stage 3: Final cleanup
        pm.append(RemoveDiagonalGatesBeforeMeasure())

        # Run optimization pipeline
        processed = pm.run(circuit)

        # Get gate counts
        processed_gates: dict[str, int] = {}
        for instruction in processed.data:
            gate_name = instruction.operation.name
            processed_gates[gate_name] = processed_gates.get(gate_name, 0) + 1

        return {
            "status": "success",
            "message": f"Ran combined unroll + optimization pipeline (level {optimization_level})",
            "pipeline_stages": ["Unroll", "Optimize", "Cleanup"],
            "passes_applied": [
                "transpile with basis_gates",
                "Optimize1qGates",
                "InverseCancellation",
                "CommutativeCancellation" if optimization_level > 1 else None,
                "RemoveDiagonalGatesBeforeMeasure",
            ],
            "configuration": {
                "basis_gates": basis_gates_list,
                "optimization_level": optimization_level,
                "coupling_map": coupling_map_json,
            },
            "original": {
                "depth": circuit.depth(),
                "size": circuit.size(),
            },
            "processed": {
                "depth": processed.depth(),
                "size": processed.size(),
                "gate_counts": processed_gates,
                "qasm": qasm2.dumps(processed),
            },
            "improvement": {
                "depth_reduction": circuit.depth() - processed.depth(),
                "depth_reduction_percent": round(
                    (circuit.depth() - processed.depth()) / circuit.depth() * 100, 2
                )
                if circuit.depth() > 0
                else 0,
                "size_change": processed.size() - circuit.size(),
            },
        }
    except Exception as e:
        logger.error(f"Failed to run combined passes: {e}")
        return {
            "status": "error",
            "message": f"Failed to run combined passes: {str(e)}",
        }


# Assisted by watsonx Code Assistant
