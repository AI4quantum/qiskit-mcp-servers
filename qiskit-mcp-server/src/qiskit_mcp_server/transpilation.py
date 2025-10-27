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

"""Enhanced transpilation functions with advanced parameter support."""

import logging
import json
from typing import Any, Dict, Optional

from qiskit import qasm2, transpile
from qiskit.transpiler import CouplingMap, Layout

logger = logging.getLogger(__name__)


async def transpile_with_backend(
    circuit_qasm: str,
    backend_name: str,
    optimization_level: int = 1,
    seed_transpiler: Optional[int] = None,
    layout_method: Optional[str] = None,
    routing_method: Optional[str] = None,
) -> Dict[str, Any]:
    """Transpile a circuit for a specific backend.

    Args:
        circuit_qasm: QASM representation of the circuit
        backend_name: Name of the target backend (e.g., 'ibm_brisbane', 'ibm_kyoto')
        optimization_level: Optimization level (0-3)
        seed_transpiler: Random seed for reproducibility
        layout_method: Layout selection method ('trivial', 'dense', 'sabre', None)
        routing_method: Routing method ('basic', 'lookahead', 'stochastic', 'sabre', None)

    Returns:
        Transpiled circuit information
    """
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService

        circuit = qasm2.loads(circuit_qasm)

        # Get backend
        service = QiskitRuntimeService()
        backend = service.backend(backend_name)

        # Validate optimization level
        if optimization_level not in [0, 1, 2, 3]:
            optimization_level = 1

        # Transpile with backend
        transpiled = transpile(
            circuit,
            backend=backend,
            optimization_level=optimization_level,
            seed_transpiler=seed_transpiler,
            layout_method=layout_method,
            routing_method=routing_method,
        )

        return {
            "status": "success",
            "message": f"Circuit transpiled for backend '{backend_name}'",
            "backend": {
                "name": backend.name,
                "num_qubits": backend.num_qubits,
                "version": str(backend.version),
            },
            "transpilation": {
                "optimization_level": optimization_level,
                "seed_transpiler": seed_transpiler,
                "layout_method": layout_method,
                "routing_method": routing_method,
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
    except ImportError:
        return {
            "status": "error",
            "message": "qiskit-ibm-runtime not installed. Install with: pip install qiskit-ibm-runtime",
        }
    except Exception as e:
        logger.error(f"Failed to transpile with backend: {e}")
        return {
            "status": "error",
            "message": f"Failed to transpile with backend: {str(e)}",
        }


async def transpile_with_coupling_map(
    circuit_qasm: str,
    coupling_map_json: str,
    optimization_level: int = 1,
    initial_layout_json: Optional[str] = None,
    layout_method: Optional[str] = None,
    routing_method: Optional[str] = None,
    seed_transpiler: Optional[int] = None,
) -> Dict[str, Any]:
    """Transpile a circuit with a custom coupling map.

    Args:
        circuit_qasm: QASM representation of the circuit
        coupling_map_json: JSON array of edges [[0,1], [1,2], ...] or "linear:N" or "grid:NxM"
        optimization_level: Optimization level (0-3)
        initial_layout_json: JSON dict mapping virtual to physical qubits {0: 5, 1: 10, ...}
        layout_method: Layout selection method ('trivial', 'dense', 'sabre', None)
        routing_method: Routing method ('basic', 'lookahead', 'stochastic', 'sabre', None)
        seed_transpiler: Random seed for reproducibility

    Returns:
        Transpiled circuit information
    """
    try:
        circuit = qasm2.loads(circuit_qasm)

        # Parse coupling map
        coupling_map = None
        if coupling_map_json:
            if coupling_map_json.startswith("linear:"):
                # Linear coupling: 0-1-2-3-...
                num_qubits = int(coupling_map_json.split(":")[1])
                coupling_map = CouplingMap.from_line(num_qubits)
            elif coupling_map_json.startswith("grid:"):
                # Grid coupling: NxM
                dims = coupling_map_json.split(":")[1].split("x")
                rows, cols = int(dims[0]), int(dims[1])
                coupling_map = CouplingMap.from_grid(rows, cols)
            else:
                # Custom coupling map from JSON
                edges = json.loads(coupling_map_json)
                coupling_map = CouplingMap(edges)

        # Parse initial layout
        initial_layout = None
        if initial_layout_json:
            layout_dict = json.loads(initial_layout_json)
            # Convert string keys to int if needed
            layout_dict = {
                int(k) if isinstance(k, str) else k: v for k, v in layout_dict.items()
            }
            initial_layout = Layout(layout_dict)

        # Validate optimization level
        if optimization_level not in [0, 1, 2, 3]:
            optimization_level = 1

        # Transpile with coupling map
        transpiled = transpile(
            circuit,
            coupling_map=coupling_map,
            initial_layout=initial_layout,
            optimization_level=optimization_level,
            layout_method=layout_method,
            routing_method=routing_method,
            seed_transpiler=seed_transpiler,
        )

        return {
            "status": "success",
            "message": "Circuit transpiled with custom coupling map",
            "coupling_map": {
                "description": coupling_map_json,
                "num_qubits": coupling_map.size() if coupling_map else None,
                "edges": coupling_map.get_edges() if coupling_map else None,
            },
            "transpilation": {
                "optimization_level": optimization_level,
                "layout_method": layout_method,
                "routing_method": routing_method,
                "seed_transpiler": seed_transpiler,
                "initial_layout": initial_layout_json,
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
        logger.error(f"Failed to transpile with coupling map: {e}")
        return {
            "status": "error",
            "message": f"Failed to transpile with coupling map: {str(e)}",
        }


async def transpile_with_layout_strategy(
    circuit_qasm: str,
    layout_method: str,
    routing_method: str = "sabre",
    optimization_level: int = 2,
    basis_gates: Optional[str] = None,
    coupling_map_json: Optional[str] = None,
    seed_transpiler: Optional[int] = None,
) -> Dict[str, Any]:
    """Transpile a circuit with specific layout and routing strategies.

    Args:
        circuit_qasm: QASM representation of the circuit
        layout_method: Layout method ('trivial', 'dense', 'sabre')
        routing_method: Routing method ('basic', 'lookahead', 'stochastic', 'sabre')
        optimization_level: Optimization level (0-3)
        basis_gates: Comma-separated basis gates
        coupling_map_json: JSON array of edges or "linear:N" or "grid:NxM"
        seed_transpiler: Random seed for reproducibility

    Returns:
        Transpiled circuit with strategy details
    """
    try:
        circuit = qasm2.loads(circuit_qasm)

        # Parse basis gates
        basis_gates_list = None
        if basis_gates:
            basis_gates_list = [g.strip() for g in basis_gates.split(",")]

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

        # Validate optimization level
        if optimization_level not in [0, 1, 2, 3]:
            optimization_level = 2

        # Transpile with strategies
        transpiled = transpile(
            circuit,
            layout_method=layout_method,
            routing_method=routing_method,
            optimization_level=optimization_level,
            basis_gates=basis_gates_list,
            coupling_map=coupling_map,
            seed_transpiler=seed_transpiler,
        )

        return {
            "status": "success",
            "message": f"Circuit transpiled with {layout_method} layout and {routing_method} routing",
            "strategy": {
                "layout_method": layout_method,
                "routing_method": routing_method,
                "optimization_level": optimization_level,
                "seed_transpiler": seed_transpiler,
            },
            "configuration": {
                "basis_gates": basis_gates_list,
                "coupling_map": coupling_map_json,
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
            "metrics": {
                "depth_reduction": circuit.depth() - transpiled.depth(),
                "depth_reduction_percent": round(
                    (circuit.depth() - transpiled.depth()) / circuit.depth() * 100, 2
                )
                if circuit.depth() > 0
                else 0,
                "size_change": transpiled.size() - circuit.size(),
            },
        }
    except Exception as e:
        logger.error(f"Failed to transpile with layout strategy: {e}")
        return {
            "status": "error",
            "message": f"Failed to transpile with layout strategy: {str(e)}",
        }


async def compare_transpilation_strategies(
    circuit_qasm: str,
    optimization_level: int = 2,
    coupling_map_json: Optional[str] = None,
    seed_transpiler: Optional[int] = None,
) -> Dict[str, Any]:
    """Compare different transpilation strategies on the same circuit.

    Args:
        circuit_qasm: QASM representation of the circuit
        optimization_level: Optimization level (0-3)
        coupling_map_json: JSON array of edges or "linear:N" or "grid:NxM"
        seed_transpiler: Random seed for reproducibility

    Returns:
        Comparison of transpilation strategies
    """
    try:
        circuit = qasm2.loads(circuit_qasm)

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

        strategies = [
            ("trivial", "basic"),
            ("dense", "basic"),
            ("sabre", "sabre"),
            ("dense", "stochastic"),
        ]

        results = []
        for layout_method, routing_method in strategies:
            try:
                transpiled = transpile(
                    circuit,
                    layout_method=layout_method,
                    routing_method=routing_method,
                    optimization_level=optimization_level,
                    coupling_map=coupling_map,
                    seed_transpiler=seed_transpiler,
                )

                results.append(
                    {
                        "strategy": f"{layout_method}+{routing_method}",
                        "layout_method": layout_method,
                        "routing_method": routing_method,
                        "depth": transpiled.depth(),
                        "size": transpiled.size(),
                        "depth_reduction": circuit.depth() - transpiled.depth(),
                        "qasm": qasm2.dumps(transpiled),
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "strategy": f"{layout_method}+{routing_method}",
                        "layout_method": layout_method,
                        "routing_method": routing_method,
                        "error": str(e),
                    }
                )

        # Find best strategy by depth
        successful_results = [r for r in results if "error" not in r]
        if successful_results:
            best = min(successful_results, key=lambda x: x["depth"])
            best_strategy = best["strategy"]
        else:
            best_strategy = None

        return {
            "status": "success",
            "message": f"Compared {len(strategies)} transpilation strategies",
            "original": {
                "depth": circuit.depth(),
                "size": circuit.size(),
                "num_qubits": circuit.num_qubits,
            },
            "configuration": {
                "optimization_level": optimization_level,
                "coupling_map": coupling_map_json,
                "seed_transpiler": seed_transpiler,
            },
            "strategies_tested": len(strategies),
            "best_strategy": best_strategy,
            "results": results,
        }
    except Exception as e:
        logger.error(f"Failed to compare strategies: {e}")
        return {"status": "error", "message": f"Failed to compare strategies: {str(e)}"}


async def transpile_for_basis_gates(
    circuit_qasm: str,
    basis_gates: str,
    optimization_level: int = 1,
    seed_transpiler: Optional[int] = None,
) -> Dict[str, Any]:
    """Transpile a circuit to a specific basis gate set.

    Args:
        circuit_qasm: QASM representation of the circuit
        basis_gates: Comma-separated basis gates (e.g., "u1,u2,u3,cx" or "sx,x,rz,cx")
        optimization_level: Optimization level (0-3)
        seed_transpiler: Random seed for reproducibility

    Returns:
        Transpiled circuit information
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

        # Transpile
        transpiled = transpile(
            circuit,
            basis_gates=basis_gates_list,
            optimization_level=optimization_level,
            seed_transpiler=seed_transpiler,
        )

        # Get transpiled gate counts
        transpiled_gates: dict[str, int] = {}
        for instruction in transpiled.data:
            gate_name = instruction.operation.name
            transpiled_gates[gate_name] = transpiled_gates.get(gate_name, 0) + 1

        return {
            "status": "success",
            "message": f"Circuit transpiled to basis gates: {basis_gates}",
            "basis_gates": basis_gates_list,
            "original": {
                "depth": circuit.depth(),
                "size": circuit.size(),
                "gate_counts": original_gates,
            },
            "transpiled": {
                "depth": transpiled.depth(),
                "size": transpiled.size(),
                "gate_counts": transpiled_gates,
                "qasm": qasm2.dumps(transpiled),
            },
            "transformation": {
                "depth_change": transpiled.depth() - circuit.depth(),
                "size_change": transpiled.size() - circuit.size(),
                "all_gates_in_basis": all(
                    g in basis_gates_list for g in transpiled_gates.keys()
                ),
            },
        }
    except Exception as e:
        logger.error(f"Failed to transpile for basis gates: {e}")
        return {
            "status": "error",
            "message": f"Failed to transpile for basis gates: {str(e)}",
        }


async def get_available_backends() -> Dict[str, Any]:
    """Get list of available IBM Quantum backends.

    Returns:
        List of available backends with their properties
    """
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService

        service = QiskitRuntimeService()
        backends = service.backends()

        backend_list = []
        for backend in backends:
            backend_list.append(
                {
                    "name": backend.name,
                    "num_qubits": backend.num_qubits,
                    "version": str(backend.version),
                    "operational": backend.operational,
                    "simulator": backend.simulator,
                }
            )

        return {
            "status": "success",
            "message": f"Found {len(backend_list)} available backends",
            "backends": backend_list,
        }
    except ImportError:
        return {
            "status": "error",
            "message": "qiskit-ibm-runtime not installed. Install with: pip install qiskit-ibm-runtime",
        }
    except Exception as e:
        logger.error(f"Failed to get backends: {e}")
        return {"status": "error", "message": f"Failed to get backends: {str(e)}"}


# Assisted by watsonx Code Assistant
