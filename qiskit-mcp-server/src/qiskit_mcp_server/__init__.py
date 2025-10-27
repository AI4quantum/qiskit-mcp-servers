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

"""Qiskit MCP Server - Model Context Protocol server for Qiskit SDK.

This package provides a comprehensive MCP server that enables AI assistants
to interact with Qiskit quantum computing functions including:
- Quantum circuit creation and manipulation
- Advanced circuit operations (composition, parameterization, decomposition)
- Circuit library (QFT, Grover, variational ansätze, Pauli evolution, oracles)
- Variational circuits (EfficientSU2, RealAmplitudes, TwoLocal, NLocal, ParameterVector)
- Quantum machine learning (Pauli/Z/ZZ feature maps, QAOA ansatz)
- Boolean logic gates (AND, OR, XOR for quantum search)
- Advanced algorithms (Phase estimation, IQP, hidden linear function, VQE, QAOA)
- Clifford & stabilizer operations (random Clifford, stabilizer states, Pauli tracking)
- Dynamic circuits (mid-circuit measurement, conditional gates, quantum teleportation)
- Advanced synthesis (single/two-qubit decomposition, unitary synthesis, optimization)
- Circuit utilities (inverse, copy, to_gate, to_instruction)
- QASM file I/O (load/save QASM 2.0 and 3.0)
- Circuit converters (DAG conversion, decomposition)
- Enhanced transpilation (backend, coupling map, layout, routing strategies)
- PassManager support (preset, optimization, analysis, unroll, combined passes)
- Quantum information operations (density matrices, fidelity, entropy)
- Qiskit Primitives (Sampler, Estimator)
- Result processing (marginal counts, filtering, expectation values)
- Backend execution (local simulators, IBM Quantum hardware submission)
- Job management (submit, retrieve, cancel, status checking)
- Algorithm solvers (VQE, QAOA, parameter optimization)
- Error mitigation (measurement calibration, ZNE, readout errors, dynamical decoupling)
- Simulation backends (Aer, statevector, noisy simulation)
- Noise models (depolarizing, thermal relaxation, damping)
- State preparation (W state, GHZ state, Dicke state, superposition)
- Quantum state tomography
- Circuit equivalence checking
- Circuit analysis and optimization
- OpenQASM 3.0 support
- Circuit visualization (text, matplotlib, Bloch sphere, Q-sphere, Hinton, city plot)
- Measurement visualization (histograms, distributions)

Total MCP Tools: 146
Coverage: 90%+
"""

from . import server
from . import qiskit_sdk
from . import quantum_info
from . import primitives
from . import advanced_circuits
from . import noise_and_mitigation
from . import state_prep_and_tomography
from . import circuit_equivalence
from . import circuit_utilities
from . import transpilation
from . import pass_manager
from . import sync

__version__ = "0.1.0"


def main():
    """Main entry point for the package."""
    server.mcp.run(transport="stdio")


# Expose important items at package level
__all__ = [
    "main",
    "server",
    "qiskit_sdk",
    "quantum_info",
    "primitives",
    "advanced_circuits",
    "noise_and_mitigation",
    "state_prep_and_tomography",
    "circuit_equivalence",
    "circuit_utilities",
    "transpilation",
    "pass_manager",
    "sync",
    "__version__",
]
