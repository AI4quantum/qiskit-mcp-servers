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

"""Tests for enhanced transpilation module."""

import pytest
from qiskit import QuantumCircuit, qasm2
from qiskit_mcp_server.transpilation import (
    transpile_with_coupling_map,
    transpile_with_layout_strategy,
    compare_transpilation_strategies,
    transpile_for_basis_gates,
)


@pytest.mark.asyncio
async def test_transpile_with_linear_coupling_map():
    """Test transpiling with a linear coupling map."""
    # Create a simple circuit
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qasm = qasm2.dumps(qc)

    result = await transpile_with_coupling_map(
        circuit_qasm=qasm, coupling_map_json="linear:3", optimization_level=1
    )

    assert result["status"] == "success"
    assert result["coupling_map"]["description"] == "linear:3"
    assert "transpiled" in result
    assert "qasm" in result["transpiled"]


@pytest.mark.asyncio
async def test_transpile_with_grid_coupling_map():
    """Test transpiling with a grid coupling map."""
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(2, 3)
    qasm = qasm2.dumps(qc)

    result = await transpile_with_coupling_map(
        circuit_qasm=qasm, coupling_map_json="grid:2x2", optimization_level=1
    )

    assert result["status"] == "success"
    assert result["coupling_map"]["description"] == "grid:2x2"


@pytest.mark.asyncio
async def test_transpile_with_custom_coupling_map():
    """Test transpiling with a custom coupling map."""
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qasm = qasm2.dumps(qc)

    # Custom coupling map: 0-1, 1-2
    coupling_map = "[[0,1], [1,0], [1,2], [2,1]]"

    result = await transpile_with_coupling_map(
        circuit_qasm=qasm, coupling_map_json=coupling_map, optimization_level=1
    )

    assert result["status"] == "success"
    assert "transpiled" in result


@pytest.mark.asyncio
async def test_transpile_with_initial_layout():
    """Test transpiling with an initial layout."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qasm = qasm2.dumps(qc)

    # Map virtual qubit 0 to physical qubit 1, and virtual 1 to physical 0
    initial_layout = '{"0": 1, "1": 0}'

    result = await transpile_with_coupling_map(
        circuit_qasm=qasm,
        coupling_map_json="linear:3",
        initial_layout_json=initial_layout,
        optimization_level=1,
    )

    assert result["status"] == "success"
    assert result["transpilation"]["initial_layout"] == initial_layout


@pytest.mark.asyncio
async def test_transpile_with_sabre_layout():
    """Test transpiling with SABRE layout method."""
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(0, 2)
    qasm = qasm2.dumps(qc)

    result = await transpile_with_layout_strategy(
        circuit_qasm=qasm,
        layout_method="sabre",
        routing_method="sabre",
        optimization_level=2,
    )

    assert result["status"] == "success"
    assert result["strategy"]["layout_method"] == "sabre"
    assert result["strategy"]["routing_method"] == "sabre"


@pytest.mark.asyncio
async def test_transpile_with_dense_layout():
    """Test transpiling with dense layout method."""
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qasm = qasm2.dumps(qc)

    result = await transpile_with_layout_strategy(
        circuit_qasm=qasm,
        layout_method="dense",
        routing_method="basic",
        coupling_map_json="linear:5",
    )

    assert result["status"] == "success"
    assert result["strategy"]["layout_method"] == "dense"


@pytest.mark.asyncio
async def test_transpile_with_seed():
    """Test that transpilation with seed is reproducible."""
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qasm = qasm2.dumps(qc)

    # Transpile twice with same seed
    result1 = await transpile_with_layout_strategy(
        circuit_qasm=qasm,
        layout_method="sabre",
        routing_method="sabre",
        seed_transpiler=42,
        coupling_map_json="linear:5",
    )

    result2 = await transpile_with_layout_strategy(
        circuit_qasm=qasm,
        layout_method="sabre",
        routing_method="sabre",
        seed_transpiler=42,
        coupling_map_json="linear:5",
    )

    assert result1["status"] == "success"
    assert result2["status"] == "success"
    # Results should be identical with same seed
    assert result1["transpiled"]["qasm"] == result2["transpiled"]["qasm"]


@pytest.mark.asyncio
async def test_compare_transpilation_strategies():
    """Test comparing different transpilation strategies."""
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.h(2)
    qasm = qasm2.dumps(qc)

    result = await compare_transpilation_strategies(
        circuit_qasm=qasm,
        optimization_level=2,
        coupling_map_json="linear:5",
        seed_transpiler=42,
    )

    assert result["status"] == "success"
    assert "results" in result
    assert len(result["results"]) > 0
    assert "best_strategy" in result


@pytest.mark.asyncio
async def test_transpile_for_basis_gates():
    """Test transpiling to specific basis gates."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.t(1)
    qasm = qasm2.dumps(qc)

    # Transpile to sx, x, rz, cx basis
    result = await transpile_for_basis_gates(
        circuit_qasm=qasm, basis_gates="sx,x,rz,cx", optimization_level=1
    )

    assert result["status"] == "success"
    assert result["basis_gates"] == ["sx", "x", "rz", "cx"]
    assert "gate_counts" in result["transpiled"]
    # All gates should be in the specified basis
    assert result["transformation"]["all_gates_in_basis"]


@pytest.mark.asyncio
async def test_transpile_for_u3_cx_basis():
    """Test transpiling to u3, cx basis gates."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.s(0)
    qc.t(1)
    qc.cx(0, 1)
    qasm = qasm2.dumps(qc)

    result = await transpile_for_basis_gates(
        circuit_qasm=qasm, basis_gates="u3,cx", optimization_level=2
    )

    assert result["status"] == "success"
    assert "u3" in result["basis_gates"]
    assert "cx" in result["basis_gates"]


@pytest.mark.asyncio
async def test_transpile_with_multiple_parameters():
    """Test transpiling with multiple parameters combined."""
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.h(2)
    qasm = qasm2.dumps(qc)

    result = await transpile_with_layout_strategy(
        circuit_qasm=qasm,
        layout_method="sabre",
        routing_method="sabre",
        optimization_level=3,
        basis_gates="sx,x,rz,cx",
        coupling_map_json="linear:5",
        seed_transpiler=123,
    )

    assert result["status"] == "success"
    assert result["strategy"]["layout_method"] == "sabre"
    assert result["strategy"]["routing_method"] == "sabre"
    assert result["strategy"]["optimization_level"] == 3
    assert result["strategy"]["seed_transpiler"] == 123


@pytest.mark.asyncio
async def test_transpile_metrics():
    """Test that transpilation returns useful metrics."""
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.h(2)
    qasm = qasm2.dumps(qc)

    result = await transpile_with_layout_strategy(
        circuit_qasm=qasm,
        layout_method="sabre",
        routing_method="sabre",
        optimization_level=2,
    )

    assert result["status"] == "success"
    assert "metrics" in result
    assert "depth_reduction" in result["metrics"]
    assert "depth_reduction_percent" in result["metrics"]
    assert "size_change" in result["metrics"]


# Assisted by watsonx Code Assistant
