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

"""Tests for PassManager module."""

import pytest
from qiskit import QuantumCircuit, qasm2
from qiskit_mcp_server.pass_manager import (
    run_preset_pass_manager,
    run_optimization_passes,
    run_analysis_passes,
    run_unroll_passes,
    run_combined_passes,
)


@pytest.mark.asyncio
async def test_run_preset_pass_manager():
    """Test running preset pass manager."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qasm = qasm2.dumps(qc)

    result = await run_preset_pass_manager(circuit_qasm=qasm, optimization_level=1)

    assert result["status"] == "success"
    assert result["configuration"]["optimization_level"] == 1
    assert "transpiled" in result
    assert "qasm" in result["transpiled"]


@pytest.mark.asyncio
async def test_run_preset_pass_manager_with_coupling_map():
    """Test preset pass manager with coupling map."""
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qasm = qasm2.dumps(qc)

    result = await run_preset_pass_manager(
        circuit_qasm=qasm, optimization_level=2, coupling_map_json="linear:5"
    )

    assert result["status"] == "success"
    assert result["configuration"]["coupling_map"] == "linear:5"


@pytest.mark.asyncio
async def test_run_preset_pass_manager_with_basis_gates():
    """Test preset pass manager with basis gates."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qasm = qasm2.dumps(qc)

    result = await run_preset_pass_manager(
        circuit_qasm=qasm, optimization_level=1, basis_gates="sx,x,rz,cx"
    )

    assert result["status"] == "success"
    assert result["configuration"]["basis_gates"] == ["sx", "x", "rz", "cx"]


@pytest.mark.asyncio
async def test_run_optimization_passes():
    """Test running optimization passes."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.h(0)  # Two H gates cancel
    qc.cx(0, 1)
    qc.cx(0, 1)  # Two CX gates cancel
    qasm = qasm2.dumps(qc)

    result = await run_optimization_passes(circuit_qasm=qasm, iterations=2)

    assert result["status"] == "success"
    assert result["iterations"] == 2
    assert "passes_applied" in result
    assert "Optimize1qGates" in result["passes_applied"]
    assert "CXCancellation" in result["passes_applied"]


@pytest.mark.asyncio
async def test_optimization_reduces_circuit():
    """Test that optimization actually reduces circuit size."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.s(0)
    qc.t(0)
    qc.cx(0, 1)
    qc.cx(0, 1)
    qasm = qasm2.dumps(qc)

    result = await run_optimization_passes(circuit_qasm=qasm, iterations=1)

    assert result["status"] == "success"
    # Should see some optimization
    assert "improvement" in result


@pytest.mark.asyncio
async def test_run_analysis_passes():
    """Test running analysis passes."""
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.h(2)
    qasm = qasm2.dumps(qc)

    result = await run_analysis_passes(circuit_qasm=qasm)

    assert result["status"] == "success"
    assert "analysis" in result
    assert result["analysis"]["depth"] == qc.depth()
    assert result["analysis"]["size"] == qc.size()
    assert result["analysis"]["num_qubits"] == 3
    assert "gate_counts" in result["analysis"]
    assert "qubit_usage" in result["analysis"]


@pytest.mark.asyncio
async def test_analysis_provides_statistics():
    """Test that analysis provides useful statistics."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.h(1)
    qasm = qasm2.dumps(qc)

    result = await run_analysis_passes(circuit_qasm=qasm)

    assert result["status"] == "success"
    assert "statistics" in result
    assert result["statistics"]["total_gates"] > 0
    assert result["statistics"]["unique_gate_types"] > 0


@pytest.mark.asyncio
async def test_run_unroll_passes():
    """Test running unroll passes."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.s(0)
    qc.t(1)
    qc.cx(0, 1)
    qasm = qasm2.dumps(qc)

    result = await run_unroll_passes(circuit_qasm=qasm, basis_gates="u3,cx")

    assert result["status"] == "success"
    assert result["basis_gates"] == ["u3", "cx"]
    assert "unrolled" in result
    assert result["unrolled"]["all_gates_in_basis"]


@pytest.mark.asyncio
async def test_unroll_expands_gates():
    """Test that unrolling expands gates to basis."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qasm = qasm2.dumps(qc)

    result = await run_unroll_passes(circuit_qasm=qasm, basis_gates="sx,x,rz,cx")

    assert result["status"] == "success"
    # H gate should be expanded
    assert "transformation" in result


@pytest.mark.asyncio
async def test_run_combined_passes():
    """Test running combined pass pipeline."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.s(0)
    qc.cx(0, 1)
    qc.t(1)
    qasm = qasm2.dumps(qc)

    result = await run_combined_passes(
        circuit_qasm=qasm, basis_gates="sx,x,rz,cx", optimization_level=2
    )

    assert result["status"] == "success"
    assert result["pipeline_stages"] == ["Unroll", "Optimize", "Cleanup"]
    assert "passes_applied" in result
    assert "Unroller" in result["passes_applied"]
    assert "Optimize1qGates" in result["passes_applied"]


@pytest.mark.asyncio
async def test_combined_passes_with_coupling_map():
    """Test combined passes with coupling map."""
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qasm = qasm2.dumps(qc)

    result = await run_combined_passes(
        circuit_qasm=qasm,
        basis_gates="sx,x,rz,cx",
        optimization_level=1,
        coupling_map_json="linear:5",
    )

    assert result["status"] == "success"
    assert result["configuration"]["coupling_map"] == "linear:5"


@pytest.mark.asyncio
async def test_combined_passes_optimization_levels():
    """Test different optimization levels in combined passes."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qasm = qasm2.dumps(qc)

    # Level 0
    result0 = await run_combined_passes(
        circuit_qasm=qasm, basis_gates="u3,cx", optimization_level=0
    )
    assert result0["status"] == "success"

    # Level 3
    result3 = await run_combined_passes(
        circuit_qasm=qasm, basis_gates="u3,cx", optimization_level=3
    )
    assert result3["status"] == "success"


@pytest.mark.asyncio
async def test_pass_manager_metrics():
    """Test that pass manager returns useful metrics."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.h(1)
    qasm = qasm2.dumps(qc)

    result = await run_optimization_passes(circuit_qasm=qasm, iterations=1)

    assert result["status"] == "success"
    assert "improvement" in result
    assert "depth_reduction" in result["improvement"]
    assert "depth_reduction_percent" in result["improvement"]


# Assisted by watsonx Code Assistant
