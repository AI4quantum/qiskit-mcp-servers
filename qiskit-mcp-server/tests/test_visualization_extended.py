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

"""Tests for extended visualization functions."""

import pytest
import json
from qiskit import QuantumCircuit, qasm2
from qiskit_mcp_server.visualization_extended import (
    plot_bloch_multivector,
    plot_state_qsphere,
    plot_state_hinton,
    plot_state_city,
    plot_state_paulivec,
    plot_histogram,
    plot_distribution,
)


@pytest.fixture
def simple_circuit_qasm():
    """Create a simple circuit QASM for testing."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qasm2.dumps(qc)


@pytest.fixture
def single_qubit_qasm():
    """Create single qubit circuit."""
    qc = QuantumCircuit(1)
    qc.h(0)
    return qasm2.dumps(qc)


@pytest.fixture
def sample_counts_json():
    """Sample measurement counts."""
    return json.dumps({"00": 512, "11": 512})


@pytest.mark.asyncio
async def test_plot_bloch_multivector(single_qubit_qasm):
    """Test Bloch multivector visualization."""
    result = await plot_bloch_multivector(single_qubit_qasm)

    if result["status"] == "error" and "not available" in result["message"]:
        pytest.skip("Visualization dependencies not installed")

    assert result["status"] == "success"
    assert result["format"] == "bloch_multivector"
    assert "image_base64" in result
    assert result["num_qubits"] == 1


@pytest.mark.asyncio
async def test_plot_state_qsphere(simple_circuit_qasm):
    """Test Q-sphere visualization."""
    result = await plot_state_qsphere(simple_circuit_qasm)

    if result["status"] == "error" and "not available" in result["message"]:
        pytest.skip("Visualization dependencies not installed")

    assert result["status"] == "success"
    assert "image_base64" in result


@pytest.mark.asyncio
async def test_plot_state_hinton(simple_circuit_qasm):
    """Test Hinton diagram visualization."""
    result = await plot_state_hinton(simple_circuit_qasm)

    if result["status"] == "error" and "not available" in result["message"]:
        pytest.skip("Visualization dependencies not installed")

    assert result["status"] == "success"
    assert "image_base64" in result


@pytest.mark.asyncio
async def test_plot_state_city(simple_circuit_qasm):
    """Test city plot visualization."""
    result = await plot_state_city(simple_circuit_qasm)

    if result["status"] == "error" and "not available" in result["message"]:
        pytest.skip("Visualization dependencies not installed")

    assert result["status"] == "success"
    assert "image_base64" in result


@pytest.mark.asyncio
async def test_plot_state_paulivec(simple_circuit_qasm):
    """Test Pauli vector visualization."""
    result = await plot_state_paulivec(simple_circuit_qasm)

    if result["status"] == "error" and "not available" in result["message"]:
        pytest.skip("Visualization dependencies not installed")

    assert result["status"] == "success"
    assert "image_base64" in result


@pytest.mark.asyncio
async def test_plot_histogram(sample_counts_json):
    """Test histogram plot."""
    result = await plot_histogram(sample_counts_json)

    if result["status"] == "error" and "not available" in result["message"]:
        pytest.skip("Visualization dependencies not installed")

    assert result["status"] == "success"
    assert "image_base64" in result


@pytest.mark.asyncio
async def test_plot_distribution(sample_counts_json):
    """Test probability distribution plot."""
    result = await plot_distribution(sample_counts_json)

    if result["status"] == "error" and "not available" in result["message"]:
        pytest.skip("Visualization dependencies not installed")

    assert result["status"] == "success"
    assert "image_base64" in result


# Assisted by watsonx Code Assistant
