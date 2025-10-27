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

"""Tests for extended circuit library functions."""

import pytest
from qiskit_mcp_server.circuit_library_extended import (
    create_two_local_circuit,
    create_n_local_circuit,
    create_pauli_feature_map,
    create_z_feature_map,
    create_zz_feature_map,
    create_qaoa_ansatz,
    create_and_gate,
    create_or_gate,
    create_xor_gate,
    create_hidden_linear_function,
    create_iqp_circuit,
    create_phase_estimation_circuit,
)


# TwoLocal and NLocal Tests
@pytest.mark.asyncio
async def test_create_two_local_circuit_basic():
    """Test basic TwoLocal circuit creation."""
    result = await create_two_local_circuit(num_qubits=3, reps=2)

    assert result["status"] == "success"
    assert result["circuit"]["name"] == "TwoLocal"
    assert result["circuit"]["num_qubits"] == 3
    assert "qasm" in result["circuit"]


@pytest.mark.asyncio
async def test_create_two_local_circuit_custom_blocks():
    """Test TwoLocal with custom rotation blocks."""
    result = await create_two_local_circuit(
        num_qubits=2, rotation_blocks="rz", entanglement_blocks="cz", reps=1
    )

    assert result["status"] == "success"
    assert result["circuit"]["num_qubits"] == 2


@pytest.mark.asyncio
async def test_create_n_local_circuit():
    """Test NLocal circuit creation."""
    result = await create_n_local_circuit(
        num_qubits=4, num_qubits_entanglement=3, reps=2
    )

    assert result["status"] == "success"
    assert result["circuit"]["num_qubits"] == 4


# Feature Map Tests
@pytest.mark.asyncio
async def test_create_pauli_feature_map():
    """Test Pauli feature map creation."""
    result = await create_pauli_feature_map(feature_dimension=3, reps=2)

    assert result["status"] == "success"
    assert result["feature_dimension"] == 3
    assert "circuit" in result
    assert "qasm" in result["circuit"]


@pytest.mark.asyncio
async def test_create_z_feature_map():
    """Test Z feature map creation."""
    result = await create_z_feature_map(feature_dimension=2, reps=1)

    assert result["status"] == "success"
    assert result["feature_dimension"] == 2
    assert result["encoding_type"] == "z"


@pytest.mark.asyncio
async def test_create_zz_feature_map():
    """Test ZZ feature map creation."""
    result = await create_zz_feature_map(
        feature_dimension=3, reps=2, entanglement="linear"
    )

    assert result["status"] == "success"
    assert result["feature_dimension"] == 3
    assert result["encoding_type"] == "zz"
    assert result["entanglement"] == "linear"


@pytest.mark.asyncio
async def test_create_zz_feature_map_full_entanglement():
    """Test ZZ feature map with full entanglement."""
    result = await create_zz_feature_map(
        feature_dimension=4, reps=1, entanglement="full"
    )

    assert result["status"] == "success"
    assert result["entanglement"] == "full"


# QAOA Test
@pytest.mark.asyncio
async def test_create_qaoa_ansatz():
    """Test QAOA ansatz creation."""
    cost_operator = "ZIZI"
    result = await create_qaoa_ansatz(cost_operator=cost_operator, reps=2)

    assert result["status"] == "success"
    assert "circuit" in result
    assert result["reps"] == 2


# Boolean Logic Gates
@pytest.mark.asyncio
async def test_create_and_gate():
    """Test AND gate creation."""
    result = await create_and_gate(num_variable_qubits=2)

    assert result["status"] == "success"
    assert result["gate_type"] == "AND"
    assert result["num_variable_qubits"] == 2


@pytest.mark.asyncio
async def test_create_or_gate():
    """Test OR gate creation."""
    result = await create_or_gate(num_variable_qubits=3)

    assert result["status"] == "success"
    assert result["gate_type"] == "OR"
    assert result["num_variable_qubits"] == 3


@pytest.mark.asyncio
async def test_create_xor_gate():
    """Test XOR gate creation."""
    result = await create_xor_gate(num_qubits=2)

    assert result["status"] == "success"
    assert result["gate_type"] == "XOR"


# Advanced Algorithm Circuits
@pytest.mark.asyncio
async def test_create_hidden_linear_function():
    """Test hidden linear function circuit."""
    result = await create_hidden_linear_function(num_qubits=4)

    assert result["status"] == "success"
    assert result["circuit"]["num_qubits"] == 4


@pytest.mark.asyncio
async def test_create_iqp_circuit():
    """Test IQP circuit creation."""
    result = await create_iqp_circuit(num_qubits=3)

    assert result["status"] == "success"
    assert result["circuit"]["name"] == "IQP"


@pytest.mark.asyncio
async def test_create_phase_estimation_circuit():
    """Test phase estimation circuit creation."""
    # Create a simple unitary (identity for testing)
    from qiskit import QuantumCircuit, qasm2

    unitary_qc = QuantumCircuit(1)
    unitary_qc.z(0)
    unitary_qasm = qasm2.dumps(unitary_qc)

    result = await create_phase_estimation_circuit(
        unitary_qasm=unitary_qasm, num_evaluation_qubits=3
    )

    assert result["status"] == "success"
    assert result["num_evaluation_qubits"] == 3
    assert "circuit" in result


# Assisted by watsonx Code Assistant
