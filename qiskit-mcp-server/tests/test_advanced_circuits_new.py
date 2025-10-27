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

"""Tests for new advanced circuits functions: Pauli evolution, oracles, TwoLocal, and ParameterVector."""

import pytest
from qiskit_mcp_server.advanced_circuits import (
    create_pauli_evolution_circuit,
    create_phase_oracle_circuit,
    create_general_two_local_circuit,
    create_parametric_circuit_with_vector,
)


@pytest.mark.asyncio
async def test_create_pauli_evolution_circuit_basic():
    """Test creating a basic Pauli evolution circuit."""
    result = await create_pauli_evolution_circuit(pauli_string="ZZ", time=0.5)

    assert result["status"] == "success"
    assert result["pauli_string"] == "ZZ"
    assert result["time"] == 0.5
    assert "circuit" in result
    assert "qasm" in result["circuit"]
    assert "num_qubits" in result["circuit"]
    assert result["circuit"]["num_qubits"] == 2


@pytest.mark.asyncio
async def test_create_pauli_evolution_circuit_with_num_qubits():
    """Test creating Pauli evolution circuit with explicit num_qubits."""
    result = await create_pauli_evolution_circuit(
        pauli_string="XY", time=1.0, num_qubits=3
    )

    assert result["status"] == "success"
    assert result["circuit"]["num_qubits"] == 3
    assert result["time"] == 1.0


@pytest.mark.asyncio
async def test_create_pauli_evolution_circuit_single_qubit():
    """Test creating single-qubit Pauli evolution circuit."""
    result = await create_pauli_evolution_circuit(pauli_string="X", time=0.25)

    assert result["status"] == "success"
    assert result["pauli_string"] == "X"
    assert result["circuit"]["num_qubits"] == 1


@pytest.mark.asyncio
async def test_create_pauli_evolution_circuit_complex():
    """Test creating complex multi-qubit Pauli evolution circuit."""
    result = await create_pauli_evolution_circuit(pauli_string="XYZI", time=2.0)

    assert result["status"] == "success"
    assert result["pauli_string"] == "XYZI"
    assert result["circuit"]["num_qubits"] == 4
    assert "usage" in result
    assert "Hamiltonian simulation" in result["usage"]


@pytest.mark.asyncio
async def test_create_pauli_evolution_circuit_zero_time():
    """Test Pauli evolution with time=0 (identity operation)."""
    result = await create_pauli_evolution_circuit(pauli_string="ZZ", time=0.0)

    assert result["status"] == "success"
    assert result["time"] == 0.0


@pytest.mark.asyncio
async def test_create_phase_oracle_circuit_simple():
    """Test creating a simple phase oracle."""
    result = await create_phase_oracle_circuit(expression="a & b", num_qubits=2)

    assert result["status"] == "success"
    assert result["expression"] == "a & b"
    assert result["num_qubits"] == 2
    assert "circuit" in result
    assert "qasm" in result["circuit"]


@pytest.mark.asyncio
async def test_create_phase_oracle_circuit_or():
    """Test creating phase oracle with OR operation."""
    result = await create_phase_oracle_circuit(expression="a | b", num_qubits=2)

    assert result["status"] == "success"
    assert result["expression"] == "a | b"
    assert "oracle_type" in result
    assert result["oracle_type"] == "phase"


@pytest.mark.asyncio
async def test_create_phase_oracle_circuit_not():
    """Test creating phase oracle with NOT operation."""
    result = await create_phase_oracle_circuit(expression="~a", num_qubits=1)

    assert result["status"] == "success"
    assert result["num_qubits"] == 1


@pytest.mark.asyncio
async def test_create_phase_oracle_circuit_complex():
    """Test creating complex phase oracle with multiple operations."""
    result = await create_phase_oracle_circuit(
        expression="(a & b) | (~c)", num_qubits=3
    )

    assert result["status"] == "success"
    assert result["expression"] == "(a & b) | (~c)"
    assert result["num_qubits"] == 3
    assert "usage" in result
    assert "Grover" in result["usage"]


@pytest.mark.asyncio
async def test_create_phase_oracle_circuit_xor():
    """Test creating phase oracle with XOR operation."""
    result = await create_phase_oracle_circuit(expression="a ^ b", num_qubits=2)

    assert result["status"] == "success"
    assert "^" in result["expression"] or "xor" in result["expression"].lower()


@pytest.mark.asyncio
async def test_create_general_two_local_circuit_basic():
    """Test creating basic TwoLocal circuit with defaults."""
    result = await create_general_two_local_circuit(num_qubits=3)

    assert result["status"] == "success"
    assert result["num_qubits"] == 3
    assert result["rotation_blocks"] == "ry"
    assert result["entanglement_blocks"] == "cx"
    assert result["entanglement"] == "full"
    assert result["reps"] == 3
    assert "circuit" in result
    assert "qasm" in result["circuit"]


@pytest.mark.asyncio
async def test_create_general_two_local_circuit_custom_rotation():
    """Test TwoLocal with custom rotation blocks."""
    result = await create_general_two_local_circuit(
        num_qubits=2, rotation_blocks="rz", entanglement_blocks="cx", reps=2
    )

    assert result["status"] == "success"
    assert result["rotation_blocks"] == "rz"
    assert result["reps"] == 2


@pytest.mark.asyncio
async def test_create_general_two_local_circuit_multiple_rotations():
    """Test TwoLocal with multiple rotation blocks."""
    result = await create_general_two_local_circuit(
        num_qubits=3, rotation_blocks="ry,rz", entanglement_blocks="cx", reps=2
    )

    assert result["status"] == "success"
    # Should handle comma-separated blocks
    assert "ry" in str(result["rotation_blocks"]) or isinstance(
        result["rotation_blocks"], list
    )


@pytest.mark.asyncio
async def test_create_general_two_local_circuit_linear_entanglement():
    """Test TwoLocal with linear entanglement."""
    result = await create_general_two_local_circuit(
        num_qubits=4, entanglement="linear", reps=2
    )

    assert result["status"] == "success"
    assert result["entanglement"] == "linear"


@pytest.mark.asyncio
async def test_create_general_two_local_circuit_circular_entanglement():
    """Test TwoLocal with circular entanglement."""
    result = await create_general_two_local_circuit(
        num_qubits=4, entanglement="circular", reps=2
    )

    assert result["status"] == "success"
    assert result["entanglement"] == "circular"


@pytest.mark.asyncio
async def test_create_general_two_local_circuit_with_barriers():
    """Test TwoLocal with barriers inserted."""
    result = await create_general_two_local_circuit(
        num_qubits=3, reps=2, insert_barriers=True
    )

    assert result["status"] == "success"
    assert result["insert_barriers"]
    # Barriers should be in the circuit
    assert "barrier" in result["circuit"]["qasm"].lower()


@pytest.mark.asyncio
async def test_create_general_two_local_circuit_cz_entanglement():
    """Test TwoLocal with CZ entanglement gates."""
    result = await create_general_two_local_circuit(
        num_qubits=3, rotation_blocks="ry", entanglement_blocks="cz", reps=2
    )

    assert result["status"] == "success"
    assert result["entanglement_blocks"] == "cz"


@pytest.mark.asyncio
async def test_create_parametric_circuit_with_vector_basic():
    """Test creating parametric circuit with ParameterVector - basic."""
    result = await create_parametric_circuit_with_vector(
        num_qubits=2, num_parameters=4, structure="ry_cx"
    )

    assert result["status"] == "success"
    assert result["num_qubits"] == 2
    assert result["num_parameters"] == 4
    assert result["structure"] == "ry_cx"
    assert "circuit" in result
    assert "parameter_vector" in result
    assert result["parameter_vector"]["name"] == "θ"
    assert result["parameter_vector"]["length"] == 4


@pytest.mark.asyncio
async def test_create_parametric_circuit_with_vector_rz_cz():
    """Test parametric circuit with rz_cz structure."""
    result = await create_parametric_circuit_with_vector(
        num_qubits=3, num_parameters=6, structure="rz_cz"
    )

    assert result["status"] == "success"
    assert result["structure"] == "rz_cz"
    assert result["num_parameters"] == 6


@pytest.mark.asyncio
async def test_create_parametric_circuit_with_vector_full_rotation():
    """Test parametric circuit with full_rotation structure."""
    result = await create_parametric_circuit_with_vector(
        num_qubits=2, num_parameters=8, structure="full_rotation"
    )

    assert result["status"] == "success"
    assert result["structure"] == "full_rotation"
    assert "parameters" in result["circuit"]


@pytest.mark.asyncio
async def test_create_parametric_circuit_with_vector_hardware_efficient():
    """Test parametric circuit with hardware_efficient structure."""
    result = await create_parametric_circuit_with_vector(
        num_qubits=3, num_parameters=9, structure="hardware_efficient"
    )

    assert result["status"] == "success"
    assert result["structure"] == "hardware_efficient"


@pytest.mark.asyncio
async def test_create_parametric_circuit_with_vector_parameter_info():
    """Test that parametric circuit includes parameter binding info."""
    result = await create_parametric_circuit_with_vector(
        num_qubits=2, num_parameters=4, structure="ry_cx"
    )

    assert result["status"] == "success"
    assert "parameter_vector" in result
    assert "example_binding" in result
    assert "usage" in result
    assert "variational" in result["usage"].lower()


@pytest.mark.asyncio
async def test_create_parametric_circuit_with_vector_single_qubit():
    """Test parametric circuit with single qubit."""
    result = await create_parametric_circuit_with_vector(
        num_qubits=1, num_parameters=2, structure="ry_cx"
    )

    assert result["status"] == "success"
    assert result["num_qubits"] == 1


@pytest.mark.asyncio
async def test_create_parametric_circuit_with_vector_many_parameters():
    """Test parametric circuit with many parameters."""
    result = await create_parametric_circuit_with_vector(
        num_qubits=4, num_parameters=20, structure="hardware_efficient"
    )

    assert result["status"] == "success"
    assert result["num_parameters"] == 20
    assert result["parameter_vector"]["length"] == 20


# Integration tests combining multiple functions
@pytest.mark.asyncio
async def test_pauli_evolution_and_oracle_integration():
    """Test that Pauli evolution and oracle circuits can be created together."""
    pauli_result = await create_pauli_evolution_circuit(pauli_string="ZZ", time=1.0)
    oracle_result = await create_phase_oracle_circuit(expression="a & b", num_qubits=2)

    assert pauli_result["status"] == "success"
    assert oracle_result["status"] == "success"
    # Both should have valid QASM
    assert "qasm" in pauli_result["circuit"]
    assert "qasm" in oracle_result["circuit"]


# Assisted by watsonx Code Assistant
