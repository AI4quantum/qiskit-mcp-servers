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

"""Tests for quantum information functions."""

import pytest
from qiskit_mcp_server.quantum_info import (
    create_pauli_operator,
    create_operator_from_circuit,
    create_density_matrix,
    calculate_state_fidelity,
    calculate_gate_fidelity,
    calculate_entropy,
    calculate_entanglement,
    partial_trace_state,
    expectation_value,
    random_quantum_state,
    random_density_matrix_state,
)


class TestPauliOperator:
    """Tests for Pauli operator creation."""

    @pytest.mark.asyncio
    async def test_create_pauli_operator(self):
        """Test creating SparsePauliOp."""
        result = await create_pauli_operator("XX,YZ,ZZ")

        assert result["status"] == "success"
        assert "operator" in result
        assert result["operator"]["num_qubits"] == 2
        assert result["operator"]["size"] == 3

    @pytest.mark.asyncio
    async def test_create_pauli_operator_with_coeffs(self):
        """Test creating SparsePauliOp with coefficients."""
        result = await create_pauli_operator("XX,YZ", "1.0,0.5")

        assert result["status"] == "success"
        assert result["operator"]["coeffs"] == [1.0, 0.5]

    @pytest.mark.asyncio
    async def test_create_pauli_operator_mismatched_coeffs(self):
        """Test error when coefficients don't match Pauli strings."""
        result = await create_pauli_operator("XX,YZ", "1.0")

        assert result["status"] == "error"
        assert "must match" in result["message"]


class TestOperatorCreation:
    """Tests for operator and density matrix creation."""

    @pytest.mark.asyncio
    async def test_create_operator_from_circuit(self, sample_circuit_qasm):
        """Test creating Operator from circuit."""
        result = await create_operator_from_circuit(sample_circuit_qasm)

        assert result["status"] == "success"
        assert "operator" in result
        assert result["operator"]["num_qubits"] == 2
        assert result["operator"]["is_unitary"] is True

    @pytest.mark.asyncio
    async def test_create_density_matrix(self, sample_circuit_qasm):
        """Test creating DensityMatrix from circuit."""
        result = await create_density_matrix(sample_circuit_qasm)

        assert result["status"] == "success"
        assert "density_matrix" in result
        assert result["density_matrix"]["num_qubits"] == 2
        assert result["density_matrix"]["is_valid"] is True
        assert 0 <= result["density_matrix"]["purity"] <= 1


class TestFidelity:
    """Tests for fidelity calculations."""

    @pytest.mark.asyncio
    async def test_state_fidelity_identical(self, sample_circuit_qasm):
        """Test fidelity of identical states."""
        result = await calculate_state_fidelity(
            sample_circuit_qasm, sample_circuit_qasm
        )

        assert result["status"] == "success"
        assert abs(result["fidelity"] - 1.0) < 1e-10

    @pytest.mark.asyncio
    async def test_state_fidelity_different(
        self, sample_circuit_qasm, sample_simple_circuit_qasm
    ):
        """Test fidelity of different states."""
        result = await calculate_state_fidelity(
            sample_circuit_qasm, sample_simple_circuit_qasm
        )

        assert result["status"] == "success"
        assert 0 <= result["fidelity"] <= 1

    @pytest.mark.asyncio
    async def test_gate_fidelity(self, sample_circuit_qasm):
        """Test gate fidelity calculation."""
        result = await calculate_gate_fidelity(sample_circuit_qasm, sample_circuit_qasm)

        assert result["status"] == "success"
        assert abs(result["average_gate_fidelity"] - 1.0) < 1e-10
        assert abs(result["process_fidelity"] - 1.0) < 1e-10


class TestEntropy:
    """Tests for entropy calculations."""

    @pytest.mark.asyncio
    async def test_calculate_entropy(self, sample_circuit_qasm):
        """Test von Neumann entropy calculation."""
        result = await calculate_entropy(sample_circuit_qasm)

        assert result["status"] == "success"
        assert "entropy" in result
        assert result["entropy"] >= 0

    @pytest.mark.asyncio
    async def test_calculate_entropy_with_partial_trace(
        self, sample_3qubit_circuit_qasm
    ):
        """Test entropy with partial trace."""
        result = await calculate_entropy(sample_3qubit_circuit_qasm, "0")

        assert result["status"] == "success"
        assert result["num_qubits"] == 2  # Traced out 1 qubit


class TestEntanglement:
    """Tests for entanglement measures."""

    @pytest.mark.asyncio
    async def test_calculate_entanglement_bell_state(self):
        """Test entanglement of formation for Bell state."""
        # Create Bell state
        bell_qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0],q[1];
"""
        result = await calculate_entanglement(bell_qasm)

        assert result["status"] == "success"
        assert result["entanglement_of_formation"] > 0

    @pytest.mark.asyncio
    async def test_calculate_entanglement_wrong_qubits(
        self, sample_3qubit_circuit_qasm
    ):
        """Test error for non-2-qubit circuit."""
        result = await calculate_entanglement(sample_3qubit_circuit_qasm)

        assert result["status"] == "error"
        assert "exactly 2 qubits" in result["message"]


class TestPartialTrace:
    """Tests for partial trace."""

    @pytest.mark.asyncio
    async def test_partial_trace(self, sample_3qubit_circuit_qasm):
        """Test partial trace operation."""
        result = await partial_trace_state(sample_3qubit_circuit_qasm, "0,2")

        assert result["status"] == "success"
        assert "reduced_state" in result
        assert result["reduced_state"]["num_qubits"] == 1
        assert result["traced_qubits"] == [0, 2]


class TestExpectationValue:
    """Tests for expectation value calculations."""

    @pytest.mark.asyncio
    async def test_expectation_value(self, sample_circuit_qasm):
        """Test expectation value calculation."""
        result = await expectation_value(sample_circuit_qasm, "ZZ")

        assert result["status"] == "success"
        assert "expectation_value" in result
        assert -1 <= result["expectation_value"] <= 1

    @pytest.mark.asyncio
    async def test_expectation_value_with_coeffs(self, sample_circuit_qasm):
        """Test expectation value with coefficients."""
        result = await expectation_value(sample_circuit_qasm, "XX,ZZ", "0.5,0.5")

        assert result["status"] == "success"
        assert "expectation_value" in result


class TestRandomStates:
    """Tests for random state generation."""

    @pytest.mark.asyncio
    async def test_random_quantum_state(self):
        """Test random statevector generation."""
        result = await random_quantum_state(2)

        assert result["status"] == "success"
        assert result["num_qubits"] == 2
        assert result["is_valid"] is True
        assert len(result["probabilities"]) == 4

    @pytest.mark.asyncio
    async def test_random_quantum_state_with_seed(self):
        """Test random statevector with seed."""
        result1 = await random_quantum_state(2, seed=42)
        result2 = await random_quantum_state(2, seed=42)

        assert result1["statevector"] == result2["statevector"]

    @pytest.mark.asyncio
    async def test_random_density_matrix(self):
        """Test random density matrix generation."""
        result = await random_density_matrix_state(2)

        assert result["status"] == "success"
        assert result["density_matrix"]["num_qubits"] == 2
        assert result["density_matrix"]["is_valid"] is True

    @pytest.mark.asyncio
    async def test_random_density_matrix_with_rank(self):
        """Test random density matrix with specified rank."""
        result = await random_density_matrix_state(2, rank=2)

        assert result["status"] == "success"
        assert result["density_matrix"]["rank"] == 2


# Assisted by watsonx Code Assistant
