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

"""Tests for Qiskit SDK functions."""

import pytest
from qiskit_mcp_server.qiskit_sdk import (
    create_quantum_circuit,
    add_gates_to_circuit,
    transpile_circuit,
    get_circuit_depth,
    get_circuit_qasm,
    get_statevector,
    visualize_circuit,
    create_random_circuit,
    get_qiskit_version,
)


class TestCreateQuantumCircuit:
    """Tests for create_quantum_circuit function."""

    @pytest.mark.asyncio
    async def test_create_circuit_success(self):
        """Test successful circuit creation."""
        result = await create_quantum_circuit(2, 2, "test_circuit")

        assert result["status"] == "success"
        assert result["circuit"]["num_qubits"] == 2
        assert result["circuit"]["num_clbits"] == 2
        assert result["circuit"]["name"] == "test_circuit"
        assert "qasm" in result["circuit"]

    @pytest.mark.asyncio
    async def test_create_circuit_no_classical(self):
        """Test circuit creation without classical bits."""
        result = await create_quantum_circuit(3, 0)

        assert result["status"] == "success"
        assert result["circuit"]["num_qubits"] == 3
        assert result["circuit"]["num_clbits"] == 0

    @pytest.mark.asyncio
    async def test_create_circuit_invalid_qubits(self):
        """Test circuit creation with invalid qubit count."""
        result = await create_quantum_circuit(0, 0)

        assert result["status"] == "error"
        assert "positive" in result["message"]


class TestAddGatesToCircuit:
    """Tests for add_gates_to_circuit function."""

    @pytest.mark.asyncio
    async def test_add_single_gate(self, sample_simple_circuit_qasm):
        """Test adding a single gate."""
        result = await add_gates_to_circuit(sample_simple_circuit_qasm, "h 0")

        assert result["status"] == "success"
        assert result["circuit"]["depth"] > 0
        assert "h q[0]" in result["circuit"]["qasm"]

    @pytest.mark.asyncio
    async def test_add_multiple_gates(self, sample_simple_circuit_qasm):
        """Test adding multiple gates."""
        result = await add_gates_to_circuit(sample_simple_circuit_qasm, "h 0; cx 0 1")

        assert result["status"] == "success"
        assert result["circuit"]["depth"] >= 2
        assert "h q[0]" in result["circuit"]["qasm"]
        assert "cx q[0],q[1]" in result["circuit"]["qasm"]

    @pytest.mark.asyncio
    async def test_add_measurement(self, sample_simple_circuit_qasm):
        """Test adding measurement."""
        result = await add_gates_to_circuit(
            sample_simple_circuit_qasm, "h 0; measure 0 0"
        )

        assert result["status"] == "success"
        assert "measure" in result["circuit"]["qasm"]

    @pytest.mark.asyncio
    async def test_add_rotation_gates(self, sample_simple_circuit_qasm):
        """Test adding rotation gates with parameters."""
        import math

        # Test RX gate
        result = await add_gates_to_circuit(
            sample_simple_circuit_qasm, f"rx {math.pi / 2} 0"
        )
        assert result["status"] == "success"
        assert "rx" in result["circuit"]["qasm"].lower()

        # Test RY gate
        result = await add_gates_to_circuit(
            sample_simple_circuit_qasm, f"ry {math.pi / 4} 1"
        )
        assert result["status"] == "success"
        assert "ry" in result["circuit"]["qasm"].lower()

        # Test RZ gate
        result = await add_gates_to_circuit(
            sample_simple_circuit_qasm, f"rz {math.pi} 0"
        )
        assert result["status"] == "success"
        assert "rz" in result["circuit"]["qasm"].lower()

    @pytest.mark.asyncio
    async def test_add_phase_gate(self, sample_simple_circuit_qasm):
        """Test adding phase gate."""
        import math

        result = await add_gates_to_circuit(
            sample_simple_circuit_qasm, f"p {math.pi / 2} 0"
        )
        assert result["status"] == "success"
        assert result["circuit"]["depth"] > 0

    @pytest.mark.asyncio
    async def test_add_u_gate(self, sample_simple_circuit_qasm):
        """Test adding U gate with three parameters."""
        import math

        result = await add_gates_to_circuit(
            sample_simple_circuit_qasm, f"u {math.pi / 2} {math.pi / 4} {math.pi / 3} 0"
        )
        assert result["status"] == "success"
        assert result["circuit"]["depth"] > 0

    @pytest.mark.asyncio
    async def test_add_advanced_gates(self, sample_simple_circuit_qasm):
        """Test adding advanced single-qubit gates."""
        result = await add_gates_to_circuit(
            sample_simple_circuit_qasm, "sdg 0; tdg 1; sx 0; sxdg 1"
        )
        assert result["status"] == "success"
        assert result["circuit"]["depth"] > 0

    @pytest.mark.asyncio
    async def test_add_barrier(self, sample_simple_circuit_qasm):
        """Test adding barrier."""
        result = await add_gates_to_circuit(
            sample_simple_circuit_qasm, "h 0; barrier 0 1; x 1"
        )
        assert result["status"] == "success"
        assert "barrier" in result["circuit"]["qasm"]

    @pytest.mark.asyncio
    async def test_add_reset(self, sample_simple_circuit_qasm):
        """Test adding reset."""
        result = await add_gates_to_circuit(
            sample_simple_circuit_qasm, "x 0; reset 0; h 0"
        )
        assert result["status"] == "success"
        assert "reset" in result["circuit"]["qasm"]

    @pytest.mark.asyncio
    async def test_mixed_rotation_and_standard_gates(self, sample_simple_circuit_qasm):
        """Test mixing rotation gates with standard gates."""
        import math

        result = await add_gates_to_circuit(
            sample_simple_circuit_qasm,
            f"h 0; rx {math.pi / 2} 0; cx 0 1; ry {math.pi / 4} 1",
        )
        assert result["status"] == "success"
        assert result["circuit"]["depth"] > 0
        assert "h q[0]" in result["circuit"]["qasm"]
        assert "cx q[0],q[1]" in result["circuit"]["qasm"]


class TestTranspileCircuit:
    """Tests for transpile_circuit function."""

    @pytest.mark.asyncio
    async def test_transpile_basic(self, sample_circuit_qasm):
        """Test basic transpilation."""
        result = await transpile_circuit(sample_circuit_qasm, 1)

        assert result["status"] == "success"
        assert "transpiled_depth" in result
        assert "original_depth" in result
        assert result["circuit"]["depth"] == result["transpiled_depth"]

    @pytest.mark.asyncio
    async def test_transpile_with_basis_gates(self, sample_circuit_qasm):
        """Test transpilation with specific basis gates."""
        result = await transpile_circuit(sample_circuit_qasm, 1, "cx,id,rz,sx,x")

        assert result["status"] == "success"
        assert "transpiled_depth" in result

    @pytest.mark.asyncio
    async def test_transpile_optimization_levels(self, sample_circuit_qasm):
        """Test different optimization levels."""
        result0 = await transpile_circuit(sample_circuit_qasm, 0)
        result3 = await transpile_circuit(sample_circuit_qasm, 3)

        assert result0["status"] == "success"
        assert result3["status"] == "success"
        # Higher optimization may reduce depth
        assert result3["transpiled_depth"] <= result0["transpiled_depth"]


class TestGetCircuitDepth:
    """Tests for get_circuit_depth function."""

    @pytest.mark.asyncio
    async def test_get_depth(self, sample_circuit_qasm):
        """Test getting circuit depth."""
        result = await get_circuit_depth(sample_circuit_qasm)

        assert result["status"] == "success"
        assert result["depth"] > 0
        assert result["num_qubits"] == 2
        assert "size" in result


class TestGetCircuitQasm:
    """Tests for get_circuit_qasm function."""

    @pytest.mark.asyncio
    async def test_get_qasm(self, sample_circuit_qasm):
        """Test getting QASM representation."""
        result = await get_circuit_qasm(sample_circuit_qasm)

        assert result["status"] == "success"
        assert "qasm" in result
        assert "OPENQASM" in result["qasm"]


class TestGetStatevector:
    """Tests for get_statevector function."""

    @pytest.mark.asyncio
    async def test_get_statevector(self, sample_circuit_qasm):
        """Test getting statevector."""
        result = await get_statevector(sample_circuit_qasm)

        assert result["status"] == "success"
        assert "statevector" in result
        assert "probabilities" in result
        assert result["num_qubits"] == 2
        # Check probabilities sum to 1
        assert abs(sum(result["probabilities"]) - 1.0) < 1e-10


class TestVisualizeCircuit:
    """Tests for visualize_circuit function."""

    @pytest.mark.asyncio
    async def test_visualize_text(self, sample_circuit_qasm):
        """Test text visualization."""
        result = await visualize_circuit(sample_circuit_qasm, "text")

        assert result["status"] == "success"
        assert result["format"] == "text"
        assert "visualization" in result

    @pytest.mark.asyncio
    async def test_visualize_mpl_fallback(self, sample_circuit_qasm):
        """Test matplotlib format falls back to text."""
        result = await visualize_circuit(sample_circuit_qasm, "mpl")

        assert result["status"] == "success"
        assert "visualization" in result


class TestCreateRandomCircuit:
    """Tests for create_random_circuit function."""

    @pytest.mark.asyncio
    async def test_create_random_circuit(self):
        """Test creating random circuit."""
        result = await create_random_circuit(3, 5)

        assert result["status"] == "success"
        assert result["circuit"]["num_qubits"] == 3
        assert result["circuit"]["depth"] > 0
        assert "qasm" in result["circuit"]

    @pytest.mark.asyncio
    async def test_create_random_circuit_with_measurement(self):
        """Test creating random circuit with measurements."""
        result = await create_random_circuit(2, 3, measure=True)

        assert result["status"] == "success"
        assert result["circuit"]["num_clbits"] > 0

    @pytest.mark.asyncio
    async def test_create_random_circuit_with_seed(self):
        """Test creating random circuit with seed for reproducibility."""
        result1 = await create_random_circuit(2, 3, seed=42)
        result2 = await create_random_circuit(2, 3, seed=42)

        assert result1["status"] == "success"
        assert result2["status"] == "success"
        # Same seed should produce same circuit
        assert result1["circuit"]["qasm"] == result2["circuit"]["qasm"]

    @pytest.mark.asyncio
    async def test_create_random_circuit_invalid_params(self):
        """Test creating random circuit with invalid parameters."""
        result = await create_random_circuit(0, 5)

        assert result["status"] == "error"


class TestGetQiskitVersion:
    """Tests for get_qiskit_version function."""

    @pytest.mark.asyncio
    async def test_get_version(self):
        """Test getting Qiskit version."""
        result = await get_qiskit_version()

        assert result["status"] == "success"
        assert "version" in result
        assert result["sdk"] == "Qiskit"


# Assisted by watsonx Code Assistant
