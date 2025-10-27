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

"""Tests for synchronous wrapper functions."""

from qiskit_mcp_server.sync import (
    create_quantum_circuit_sync,
    add_gates_to_circuit_sync,
    transpile_circuit_sync,
    get_circuit_depth_sync,
    get_statevector_sync,
    create_random_circuit_sync,
    get_qiskit_version_sync,
)


class TestSyncWrappers:
    """Tests for synchronous wrapper functions."""

    def test_create_quantum_circuit_sync(self):
        """Test synchronous circuit creation."""
        result = create_quantum_circuit_sync(2, 2, "test")

        assert result["status"] == "success"
        assert result["circuit"]["num_qubits"] == 2

    def test_add_gates_sync(self, sample_simple_circuit_qasm):
        """Test synchronous gate addition."""
        result = add_gates_to_circuit_sync(sample_simple_circuit_qasm, "h 0")

        assert result["status"] == "success"

    def test_transpile_sync(self, sample_circuit_qasm):
        """Test synchronous transpilation."""
        result = transpile_circuit_sync(sample_circuit_qasm, 1)

        assert result["status"] == "success"

    def test_get_depth_sync(self, sample_circuit_qasm):
        """Test synchronous depth retrieval."""
        result = get_circuit_depth_sync(sample_circuit_qasm)

        assert result["status"] == "success"
        assert "depth" in result

    def test_get_statevector_sync(self, sample_circuit_qasm):
        """Test synchronous statevector retrieval."""
        result = get_statevector_sync(sample_circuit_qasm)

        assert result["status"] == "success"
        assert "statevector" in result

    def test_create_random_circuit_sync(self):
        """Test synchronous random circuit creation."""
        result = create_random_circuit_sync(2, 3)

        assert result["status"] == "success"
        assert result["circuit"]["num_qubits"] == 2

    def test_get_version_sync(self):
        """Test synchronous version retrieval."""
        result = get_qiskit_version_sync()

        assert result["status"] == "success"
        assert "version" in result


# Assisted by watsonx Code Assistant
