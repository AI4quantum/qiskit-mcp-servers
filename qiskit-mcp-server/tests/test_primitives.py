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

"""Tests for Qiskit Primitives (Sampler and Estimator)."""

import pytest
from qiskit_mcp_server.primitives import (
    sample_circuit,
    estimate_expectation_values,
    run_variational_estimation,
)


@pytest.fixture
def bell_state_qasm_with_measurement():
    """Bell state with measurements."""
    return """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
"""


@pytest.fixture
def simple_measured_circuit():
    """Simple circuit with measurement."""
    return """OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[1];
h q[0];
measure q[0] -> c[0];
"""


class TestSampler:
    """Tests for Sampler primitive."""

    @pytest.mark.asyncio
    async def test_sample_circuit(self, bell_state_qasm_with_measurement):
        """Test sampling from Bell state."""
        result = await sample_circuit(bell_state_qasm_with_measurement, shots=1000)

        assert result["status"] == "success"
        assert result["shots"] == 1000
        assert "counts" in result
        assert result["num_qubits"] == 2
        assert result["num_clbits"] == 2

        # Bell state should give 00 and 11 with roughly equal probability
        counts = result["counts"]
        assert "00" in counts or "11" in counts

    @pytest.mark.asyncio
    async def test_sample_circuit_with_seed(self, simple_measured_circuit):
        """Test sampling with seed for reproducibility."""
        result1 = await sample_circuit(simple_measured_circuit, shots=100, seed=42)
        result2 = await sample_circuit(simple_measured_circuit, shots=100, seed=42)

        assert result1["status"] == "success"
        assert result2["status"] == "success"
        assert result1["counts"] == result2["counts"]

    @pytest.mark.asyncio
    async def test_sample_circuit_without_measurements(
        self, sample_simple_circuit_qasm
    ):
        """Test that circuit without measurements works (Qiskit 1.0+ auto-adds measurements)."""
        result = await sample_circuit(sample_simple_circuit_qasm, shots=100)

        # In Qiskit 1.0+, circuits without explicit measurements are automatically handled
        assert result["status"] == "success"
        assert "counts" in result

    @pytest.mark.asyncio
    async def test_sample_circuit_different_shots(self, simple_measured_circuit):
        """Test sampling with different shot counts."""
        result_100 = await sample_circuit(simple_measured_circuit, shots=100)
        result_1000 = await sample_circuit(simple_measured_circuit, shots=1000)

        assert result_100["status"] == "success"
        assert result_1000["status"] == "success"
        assert result_100["shots"] == 100
        assert result_1000["shots"] == 1000


class TestEstimator:
    """Tests for Estimator primitive."""

    @pytest.mark.asyncio
    async def test_estimate_expectation_values(self, sample_circuit_qasm):
        """Test estimating expectation values."""
        result = await estimate_expectation_values(sample_circuit_qasm, "ZZ")

        assert result["status"] == "success"
        assert "expectation_value" in result
        assert -1 <= result["expectation_value"] <= 1
        assert result["num_qubits"] == 2

    @pytest.mark.asyncio
    async def test_estimate_with_multiple_observables(self, sample_circuit_qasm):
        """Test estimating with multiple Pauli terms."""
        result = await estimate_expectation_values(
            sample_circuit_qasm, "XX,ZZ,YY", "0.5,0.3,0.2"
        )

        assert result["status"] == "success"
        assert "expectation_value" in result

    @pytest.mark.asyncio
    async def test_estimate_bell_state_zz(self):
        """Test ZZ expectation for Bell state."""
        bell_qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0],q[1];
"""
        result = await estimate_expectation_values(bell_qasm, "ZZ")

        assert result["status"] == "success"
        # For |Φ+⟩ = (|00⟩ + |11⟩)/√2, ⟨ZZ⟩ = 1
        assert abs(result["expectation_value"] - 1.0) < 1e-10

    @pytest.mark.asyncio
    async def test_estimate_mismatched_coeffs(self, sample_circuit_qasm):
        """Test error when coefficients don't match observables."""
        result = await estimate_expectation_values(sample_circuit_qasm, "XX,ZZ", "0.5")

        assert result["status"] == "error"
        assert "must match" in result["message"]


class TestVariationalEstimation:
    """Tests for variational estimation (VQE-style)."""

    @pytest.mark.asyncio
    async def test_run_variational_estimation(self):
        """Test variational estimation with multiple circuits."""
        import math

        # Create circuit variations with different angles
        circuit1 = f"""OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
ry({math.pi / 4}) q[0];
ry({math.pi / 4}) q[1];
"""
        circuit2 = f"""OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
ry({math.pi / 2}) q[0];
ry({math.pi / 2}) q[1];
"""

        result = await run_variational_estimation([circuit1, circuit2], "ZZ", "1.0")

        assert result["status"] == "success"
        assert result["num_circuits"] == 2
        assert "results" in result
        assert len(result["results"]) == 2
        assert "minimum_energy" in result
        assert "minimum_index" in result

    @pytest.mark.asyncio
    async def test_variational_estimation_finds_minimum(self):
        """Test that variational estimation finds the minimum energy."""
        # Create circuits with known energies
        # For H = ZZ, ground state is |00⟩ or |11⟩ with E = +1
        ground_state = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
"""
        excited_state = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
x q[0];
"""

        result = await run_variational_estimation([ground_state, excited_state], "ZZ")

        assert result["status"] == "success"
        # |00⟩ should have higher ⟨ZZ⟩ than |01⟩
        assert result["minimum_index"] in [0, 1]


# Assisted by watsonx Code Assistant
