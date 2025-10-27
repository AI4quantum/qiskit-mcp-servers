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

"""Test configuration and fixtures for Qiskit MCP Server tests."""

import pytest


@pytest.fixture
def sample_circuit_qasm():
    """Sample QASM circuit for testing."""
    return """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
"""


@pytest.fixture
def sample_simple_circuit_qasm():
    """Simple QASM circuit for testing."""
    return """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
"""


@pytest.fixture
def sample_3qubit_circuit_qasm():
    """3-qubit QASM circuit for testing."""
    return """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
h q[0];
h q[1];
h q[2];
"""


# Assisted by watsonx Code Assistant
