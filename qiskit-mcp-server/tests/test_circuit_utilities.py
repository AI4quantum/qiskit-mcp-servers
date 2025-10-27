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

"""Tests for circuit utilities module."""

import pytest
import os
import tempfile
from qiskit_mcp_server.circuit_utilities import (
    circuit_inverse,
    circuit_copy,
    circuit_reverse_bits,
    circuit_to_gate,
    circuit_to_instruction,
    load_qasm2_file,
    save_qasm2_file,
    load_qasm3_file,
    save_qasm3_file,
    convert_circuit_to_dag,
    convert_dag_to_circuit_wrapper,
    decompose_circuit,
)


@pytest.mark.asyncio
async def test_circuit_inverse():
    """Test circuit inverse operation."""
    # Create a simple circuit
    from qiskit import QuantumCircuit, qasm2

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qasm = qasm2.dumps(qc)

    result = await circuit_inverse(qasm)

    assert result["status"] == "success"
    assert (
        "inverse" in result["circuit"]["name"] or result["circuit"]["name"] == "inverse"
    )
    assert result["circuit"]["num_qubits"] == 2


@pytest.mark.asyncio
async def test_circuit_copy():
    """Test circuit copy operation."""
    from qiskit import QuantumCircuit, qasm2

    qc = QuantumCircuit(2, name="original")
    qc.h(0)
    qc.cx(0, 1)
    qasm = qasm2.dumps(qc)

    result = await circuit_copy(qasm, name="copied")

    assert result["status"] == "success"
    assert result["circuit"]["name"] == "copied"
    assert result["circuit"]["num_qubits"] == 2


@pytest.mark.asyncio
async def test_circuit_reverse_bits():
    """Test circuit reverse bits operation."""
    from qiskit import QuantumCircuit, qasm2

    qc = QuantumCircuit(3)
    qc.h(0)
    qc.x(2)
    qasm = qasm2.dumps(qc)

    result = await circuit_reverse_bits(qasm)

    assert result["status"] == "success"
    assert result["circuit"]["num_qubits"] == 3


@pytest.mark.asyncio
async def test_circuit_to_gate():
    """Test converting circuit to gate."""
    from qiskit import QuantumCircuit, qasm2

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qasm = qasm2.dumps(qc)

    result = await circuit_to_gate(qasm, label="my_gate")

    assert result["status"] == "success"
    assert result["gate"]["label"] == "my_gate"
    assert result["gate"]["num_qubits"] == 2
    assert "usage_example" in result


@pytest.mark.asyncio
async def test_circuit_to_instruction():
    """Test converting circuit to instruction."""
    from qiskit import QuantumCircuit, qasm2

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qasm = qasm2.dumps(qc)

    result = await circuit_to_instruction(qasm, label="my_instruction")

    assert result["status"] == "success"
    assert result["instruction"]["label"] == "my_instruction"
    assert result["instruction"]["num_qubits"] == 2
    assert "usage_example" in result


@pytest.mark.asyncio
async def test_save_and_load_qasm2_file():
    """Test saving and loading QASM 2.0 files."""
    from qiskit import QuantumCircuit, qasm2

    # Create a circuit
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    qasm = qasm2.dumps(qc)

    # Use temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".qasm", delete=False) as f:
        temp_path = f.name

    try:
        # Save
        save_result = await save_qasm2_file(qasm, temp_path)
        assert save_result["status"] == "success"
        assert os.path.exists(temp_path)

        # Load
        load_result = await load_qasm2_file(temp_path)
        assert load_result["status"] == "success"
        assert load_result["circuit"]["num_qubits"] == 2
        assert load_result["circuit"]["num_clbits"] == 2

    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)


@pytest.mark.asyncio
async def test_save_and_load_qasm3_file():
    """Test saving and loading QASM 3.0 files."""
    pytest.importorskip(
        "qiskit_qasm3_import",
        reason="qiskit_qasm3_import required for QASM 3.0 loading",
    )
    from qiskit import QuantumCircuit, qasm2

    # Create a circuit
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qasm = qasm2.dumps(qc)

    # Use temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".qasm", delete=False) as f:
        temp_path = f.name

    try:
        # Save
        save_result = await save_qasm3_file(qasm, temp_path)
        assert save_result["status"] == "success"
        assert os.path.exists(temp_path)

        # Load
        load_result = await load_qasm3_file(temp_path)
        assert load_result["status"] == "success"
        assert load_result["circuit"]["num_qubits"] == 2

    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)


@pytest.mark.asyncio
async def test_load_nonexistent_file():
    """Test loading a file that doesn't exist."""
    result = await load_qasm2_file("/nonexistent/path/file.qasm")

    assert result["status"] == "error"
    assert "not found" in result["message"].lower()


@pytest.mark.asyncio
async def test_convert_circuit_to_dag():
    """Test converting circuit to DAG."""
    from qiskit import QuantumCircuit, qasm2

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.h(1)
    qasm = qasm2.dumps(qc)

    result = await convert_circuit_to_dag(qasm)

    assert result["status"] == "success"
    assert result["dag"]["num_qubits"] == 2
    assert result["dag"]["num_operations"] > 0
    assert result["dag"]["depth"] > 0
    assert "operation_counts" in result["dag"]


@pytest.mark.asyncio
async def test_convert_dag_to_circuit():
    """Test converting DAG back to circuit."""
    from qiskit import QuantumCircuit, qasm2

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qasm = qasm2.dumps(qc)

    result = await convert_dag_to_circuit_wrapper(qasm)

    assert result["status"] == "success"
    assert result["original"]["num_qubits"] == 2
    assert result["reconstructed"]["num_qubits"] == 2
    assert result["reconstructed"]["depth"] == result["original"]["depth"]


@pytest.mark.asyncio
async def test_decompose_circuit():
    """Test decomposing a circuit."""
    from qiskit import QuantumCircuit, qasm2

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qasm = qasm2.dumps(qc)

    result = await decompose_circuit(qasm, reps=1)

    assert result["status"] == "success"
    assert result["decomposed"]["num_qubits"] == 2
    # Decomposition may increase size
    assert "increase" in result


@pytest.mark.asyncio
async def test_decompose_circuit_with_specific_gates():
    """Test decomposing specific gates in a circuit."""
    from qiskit import QuantumCircuit, qasm2

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.h(1)
    qasm = qasm2.dumps(qc)

    result = await decompose_circuit(qasm, gates_to_decompose="h", reps=1)

    assert result["status"] == "success"
    assert result["decomposed"]["num_qubits"] == 2


# Assisted by watsonx Code Assistant
