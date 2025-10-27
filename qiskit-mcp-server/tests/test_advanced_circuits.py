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

"""Tests for advanced circuit operations."""

import pytest
from qiskit_mcp_server.qiskit_sdk import create_quantum_circuit, add_gates_to_circuit
from qiskit_mcp_server.advanced_circuits import (
    compose_circuits,
    tensor_circuits,
    create_parametric_circuit,
    decompose_circuit,
    add_controlled_gate,
    add_power_gate,
    create_qft_circuit,
    create_efficient_su2,
    create_real_amplitudes,
    simulate_with_aer,
    get_unitary_matrix,
    analyze_circuit,
    get_circuit_instructions,
    convert_to_qasm3,
    draw_circuit_text,
)


@pytest.mark.asyncio
async def test_compose_circuits():
    """Test circuit composition."""
    # Create two circuits
    result1 = await create_quantum_circuit(2, 0, "circuit1")
    assert result1["status"] == "success"
    qasm1 = result1["circuit"]["qasm"]

    # Add gates to first circuit
    result1 = await add_gates_to_circuit(qasm1, "h 0; h 1")
    qasm1 = result1["circuit"]["qasm"]

    # Create second circuit
    result2 = await create_quantum_circuit(2, 0, "circuit2")
    qasm2 = result2["circuit"]["qasm"]
    result2 = await add_gates_to_circuit(qasm2, "cx 0 1")
    qasm2 = result2["circuit"]["qasm"]

    # Compose circuits
    result = await compose_circuits(qasm1, qasm2)
    assert result["status"] == "success"
    assert result["circuit"]["num_qubits"] == 2
    assert result["circuit"]["depth"] > 0


@pytest.mark.asyncio
async def test_tensor_circuits():
    """Test circuit tensor product."""
    # Create two circuits with different qubit counts
    result1 = await create_quantum_circuit(2, 0, "circuit1")
    qasm1 = result1["circuit"]["qasm"]
    result1 = await add_gates_to_circuit(qasm1, "h 0")
    qasm1 = result1["circuit"]["qasm"]

    result2 = await create_quantum_circuit(1, 0, "circuit2")
    qasm2 = result2["circuit"]["qasm"]
    result2 = await add_gates_to_circuit(qasm2, "x 0")
    qasm2 = result2["circuit"]["qasm"]

    # Tensor circuits
    result = await tensor_circuits(qasm1, qasm2)
    assert result["status"] == "success"
    assert result["circuit"]["num_qubits"] == 3  # 2 + 1


@pytest.mark.asyncio
async def test_create_parametric_circuit():
    """Test parametric circuit creation."""
    result = await create_parametric_circuit(2, "theta,phi", 0)
    assert result["status"] == "success"
    assert "theta" in result["parameters"]
    assert "phi" in result["parameters"]


@pytest.mark.asyncio
async def test_decompose_circuit():
    """Test circuit decomposition."""
    result = await create_quantum_circuit(2, 0)
    qasm = result["circuit"]["qasm"]
    result = await add_gates_to_circuit(qasm, "h 0; cx 0 1")
    qasm = result["circuit"]["qasm"]

    result = await decompose_circuit(qasm, reps=1)
    assert result["status"] == "success"
    assert "decomposed_depth" in result


@pytest.mark.asyncio
async def test_add_controlled_gate():
    """Test adding controlled gates."""
    result = await create_quantum_circuit(3, 0)
    qasm = result["circuit"]["qasm"]

    # Add CX gate
    result = await add_controlled_gate(qasm, "x", "0", "1")
    assert result["status"] == "success"
    qasm = result["circuit"]["qasm"]

    # Add CCX gate
    result = await add_controlled_gate(qasm, "x", "0,1", "2")
    assert result["status"] == "success"


@pytest.mark.asyncio
async def test_add_power_gate():
    """Test adding power gates."""
    result = await create_quantum_circuit(1, 0)
    qasm = result["circuit"]["qasm"]

    # Add sqrt(X) gate
    result = await add_power_gate(qasm, "x", 0.5, 0)
    assert result["status"] == "success"
    assert result["circuit"]["num_qubits"] == 1


@pytest.mark.asyncio
async def test_create_qft_circuit():
    """Test QFT circuit creation."""
    result = await create_qft_circuit(3, inverse=False, do_swaps=True)
    assert result["status"] == "success"
    assert result["circuit"]["num_qubits"] == 3
    assert "QFT" in result["circuit"]["name"]

    # Test inverse QFT
    result = await create_qft_circuit(3, inverse=True)
    assert result["status"] == "success"
    assert "IQFT" in result["circuit"]["name"]


@pytest.mark.asyncio
async def test_create_efficient_su2():
    """Test EfficientSU2 circuit creation."""
    result = await create_efficient_su2(2, reps=2, entanglement="linear")
    assert result["status"] == "success"
    assert result["circuit"]["num_qubits"] == 2
    assert result["circuit"]["num_parameters"] > 0
    assert len(result["parameters"]) > 0


@pytest.mark.asyncio
async def test_create_real_amplitudes():
    """Test RealAmplitudes circuit creation."""
    result = await create_real_amplitudes(2, reps=2, entanglement="full")
    assert result["status"] == "success"
    assert result["circuit"]["num_qubits"] == 2
    assert result["circuit"]["num_parameters"] > 0


@pytest.mark.asyncio
async def test_simulate_with_aer():
    """Test Aer simulation."""
    # Create a Bell state circuit
    result = await create_quantum_circuit(2, 2)
    qasm = result["circuit"]["qasm"]
    result = await add_gates_to_circuit(qasm, "h 0; cx 0 1; measure 0 0; measure 1 1")
    qasm = result["circuit"]["qasm"]

    # Simulate
    result = await simulate_with_aer(qasm, shots=100)
    assert result["status"] == "success"
    assert "counts" in result
    assert result["shots"] == 100


@pytest.mark.asyncio
async def test_get_unitary_matrix():
    """Test getting unitary matrix."""
    result = await create_quantum_circuit(2, 0)
    qasm = result["circuit"]["qasm"]
    result = await add_gates_to_circuit(qasm, "h 0; cx 0 1")
    qasm = result["circuit"]["qasm"]

    result = await get_unitary_matrix(qasm)
    assert result["status"] == "success"
    assert result["num_qubits"] == 2
    assert result["dimension"] == 4
    assert result["is_unitary"] is True


@pytest.mark.asyncio
async def test_analyze_circuit():
    """Test circuit analysis."""
    result = await create_quantum_circuit(3, 0)
    qasm = result["circuit"]["qasm"]
    result = await add_gates_to_circuit(qasm, "h 0; h 1; h 2; cx 0 1; cx 1 2")
    qasm = result["circuit"]["qasm"]

    result = await analyze_circuit(qasm)
    assert result["status"] == "success"
    assert result["num_qubits"] == 3
    assert result["depth"] > 0
    assert result["size"] == 5  # 3 H gates + 2 CX gates
    assert "gate_counts" in result
    assert result["gate_counts"]["h"] == 3
    assert result["gate_counts"]["cx"] == 2
    assert result["two_qubit_gate_count"] == 2


@pytest.mark.asyncio
async def test_get_circuit_instructions():
    """Test getting circuit instructions."""
    result = await create_quantum_circuit(2, 0)
    qasm = result["circuit"]["qasm"]
    result = await add_gates_to_circuit(qasm, "h 0; cx 0 1; rx 1.57 1")
    qasm = result["circuit"]["qasm"]

    result = await get_circuit_instructions(qasm)
    assert result["status"] == "success"
    assert result["num_instructions"] == 3
    assert len(result["instructions"]) == 3

    # Check instruction structure
    inst = result["instructions"][0]
    assert "index" in inst
    assert "gate" in inst
    assert "qubits" in inst
    assert "params" in inst


@pytest.mark.asyncio
async def test_convert_to_qasm3():
    """Test QASM 3 conversion."""
    result = await create_quantum_circuit(2, 2)
    qasm2 = result["circuit"]["qasm"]
    result = await add_gates_to_circuit(qasm2, "h 0; cx 0 1")
    qasm2 = result["circuit"]["qasm"]

    result = await convert_to_qasm3(qasm2)
    assert result["status"] == "success"
    assert "qasm3" in result
    assert "qasm2" in result
    assert "OPENQASM 3" in result["qasm3"] or "OPENQASM 2" in result["qasm2"]


@pytest.mark.asyncio
async def test_draw_circuit_text():
    """Test text circuit drawing."""
    result = await create_quantum_circuit(2, 0)
    qasm = result["circuit"]["qasm"]
    result = await add_gates_to_circuit(qasm, "h 0; cx 0 1")
    qasm = result["circuit"]["qasm"]

    result = await draw_circuit_text(qasm)
    assert result["status"] == "success"
    assert result["format"] == "text"
    assert "drawing" in result
    assert len(result["drawing"]) > 0
    assert result["circuit_info"]["num_qubits"] == 2


@pytest.mark.asyncio
async def test_draw_circuit_text_with_fold():
    """Test text circuit drawing with folding."""
    result = await create_quantum_circuit(2, 0)
    qasm = result["circuit"]["qasm"]
    result = await add_gates_to_circuit(qasm, "h 0; h 1; cx 0 1; cx 1 0; h 0; h 1")
    qasm = result["circuit"]["qasm"]

    result = await draw_circuit_text(qasm, fold=20)
    assert result["status"] == "success"
    assert "drawing" in result


# Assisted by watsonx Code Assistant
