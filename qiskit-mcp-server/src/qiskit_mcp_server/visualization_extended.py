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

"""Extended visualization including state plots, Bloch sphere, and distribution plots."""

import logging
from typing import Any, Dict
import io
import base64

from qiskit import qasm2
from qiskit.quantum_info import Statevector

logger = logging.getLogger(__name__)


# ============================================================================
# State Visualization
# ============================================================================


async def plot_bloch_multivector(circuit_qasm: str) -> Dict[str, Any]:
    """Plot state on Bloch sphere (multi-qubit).

    Args:
        circuit_qasm: QASM representation of circuit

    Returns:
        Base64 encoded PNG image
    """
    try:
        from qiskit.visualization import plot_bloch_multivector
        import matplotlib.pyplot as plt

        circuit = qasm2.loads(circuit_qasm)
        circuit.remove_final_measurements(inplace=True)

        # Get statevector
        state = Statevector.from_instruction(circuit)

        # Create plot
        fig = plot_bloch_multivector(state)

        # Save to base64
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)

        return {
            "status": "success",
            "format": "bloch_multivector",
            "num_qubits": circuit.num_qubits,
            "image_base64": img_base64,
        }
    except ImportError:
        return {
            "status": "error",
            "message": "Visualization not available. Install with: pip install qiskit[visualization]",
        }
    except Exception as e:
        logger.error(f"Failed to plot Bloch multivector: {e}")
        return {
            "status": "error",
            "message": f"Failed to plot Bloch multivector: {str(e)}",
        }


async def plot_state_qsphere(circuit_qasm: str) -> Dict[str, Any]:
    """Plot state on Q-sphere.

    Args:
        circuit_qasm: QASM representation of circuit

    Returns:
        Base64 encoded PNG image
    """
    try:
        from qiskit.visualization import plot_state_qsphere
        import matplotlib.pyplot as plt

        circuit = qasm2.loads(circuit_qasm)
        circuit.remove_final_measurements(inplace=True)

        state = Statevector.from_instruction(circuit)

        fig = plot_state_qsphere(state)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)

        return {
            "status": "success",
            "format": "qsphere",
            "num_qubits": circuit.num_qubits,
            "image_base64": img_base64,
        }
    except ImportError:
        return {
            "status": "error",
            "message": "Visualization not available. Install with: pip install qiskit[visualization]",
        }
    except Exception as e:
        logger.error(f"Failed to plot Q-sphere: {e}")
        return {"status": "error", "message": f"Failed to plot Q-sphere: {str(e)}"}


async def plot_state_hinton(circuit_qasm: str) -> Dict[str, Any]:
    """Plot state as Hinton diagram.

    Args:
        circuit_qasm: QASM representation of circuit

    Returns:
        Base64 encoded PNG image
    """
    try:
        from qiskit.visualization import plot_state_hinton
        import matplotlib.pyplot as plt

        circuit = qasm2.loads(circuit_qasm)
        circuit.remove_final_measurements(inplace=True)

        state = Statevector.from_instruction(circuit)

        fig = plot_state_hinton(state)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)

        return {
            "status": "success",
            "format": "hinton",
            "num_qubits": circuit.num_qubits,
            "image_base64": img_base64,
        }
    except ImportError:
        return {
            "status": "error",
            "message": "Visualization not available. Install with: pip install qiskit[visualization]",
        }
    except Exception as e:
        logger.error(f"Failed to plot Hinton diagram: {e}")
        return {
            "status": "error",
            "message": f"Failed to plot Hinton diagram: {str(e)}",
        }


async def plot_state_city(circuit_qasm: str) -> Dict[str, Any]:
    """Plot state as city/bar plot.

    Args:
        circuit_qasm: QASM representation of circuit

    Returns:
        Base64 encoded PNG image
    """
    try:
        from qiskit.visualization import plot_state_city
        import matplotlib.pyplot as plt

        circuit = qasm2.loads(circuit_qasm)
        circuit.remove_final_measurements(inplace=True)

        state = Statevector.from_instruction(circuit)

        fig = plot_state_city(state)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)

        return {
            "status": "success",
            "format": "city",
            "num_qubits": circuit.num_qubits,
            "image_base64": img_base64,
        }
    except ImportError:
        return {
            "status": "error",
            "message": "Visualization not available. Install with: pip install qiskit[visualization]",
        }
    except Exception as e:
        logger.error(f"Failed to plot state city: {e}")
        return {"status": "error", "message": f"Failed to plot state city: {str(e)}"}


async def plot_state_paulivec(circuit_qasm: str) -> Dict[str, Any]:
    """Plot state as Pauli vector.

    Args:
        circuit_qasm: QASM representation of circuit

    Returns:
        Base64 encoded PNG image
    """
    try:
        from qiskit.visualization import plot_state_paulivec
        import matplotlib.pyplot as plt

        circuit = qasm2.loads(circuit_qasm)
        circuit.remove_final_measurements(inplace=True)

        state = Statevector.from_instruction(circuit)

        fig = plot_state_paulivec(state)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)

        return {
            "status": "success",
            "format": "paulivec",
            "num_qubits": circuit.num_qubits,
            "image_base64": img_base64,
        }
    except ImportError:
        return {
            "status": "error",
            "message": "Visualization not available. Install with: pip install qiskit[visualization]",
        }
    except Exception as e:
        logger.error(f"Failed to plot Pauli vector: {e}")
        return {"status": "error", "message": f"Failed to plot Pauli vector: {str(e)}"}


# ============================================================================
# Distribution Visualization
# ============================================================================


async def plot_histogram(counts_json: str) -> Dict[str, Any]:
    """Plot measurement counts as histogram.

    Args:
        counts_json: JSON string of counts (e.g., '{"00": 500, "11": 524}')

    Returns:
        Base64 encoded PNG image
    """
    try:
        from qiskit.visualization import plot_histogram
        import matplotlib.pyplot as plt
        import json

        counts = json.loads(counts_json)

        fig = plot_histogram(counts)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)

        return {
            "status": "success",
            "format": "histogram",
            "num_outcomes": len(counts),
            "image_base64": img_base64,
        }
    except ImportError:
        return {
            "status": "error",
            "message": "Visualization not available. Install with: pip install qiskit[visualization]",
        }
    except Exception as e:
        logger.error(f"Failed to plot histogram: {e}")
        return {"status": "error", "message": f"Failed to plot histogram: {str(e)}"}


async def plot_distribution(
    distribution_json: str, title: str = "Distribution"
) -> Dict[str, Any]:
    """Plot probability distribution.

    Args:
        distribution_json: JSON dict of probabilities (e.g., '{"00": 0.5, "11": 0.5}')
        title: Plot title

    Returns:
        Base64 encoded PNG image
    """
    try:
        from qiskit.visualization import plot_distribution
        import matplotlib.pyplot as plt
        import json

        distribution = json.loads(distribution_json)

        fig = plot_distribution(distribution, title=title)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)

        return {
            "status": "success",
            "format": "distribution",
            "title": title,
            "image_base64": img_base64,
        }
    except ImportError:
        return {
            "status": "error",
            "message": "Visualization not available. Install with: pip install qiskit[visualization]",
        }
    except Exception as e:
        logger.error(f"Failed to plot distribution: {e}")
        return {"status": "error", "message": f"Failed to plot distribution: {str(e)}"}


# Assisted by watsonx Code Assistant
