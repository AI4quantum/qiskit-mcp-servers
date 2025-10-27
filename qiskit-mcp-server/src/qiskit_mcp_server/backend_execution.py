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

"""Backend execution utilities for running circuits on quantum hardware and simulators."""

import logging
from typing import Any, Dict

from qiskit import qasm2
from qiskit.primitives import StatevectorSampler

logger = logging.getLogger(__name__)


# ============================================================================
# Backend Information
# ============================================================================


async def list_available_backends(simulator_only: bool = False) -> Dict[str, Any]:
    """List all available quantum backends.

    Args:
        simulator_only: If True, only list simulator backends

    Returns:
        Dictionary containing list of available backends
    """
    try:
        backends = []

        # Always available: Statevector simulators
        backends.append(
            {
                "name": "statevector_simulator",
                "backend_type": "simulator",
                "provider": "qiskit_primitives",
                "num_qubits": 32,
                "description": "Statevector simulator for ideal quantum circuits",
                "coupling_map": None,
                "basis_gates": None,
                "available": True,
            }
        )

        # Try to get IBM Quantum backends
        if not simulator_only:
            try:
                from qiskit_ibm_runtime import QiskitRuntimeService

                try:
                    service = QiskitRuntimeService()
                    ibm_backends = service.backends()

                    for backend in ibm_backends:
                        backends.append(
                            {
                                "name": backend.name,
                                "backend_type": "hardware"
                                if not backend.simulator
                                else "simulator",
                                "provider": "ibm_quantum",
                                "num_qubits": backend.num_qubits,
                                "description": str(backend.description)
                                if hasattr(backend, "description")
                                else "IBM Quantum backend",
                                "coupling_map": backend.coupling_map.get_edges()
                                if hasattr(backend, "coupling_map")
                                and backend.coupling_map
                                else None,
                                "basis_gates": backend.basis_gates
                                if hasattr(backend, "basis_gates")
                                else None,
                                "available": backend.status().operational
                                if hasattr(backend, "status")
                                else True,
                                "pending_jobs": backend.status().pending_jobs
                                if hasattr(backend, "status")
                                else 0,
                            }
                        )
                except Exception as e:
                    logger.warning(f"Could not access IBM Quantum backends: {e}")
                    backends.append(
                        {
                            "name": "ibm_quantum",
                            "backend_type": "note",
                            "provider": "ibm_quantum",
                            "description": f"IBM Quantum backends not accessible: {str(e)}",
                            "available": False,
                        }
                    )
            except ImportError:
                backends.append(
                    {
                        "name": "ibm_quantum",
                        "backend_type": "note",
                        "provider": "ibm_quantum",
                        "description": "IBM Quantum Runtime not installed. Install with: pip install qiskit-ibm-runtime",
                        "available": False,
                    }
                )

        # Try to get Aer backends
        try:
            from qiskit_aer import Aer

            aer_backends = Aer.backends()
            for backend_name in aer_backends:
                backend = Aer.get_backend(backend_name)
                backends.append(
                    {
                        "name": backend_name,
                        "backend_type": "simulator",
                        "provider": "qiskit_aer",
                        "num_qubits": backend.configuration().n_qubits
                        if hasattr(backend.configuration(), "n_qubits")
                        else 32,
                        "description": backend.configuration().description
                        if hasattr(backend.configuration(), "description")
                        else "Aer simulator",
                        "coupling_map": None,
                        "basis_gates": backend.configuration().basis_gates
                        if hasattr(backend.configuration(), "basis_gates")
                        else None,
                        "available": True,
                    }
                )
        except ImportError:
            backends.append(
                {
                    "name": "aer_simulator",
                    "backend_type": "note",
                    "provider": "qiskit_aer",
                    "description": "Qiskit Aer not installed. Install with: pip install qiskit-aer",
                    "available": False,
                }
            )

        return {
            "status": "success",
            "message": f"Found {len(backends)} backends",
            "backends": backends,
            "total_count": len(backends),
            "simulator_only": simulator_only,
        }
    except Exception as e:
        logger.error(f"Failed to list backends: {e}")
        return {"status": "error", "message": f"Failed to list backends: {str(e)}"}


async def get_backend_properties(backend_name: str) -> Dict[str, Any]:
    """Get detailed properties of a specific backend.

    Args:
        backend_name: Name of the backend

    Returns:
        Detailed backend properties
    """
    try:
        backend_info = {
            "name": backend_name,
            "properties": {},
            "configuration": {},
            "status": {},
        }

        # Handle statevector simulator
        if backend_name == "statevector_simulator":
            return {
                "status": "success",
                "backend_name": backend_name,
                "backend_type": "simulator",
                "provider": "qiskit_primitives",
                "num_qubits": 32,
                "basis_gates": ["id", "rz", "sx", "x", "cx", "reset"],
                "coupling_map": None,
                "description": "Statevector simulator - supports all gates",
                "simulator": True,
                "local": True,
            }

        # Try IBM Quantum
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService

            service = QiskitRuntimeService()
            backend = service.backend(backend_name)

            config = backend.configuration()
            status = backend.status()

            backend_info["backend_type"] = (
                "hardware" if not backend.simulator else "simulator"
            )
            backend_info["provider"] = "ibm_quantum"
            backend_info["num_qubits"] = backend.num_qubits
            backend_info["basis_gates"] = (
                config.basis_gates if hasattr(config, "basis_gates") else None
            )
            backend_info["coupling_map"] = (
                backend.coupling_map.get_edges()
                if hasattr(backend, "coupling_map") and backend.coupling_map
                else None
            )
            backend_info["max_shots"] = (
                config.max_shots if hasattr(config, "max_shots") else None
            )
            backend_info["max_experiments"] = (
                config.max_experiments if hasattr(config, "max_experiments") else None
            )
            backend_info["operational"] = status.operational
            backend_info["pending_jobs"] = status.pending_jobs
            backend_info["status_msg"] = (
                status.status_msg if hasattr(status, "status_msg") else None
            )

            return {"status": "success", **backend_info}
        except Exception as e:
            logger.debug(f"Could not access IBM backend {backend_name}: {e}")

        # Try Aer
        try:
            from qiskit_aer import Aer

            backend = Aer.get_backend(backend_name)
            config = backend.configuration()

            return {
                "status": "success",
                "backend_name": backend_name,
                "backend_type": "simulator",
                "provider": "qiskit_aer",
                "num_qubits": config.n_qubits if hasattr(config, "n_qubits") else 32,
                "basis_gates": config.basis_gates
                if hasattr(config, "basis_gates")
                else None,
                "coupling_map": None,
                "description": config.description
                if hasattr(config, "description")
                else "Aer simulator",
                "simulator": True,
                "local": True,
            }
        except Exception as e:
            logger.debug(f"Could not access Aer backend {backend_name}: {e}")

        return {
            "status": "error",
            "message": f"Backend '{backend_name}' not found or not accessible",
        }

    except Exception as e:
        logger.error(f"Failed to get backend properties: {e}")
        return {
            "status": "error",
            "message": f"Failed to get backend properties: {str(e)}",
        }


# ============================================================================
# Job Execution (Local Simulators)
# ============================================================================


async def execute_circuit_local(
    circuit_qasm: str,
    shots: int = 1024,
    backend_name: str = "statevector_simulator",
) -> Dict[str, Any]:
    """Execute circuit on local simulator backend.

    Args:
        circuit_qasm: QASM representation of circuit
        shots: Number of measurement shots
        backend_name: Local backend name (default: statevector_simulator)

    Returns:
        Execution results including counts
    """
    try:
        circuit = qasm2.loads(circuit_qasm)

        if backend_name == "statevector_simulator":
            # Use StatevectorSampler
            sampler = StatevectorSampler()
            job = sampler.run([circuit], shots=shots)
            result = job.result()

            # Extract counts from result
            pub_result = result[0]
            counts = pub_result.data.meas.get_counts()

            return {
                "status": "success",
                "message": f"Circuit executed on {backend_name}",
                "backend": backend_name,
                "shots": shots,
                "counts": dict(counts),
                "num_qubits": circuit.num_qubits,
                "circuit_depth": circuit.depth(),
                "execution_type": "local_simulator",
            }

        # Try Aer backends
        try:
            from qiskit_aer import Aer
            from qiskit import transpile

            backend = Aer.get_backend(backend_name)
            transpiled = transpile(circuit, backend)
            job = backend.run(transpiled, shots=shots)
            result = job.result()
            counts = result.get_counts()

            return {
                "status": "success",
                "message": f"Circuit executed on {backend_name}",
                "backend": backend_name,
                "shots": shots,
                "counts": dict(counts),
                "num_qubits": circuit.num_qubits,
                "circuit_depth": circuit.depth(),
                "execution_type": "aer_simulator",
            }
        except ImportError:
            return {
                "status": "error",
                "message": f"Backend '{backend_name}' requires Aer. Install with: pip install qiskit-aer",
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Backend '{backend_name}' not available: {str(e)}",
            }

    except Exception as e:
        logger.error(f"Failed to execute circuit: {e}")
        return {"status": "error", "message": f"Failed to execute circuit: {str(e)}"}


async def submit_job_to_ibm(
    circuit_qasm: str,
    backend_name: str,
    shots: int = 1024,
) -> Dict[str, Any]:
    """Submit job to IBM Quantum backend.

    Args:
        circuit_qasm: QASM representation of circuit
        backend_name: IBM backend name
        shots: Number of measurement shots

    Returns:
        Job submission information
    """
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

        circuit = qasm2.loads(circuit_qasm)

        # Get service and backend
        service = QiskitRuntimeService()
        backend = service.backend(backend_name)

        # Transpile for backend
        from qiskit import transpile

        transpiled = transpile(circuit, backend, optimization_level=3)

        # Submit job using Sampler primitive
        sampler = Sampler(backend)
        job = sampler.run([transpiled], shots=shots)

        return {
            "status": "success",
            "message": f"Job submitted to {backend_name}",
            "job_id": job.job_id(),
            "backend": backend_name,
            "shots": shots,
            "num_qubits": circuit.num_qubits,
            "circuit_depth": transpiled.depth(),
            "transpiled_depth": transpiled.depth(),
            "execution_type": "ibm_quantum",
            "instructions": "Use retrieve_job_result tool with this job_id to get results",
        }

    except ImportError:
        return {
            "status": "error",
            "message": "IBM Quantum Runtime not installed. Install with: pip install qiskit-ibm-runtime",
        }
    except Exception as e:
        logger.error(f"Failed to submit job: {e}")
        return {"status": "error", "message": f"Failed to submit job to IBM: {str(e)}"}


async def retrieve_job_result(job_id: str) -> Dict[str, Any]:
    """Retrieve results from a submitted IBM Quantum job.

    Args:
        job_id: Job ID from submit_job_to_ibm

    Returns:
        Job results if completed
    """
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService

        service = QiskitRuntimeService()
        job = service.job(job_id)

        status = job.status()

        if status.name == "DONE":
            result = job.result()
            pub_result = result[0]
            counts = pub_result.data.meas.get_counts()

            return {
                "status": "success",
                "job_id": job_id,
                "job_status": "completed",
                "counts": dict(counts),
                "execution_time": job.metrics()["usage"]["seconds"]
                if hasattr(job, "metrics")
                else None,
            }
        else:
            return {
                "status": "pending",
                "job_id": job_id,
                "job_status": status.name,
                "message": f"Job is {status.name}. Check again later.",
                "queue_position": job.queue_position()
                if hasattr(job, "queue_position")
                else None,
            }

    except ImportError:
        return {
            "status": "error",
            "message": "IBM Quantum Runtime not installed. Install with: pip install qiskit-ibm-runtime",
        }
    except Exception as e:
        logger.error(f"Failed to retrieve job result: {e}")
        return {"status": "error", "message": f"Failed to retrieve job: {str(e)}"}


async def cancel_job(job_id: str) -> Dict[str, Any]:
    """Cancel a submitted IBM Quantum job.

    Args:
        job_id: Job ID to cancel

    Returns:
        Cancellation status
    """
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService

        service = QiskitRuntimeService()
        job = service.job(job_id)

        if job.status().name in ["DONE", "CANCELLED", "ERROR"]:
            return {
                "status": "error",
                "message": f"Job {job_id} is already {job.status().name} and cannot be cancelled",
            }

        job.cancel()

        return {
            "status": "success",
            "message": f"Job {job_id} cancelled successfully",
            "job_id": job_id,
        }

    except ImportError:
        return {"status": "error", "message": "IBM Quantum Runtime not installed"}
    except Exception as e:
        logger.error(f"Failed to cancel job: {e}")
        return {"status": "error", "message": f"Failed to cancel job: {str(e)}"}


async def get_job_status(job_id: str) -> Dict[str, Any]:
    """Get status of a submitted IBM Quantum job.

    Args:
        job_id: Job ID to check

    Returns:
        Job status information
    """
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService

        service = QiskitRuntimeService()
        job = service.job(job_id)

        status = job.status()

        return {
            "status": "success",
            "job_id": job_id,
            "job_status": status.name,
            "queue_position": job.queue_position()
            if hasattr(job, "queue_position") and callable(job.queue_position)
            else None,
            "creation_date": str(job.creation_date)
            if hasattr(job, "creation_date")
            else None,
            "backend": job.backend().name if hasattr(job, "backend") else None,
        }

    except ImportError:
        return {"status": "error", "message": "IBM Quantum Runtime not installed"}
    except Exception as e:
        logger.error(f"Failed to get job status: {e}")
        return {"status": "error", "message": f"Failed to get job status: {str(e)}"}


async def estimate_circuit_cost(
    circuit_qasm: str,
    backend_name: str,
) -> Dict[str, Any]:
    """Estimate resource cost for running circuit on a backend.

    Args:
        circuit_qasm: QASM representation of circuit
        backend_name: Backend to estimate for

    Returns:
        Estimated resource requirements
    """
    try:
        from qiskit import transpile

        circuit = qasm2.loads(circuit_qasm)

        # Get backend
        if backend_name == "statevector_simulator":
            return {
                "status": "success",
                "backend": backend_name,
                "estimated_cost": "free",
                "estimated_time": "< 1 second",
                "circuit_depth": circuit.depth(),
                "num_qubits": circuit.num_qubits,
                "gate_count": circuit.size(),
            }

        # Try IBM backend
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService

            service = QiskitRuntimeService()
            backend = service.backend(backend_name)

            # Transpile to get accurate gate count
            transpiled = transpile(circuit, backend, optimization_level=3)

            return {
                "status": "success",
                "backend": backend_name,
                "backend_type": "ibm_quantum",
                "original_depth": circuit.depth(),
                "original_gates": circuit.size(),
                "transpiled_depth": transpiled.depth(),
                "transpiled_gates": transpiled.size(),
                "num_qubits": transpiled.num_qubits,
                "coupling_map_required": True,
                "estimated_queue_time": "varies by queue position",
                "note": "Actual cost depends on your IBM Quantum plan",
            }
        except Exception as e:
            logger.debug(f"Could not estimate for IBM backend: {e}")

        # Try Aer
        try:
            from qiskit_aer import Aer

            backend = Aer.get_backend(backend_name)
            transpiled = transpile(circuit, backend)

            return {
                "status": "success",
                "backend": backend_name,
                "backend_type": "aer_simulator",
                "estimated_cost": "free",
                "estimated_time": "< 1 minute (depends on circuit complexity)",
                "circuit_depth": transpiled.depth(),
                "num_qubits": transpiled.num_qubits,
                "gate_count": transpiled.size(),
            }
        except Exception as e:
            logger.debug(f"Could not estimate for Aer backend: {e}")

        return {
            "status": "error",
            "message": f"Could not estimate cost for backend '{backend_name}'",
        }

    except Exception as e:
        logger.error(f"Failed to estimate circuit cost: {e}")
        return {"status": "error", "message": f"Failed to estimate cost: {str(e)}"}


# Assisted by watsonx Code Assistant
