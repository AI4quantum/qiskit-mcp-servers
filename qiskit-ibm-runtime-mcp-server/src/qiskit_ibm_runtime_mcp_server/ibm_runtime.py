"""Core IBM Runtime functions for the MCP server."""

import logging
from typing import Any, Dict, Optional

from qiskit_ibm_runtime import QiskitRuntimeService  # type: ignore[import-untyped]


def least_busy(backends):
    """Find the least busy backend from a list of backends."""
    if not backends:
        return None

    operational_backends = [
        b for b in backends if hasattr(b, "status") and b.status().operational
    ]
    if not operational_backends:
        return None

    return min(operational_backends, key=lambda b: b.status().pending_jobs)


logger = logging.getLogger(__name__)

# Global service instance
service: Optional[QiskitRuntimeService] = None


def initialize_service(
    token: Optional[str] = None, channel: str = "ibm_quantum_platform"
) -> QiskitRuntimeService:
    """
    Initialize the Qiskit IBM Runtime service.

    Args:
        token: IBM Quantum API token (optional if saved)
        channel: Service channel ('ibm_quantum_platform')

    Returns:
        QiskitRuntimeService: Initialized service instance
    """
    global service

    try:
        if token:
            # Save account with provided token
            try:
                QiskitRuntimeService.save_account(
                    channel=channel, token=token, overwrite=True
                )
                logger.info(f"Saved IBM Quantum account for channel: {channel}")
            except Exception as e:
                logger.error(f"Failed to save account: {e}")
                raise ValueError("Invalid token or channel") from e

        # Initialize service
        try:
            service = QiskitRuntimeService(channel=channel)
            logger.info(
                f"Successfully initialized IBM Runtime service on channel: {channel}"
            )
            return service
        except Exception as e:
            if "No account" in str(e) and not token:
                raise ValueError("No IBM Quantum token provided") from e
            logger.error(f"Failed to initialize IBM Runtime service: {e}")
            raise

    except Exception as e:
        if not isinstance(e, ValueError):
            logger.error(f"Failed to initialize IBM Runtime service: {e}")
        raise


async def setup_ibm_quantum_account(
    token: str, channel: str = "ibm_quantum_platform"
) -> Dict[str, Any]:
    """
    Set up IBM Quantum account with credentials.

    Args:
        token: IBM Quantum API token
        channel: Service channel ('ibm_quantum_platform')

    Returns:
        Setup status and information
    """
    if not token or not token.strip():
        return {"status": "error", "message": "Token is required and cannot be empty"}

    if channel not in ["ibm_quantum_platform"]:
        return {
            "status": "error",
            "message": "Channel must be 'ibm_quantum_platform'",
        }

    try:
        service_instance = initialize_service(token.strip(), channel)

        # Get backend count for response
        try:
            backends = service_instance.backends()
            backend_count = len(backends)
        except Exception:
            backend_count = 0

        return {
            "status": "success",
            "message": f"IBM Quantum account set up successfully for channel: {channel}",
            "channel": service_instance._channel,
            "available_backends": backend_count,
        }
    except Exception as e:
        logger.error(f"Failed to set up IBM Quantum account: {e}")
        return {"status": "error", "message": f"Failed to set up account: {str(e)}"}


async def list_backends() -> Dict[str, Any]:
    """
    List available IBM Quantum backends.

    Returns:
        List of backends with their properties
    """
    global service

    try:
        if service is None:
            service = initialize_service()

        backends = service.backends()
        backend_list = []

        for backend in backends:
            try:
                status = backend.status()
                backend_info = {
                    "name": backend.name,
                    "num_qubits": getattr(backend, "num_qubits", 0),
                    "simulator": getattr(backend, "simulator", False),
                    "operational": status.operational,
                    "pending_jobs": status.pending_jobs,
                    "status_msg": status.status_msg,
                }
                backend_list.append(backend_info)
            except Exception as be:
                logger.warning(f"Failed to get status for backend {backend.name}: {be}")
                backend_info = {
                    "name": backend.name,
                    "num_qubits": getattr(backend, "num_qubits", 0),
                    "simulator": getattr(backend, "simulator", False),
                    "operational": False,
                    "pending_jobs": 0,
                    "status_msg": "Status unavailable",
                }
                backend_list.append(backend_info)

        return {
            "status": "success",
            "backends": backend_list,
            "total_backends": len(backend_list),
        }

    except Exception as e:
        logger.error(f"Failed to list backends: {e}")
        return {"status": "error", "message": f"Failed to list backends: {str(e)}"}


async def least_busy_backend() -> Dict[str, Any]:
    """
    Find the least busy operational backend.

    Returns:
        Information about the least busy backend
    """
    global service

    try:
        if service is None:
            service = initialize_service()

        backends = service.backends(operational=True, simulator=False)

        if not backends:
            return {
                "status": "error",
                "message": "No operational quantum backends available",
            }

        backend = least_busy(backends)
        status = backend.status()

        return {
            "status": "success",
            "backend_name": backend.name,
            "num_qubits": getattr(backend, "num_qubits", 0),
            "pending_jobs": status.pending_jobs,
            "operational": status.operational,
            "status_msg": status.status_msg,
        }

    except Exception as e:
        logger.error(f"Failed to find least busy backend: {e}")
        return {
            "status": "error",
            "message": f"Failed to find least busy backend: {str(e)}",
        }


async def get_backend_properties(backend_name: str) -> Dict[str, Any]:
    """
    Get detailed properties of a specific backend.

    Args:
        backend_name: Name of the backend

    Returns:
        Backend properties and capabilities
    """
    global service

    try:
        if service is None:
            service = initialize_service()

        backend = service.backend(backend_name)
        status = backend.status()

        # Get configuration
        try:
            config = backend.configuration()
            basis_gates = getattr(config, "basis_gates", [])
            coupling_map = getattr(config, "coupling_map", [])
            max_shots = getattr(config, "max_shots", 0)
            max_experiments = getattr(config, "max_experiments", 0)
        except Exception:
            basis_gates = []
            coupling_map = []
            max_shots = 0
            max_experiments = 0

        properties = {
            "status": "success",
            "backend_name": backend.name,
            "num_qubits": getattr(backend, "num_qubits", 0),
            "simulator": getattr(backend, "simulator", False),
            "operational": status.operational,
            "pending_jobs": status.pending_jobs,
            "status_msg": status.status_msg,
            "basis_gates": basis_gates,
            "coupling_map": coupling_map,
            "max_shots": max_shots,
            "max_experiments": max_experiments,
        }

        return properties

    except Exception as e:
        logger.error(f"Failed to get backend properties: {e}")
        return {
            "status": "error",
            "message": f"Failed to get backend properties: {str(e)}",
        }


async def list_my_jobs(limit: int = 10) -> Dict[str, Any]:
    """
    List user's recent jobs.

    Args:
        limit: Maximum number of jobs to retrieve

    Returns:
        List of jobs with their information
    """
    global service

    try:
        if service is None:
            service = initialize_service()

        jobs = service.jobs(limit=limit)
        job_list = []

        for job in jobs:
            try:
                job_info = {
                    "job_id": job.job_id(),
                    "status": job.status(),
                    "creation_date": getattr(job, "creation_date", "Unknown"),
                    "backend": job.backend().name if job.backend() else "Unknown",
                    "tags": getattr(job, "tags", []),
                    "error_message": job.error_message()
                    if hasattr(job, "error_message")
                    else None,
                }
                job_list.append(job_info)
            except Exception as je:
                logger.warning(f"Failed to get info for job: {je}")
                continue

        return {"status": "success", "jobs": job_list, "total_jobs": len(job_list)}

    except Exception as e:
        logger.error(f"Failed to list jobs: {e}")
        return {"status": "error", "message": f"Failed to list jobs: {str(e)}"}


async def get_job_status(job_id: str) -> Dict[str, Any]:
    """
    Get status of a specific job.

    Args:
        job_id: ID of the job

    Returns:
        Job status information
    """
    global service

    try:
        if service is None:
            return {
                "status": "error",
                "message": "Failed to get job status: service not initialized",
            }

        job = service.job(job_id)

        job_info = {
            "status": "success",
            "job_id": job.job_id(),
            "job_status": job.status(),
            "creation_date": getattr(job, "creation_date", "Unknown"),
            "backend": job.backend().name if job.backend() else "Unknown",
            "tags": getattr(job, "tags", []),
            "error_message": job.error_message()
            if hasattr(job, "error_message")
            else None,
        }

        return job_info

    except Exception as e:
        logger.error(f"Failed to get job status: {e}")
        return {"status": "error", "message": f"Failed to get job status: {str(e)}"}


async def cancel_job(job_id: str) -> Dict[str, Any]:
    """
    Cancel a specific job.

    Args:
        job_id: ID of the job to cancel

    Returns:
        Cancellation status
    """
    global service

    try:
        if service is None:
            return {
                "status": "error",
                "message": "Failed to cancel job: service not initialized",
            }

        job = service.job(job_id)
        job.cancel()

        return {
            "status": "success",
            "job_id": job_id,
            "message": "Job cancellation requested",
        }
    except Exception as e:
        logger.error(f"Failed to cancel job: {e}")
        return {"status": "error", "message": f"Failed to cancel job: {str(e)}"}


async def get_service_status() -> str:
    """
    Get current IBM Quantum service status.

    Returns:
        Service connection status and basic information
    """
    global service

    try:
        if service is None:
            service = initialize_service()

        # Test connectivity by listing backends
        backends = service.backends()
        backend_count = len(backends)

        status_info = {
            "connected": True,
            "channel": service._channel,
            "available_backends": backend_count,
            "service": "IBM Quantum",
        }

        return f"IBM Quantum Service Status: {status_info}"

    except Exception as e:
        logger.error(f"Failed to check service status: {e}")
        status_info = {"connected": False, "error": str(e), "service": "IBM Quantum"}
        return f"IBM Quantum Service Status: {status_info}"
