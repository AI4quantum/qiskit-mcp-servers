"""Unit tests for sync wrapper functions."""

import pytest
from unittest.mock import patch

from qiskit_ibm_runtime_mcp_server.sync import (
    setup_ibm_quantum_account_sync,
    list_backends_sync,
    least_busy_backend_sync,
    get_backend_properties_sync,
    list_my_jobs_sync,
    get_job_status_sync,
    cancel_job_sync,
    get_service_status_sync,
)


class TestSetupIBMQuantumAccountSync:
    """Test setup_ibm_quantum_account_sync function."""

    def test_setup_account_sync_success(self):
        """Test successful account setup with sync wrapper."""
        mock_response = {
            "status": "success",
            "message": "IBM Quantum account set up successfully",
            "channel": "ibm_quantum_platform",
            "available_backends": 10,
        }

        with patch("qiskit_ibm_runtime_mcp_server.sync._run_async") as mock_run:
            mock_run.return_value = mock_response

            result = setup_ibm_quantum_account_sync("test_token")

            assert result["status"] == "success"
            assert result["available_backends"] == 10

    def test_setup_account_sync_empty_token(self):
        """Test validation of empty token."""
        mock_response = {
            "status": "error",
            "message": "Token is required and cannot be empty",
        }

        with patch("qiskit_ibm_runtime_mcp_server.sync._run_async") as mock_run:
            mock_run.return_value = mock_response

            result = setup_ibm_quantum_account_sync("")

            assert result["status"] == "error"


class TestListBackendsSync:
    """Test list_backends_sync function."""

    def test_list_backends_sync_success(self):
        """Test successful backends listing with sync wrapper."""
        mock_response = {
            "status": "success",
            "backends": [
                {
                    "name": "ibm_brisbane",
                    "num_qubits": 133,
                    "simulator": False,
                    "operational": True,
                    "pending_jobs": 5,
                },
                {
                    "name": "ibm_kyoto",
                    "num_qubits": 127,
                    "simulator": False,
                    "operational": True,
                    "pending_jobs": 10,
                },
            ],
            "total_backends": 2,
        }

        with patch("qiskit_ibm_runtime_mcp_server.sync._run_async") as mock_run:
            mock_run.return_value = mock_response

            result = list_backends_sync()

            assert result["status"] == "success"
            assert result["total_backends"] == 2
            assert len(result["backends"]) == 2

    def test_list_backends_sync_error(self):
        """Test error handling in sync wrapper."""
        mock_response = {
            "status": "error",
            "message": "Failed to list backends: service not initialized",
        }

        with patch("qiskit_ibm_runtime_mcp_server.sync._run_async") as mock_run:
            mock_run.return_value = mock_response

            result = list_backends_sync()

            assert result["status"] == "error"


class TestLeastBusyBackendSync:
    """Test least_busy_backend_sync function."""

    def test_least_busy_backend_sync_success(self):
        """Test successful least busy backend retrieval with sync wrapper."""
        mock_response = {
            "status": "success",
            "backend_name": "ibm_brisbane",
            "num_qubits": 133,
            "pending_jobs": 5,
            "operational": True,
        }

        with patch("qiskit_ibm_runtime_mcp_server.sync._run_async") as mock_run:
            mock_run.return_value = mock_response

            result = least_busy_backend_sync()

            assert result["status"] == "success"
            assert result["backend_name"] == "ibm_brisbane"
            assert result["pending_jobs"] == 5

    def test_least_busy_backend_sync_no_backends(self):
        """Test handling when no backends are available."""
        mock_response = {
            "status": "error",
            "message": "No operational quantum backends available",
        }

        with patch("qiskit_ibm_runtime_mcp_server.sync._run_async") as mock_run:
            mock_run.return_value = mock_response

            result = least_busy_backend_sync()

            assert result["status"] == "error"


class TestGetBackendPropertiesSync:
    """Test get_backend_properties_sync function."""

    def test_get_backend_properties_sync_success(self):
        """Test successful backend properties retrieval with sync wrapper."""
        mock_response = {
            "status": "success",
            "backend_name": "ibm_brisbane",
            "num_qubits": 133,
            "simulator": False,
            "operational": True,
            "basis_gates": ["id", "rz", "sx", "x", "cx"],
            "max_shots": 100000,
        }

        with patch("qiskit_ibm_runtime_mcp_server.sync._run_async") as mock_run:
            mock_run.return_value = mock_response

            result = get_backend_properties_sync("ibm_brisbane")

            assert result["status"] == "success"
            assert result["backend_name"] == "ibm_brisbane"
            assert result["num_qubits"] == 133


class TestListMyJobsSync:
    """Test list_my_jobs_sync function."""

    def test_list_my_jobs_sync_success(self):
        """Test successful jobs listing with sync wrapper."""
        mock_response = {
            "status": "success",
            "jobs": [
                {
                    "job_id": "job_123",
                    "status": "DONE",
                    "backend": "ibm_brisbane",
                    "creation_date": "2024-01-01",
                },
                {
                    "job_id": "job_456",
                    "status": "RUNNING",
                    "backend": "ibm_kyoto",
                    "creation_date": "2024-01-02",
                },
            ],
            "total_jobs": 2,
        }

        with patch("qiskit_ibm_runtime_mcp_server.sync._run_async") as mock_run:
            mock_run.return_value = mock_response

            result = list_my_jobs_sync(limit=10)

            assert result["status"] == "success"
            assert result["total_jobs"] == 2
            assert len(result["jobs"]) == 2


class TestGetJobStatusSync:
    """Test get_job_status_sync function."""

    def test_get_job_status_sync_success(self):
        """Test successful job status retrieval with sync wrapper."""
        mock_response = {
            "status": "success",
            "job_id": "job_123",
            "job_status": "DONE",
            "backend": "ibm_brisbane",
            "creation_date": "2024-01-01",
        }

        with patch("qiskit_ibm_runtime_mcp_server.sync._run_async") as mock_run:
            mock_run.return_value = mock_response

            result = get_job_status_sync("job_123")

            assert result["status"] == "success"
            assert result["job_id"] == "job_123"
            assert result["job_status"] == "DONE"


class TestCancelJobSync:
    """Test cancel_job_sync function."""

    def test_cancel_job_sync_success(self):
        """Test successful job cancellation with sync wrapper."""
        mock_response = {
            "status": "success",
            "job_id": "job_123",
            "message": "Job cancellation requested",
        }

        with patch("qiskit_ibm_runtime_mcp_server.sync._run_async") as mock_run:
            mock_run.return_value = mock_response

            result = cancel_job_sync("job_123")

            assert result["status"] == "success"
            assert result["job_id"] == "job_123"


class TestGetServiceStatusSync:
    """Test get_service_status_sync function."""

    def test_get_service_status_sync_success(self):
        """Test successful service status check with sync wrapper."""
        mock_response = "IBM Quantum Service Status: {'connected': True}"

        with patch("qiskit_ibm_runtime_mcp_server.sync._run_async") as mock_run:
            mock_run.return_value = mock_response

            result = get_service_status_sync()

            assert "connected" in result
