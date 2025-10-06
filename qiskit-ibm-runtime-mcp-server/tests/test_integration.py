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

"""Integration tests for Qiskit IBM Runtime MCP Server."""

import pytest
from unittest.mock import patch, Mock

from qiskit_ibm_runtime_mcp_server.server import mcp


class TestMCPServerIntegration:
    """Test MCP server integration."""

    @pytest.mark.asyncio
    async def test_server_initialization(self, mock_env_vars):
        """Test that server initializes correctly."""
        # Server should initialize without errors
        assert mcp is not None
        assert mcp.name == "Qiskit IBM Runtime"

    @pytest.mark.asyncio
    async def test_service_initialization_flow(
        self, mock_env_vars, mock_runtime_service
    ):
        """Test service initialization flow."""
        from qiskit_ibm_runtime_mcp_server.ibm_runtime import initialize_service

        with patch(
            "qiskit_ibm_runtime_mcp_server.ibm_runtime.QiskitRuntimeService"
        ) as mock_qrs:
            mock_qrs.return_value = mock_runtime_service

            service = initialize_service()

            assert service == mock_runtime_service


class TestToolIntegration:
    """Test MCP tool integration."""

    @pytest.mark.asyncio
    async def test_setup_and_list_backends_workflow(
        self, mock_env_vars, mock_runtime_service
    ):
        """Test setup account -> list backends workflow."""
        from qiskit_ibm_runtime_mcp_server.ibm_runtime import (
            setup_ibm_quantum_account,
            list_backends,
        )

        with patch(
            "qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service"
        ) as mock_init:
            mock_init.return_value = mock_runtime_service

            # 1. Setup account
            setup_result = await setup_ibm_quantum_account("test_token")
            assert setup_result["status"] == "success"

            # 2. List backends
            backends_result = await list_backends()
            assert backends_result["status"] == "success"
            assert len(backends_result["backends"]) > 0

    @pytest.mark.asyncio
    async def test_backend_analysis_workflow(self, mock_env_vars, mock_runtime_service):
        """Test backend analysis workflow: list -> least busy -> properties."""
        from qiskit_ibm_runtime_mcp_server.ibm_runtime import (
            list_backends,
            least_busy_backend,
            get_backend_properties,
        )

        with patch(
            "qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service"
        ) as mock_init:
            with patch(
                "qiskit_ibm_runtime_mcp_server.ibm_runtime.least_busy"
            ) as mock_least_busy:
                mock_init.return_value = mock_runtime_service

                # Mock least busy backend
                mock_backend = Mock()
                mock_backend.name = "ibm_brisbane"
                mock_backend.num_qubits = 127
                mock_backend.status.return_value = Mock(
                    operational=True, pending_jobs=2, status_msg="active"
                )
                mock_least_busy.return_value = mock_backend

                # 1. List all backends
                backends_result = await list_backends()
                assert backends_result["status"] == "success"

                # 2. Get least busy backend
                least_busy_result = await least_busy_backend()
                assert least_busy_result["status"] == "success"
                backend_name = least_busy_result["backend_name"]

                # 3. Get properties of the least busy backend
                properties_result = await get_backend_properties(backend_name)
                assert properties_result["status"] == "success"

    @pytest.mark.asyncio
    async def test_job_management_workflow(self, mock_env_vars, mock_runtime_service):
        """Test job management workflow: list jobs -> get status -> cancel."""
        from qiskit_ibm_runtime_mcp_server.ibm_runtime import (
            list_my_jobs,
            get_job_status,
            cancel_job,
        )

        with patch(
            "qiskit_ibm_runtime_mcp_server.ibm_runtime.service", mock_runtime_service
        ):
            # 1. List jobs
            jobs_result = await list_my_jobs(5)
            assert jobs_result["status"] == "success"

            if jobs_result["total_jobs"] > 0:
                job_id = jobs_result["jobs"][0]["job_id"]

                # 2. Get job status
                status_result = await get_job_status(job_id)
                assert status_result["status"] == "success"

                # 3. Cancel job (if not already completed)
                cancel_result = await cancel_job(job_id)
                assert cancel_result["status"] == "success"


class TestResourceIntegration:
    """Test MCP resource integration."""

    @pytest.mark.asyncio
    async def test_service_status_resource(self, mock_env_vars, mock_runtime_service):
        """Test ibm_quantum://status resource."""
        from qiskit_ibm_runtime_mcp_server.ibm_runtime import get_service_status

        with patch(
            "qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service"
        ) as mock_init:
            mock_init.return_value = mock_runtime_service

            result = await get_service_status()

            assert "IBM Quantum Service Status" in result
            assert "connected" in result.lower()


class TestErrorHandlingIntegration:
    """Test error handling in integration scenarios."""

    @pytest.mark.asyncio
    async def test_authentication_failure_recovery(self, mock_env_vars):
        """Test recovery from authentication failures."""
        from qiskit_ibm_runtime_mcp_server.ibm_runtime import setup_ibm_quantum_account

        # First call fails with authentication error
        with patch(
            "qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service"
        ) as mock_init:
            mock_init.side_effect = [
                ValueError("Invalid token"),
                Mock(),  # Second call succeeds
            ]

            # First attempt should fail
            result1 = await setup_ibm_quantum_account("invalid_token")
            assert result1["status"] == "error"

            # Reset the mock for second attempt
            mock_init.side_effect = None
            mock_init.return_value = Mock()

            # Second attempt should succeed
            result2 = await setup_ibm_quantum_account("valid_token")
            assert result2["status"] == "success"

    @pytest.mark.asyncio
    async def test_service_unavailable_handling(self, mock_env_vars):
        """Test handling when quantum service is unavailable."""
        from qiskit_ibm_runtime_mcp_server.ibm_runtime import list_backends

        with patch(
            "qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service"
        ) as mock_init:
            mock_init.side_effect = Exception("Service unavailable")

            result = await list_backends()

            assert result["status"] == "error"
            assert "Failed to list backends" in result["message"]

    @pytest.mark.asyncio
    async def test_network_connectivity_issues(
        self, mock_env_vars, mock_runtime_service
    ):
        """Test handling of network connectivity issues."""
        from qiskit_ibm_runtime_mcp_server.ibm_runtime import get_backend_properties

        with patch(
            "qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service"
        ) as mock_init:
            mock_init.return_value = mock_runtime_service
            mock_runtime_service.backend.side_effect = Exception("Network timeout")

            result = await get_backend_properties("ibm_brisbane")

            assert result["status"] == "error"
            assert "Failed to get backend properties" in result["message"]


class TestEndToEndScenarios:
    """Test end-to-end scenarios."""

    @pytest.mark.asyncio
    async def test_complete_backend_exploration(
        self, mock_env_vars, mock_runtime_service
    ):
        """Test complete backend exploration scenario."""
        from qiskit_ibm_runtime_mcp_server.ibm_runtime import (
            setup_ibm_quantum_account,
            list_backends,
            least_busy_backend,
            get_backend_properties,
            get_service_status,
        )

        with patch(
            "qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service"
        ) as mock_init:
            with patch(
                "qiskit_ibm_runtime_mcp_server.ibm_runtime.least_busy"
            ) as mock_least_busy:
                mock_init.return_value = mock_runtime_service

                # Mock least busy backend
                mock_backend = Mock()
                mock_backend.name = "ibm_brisbane"
                mock_backend.num_qubits = 127
                mock_backend.status.return_value = Mock(
                    operational=True, pending_jobs=2, status_msg="active"
                )
                mock_least_busy.return_value = mock_backend

                # 1. Setup account
                setup_result = await setup_ibm_quantum_account("test_token")
                assert setup_result["status"] == "success"

                # 2. Check service status
                status_result = await get_service_status()
                assert "connected" in status_result.lower()

                # 3. List all backends
                backends_result = await list_backends()
                assert backends_result["status"] == "success"

                # 4. Find least busy backend
                least_busy_result = await least_busy_backend()
                assert least_busy_result["status"] == "success"

                # 5. Get detailed properties of recommended backend
                properties_result = await get_backend_properties(
                    least_busy_result["backend_name"]
                )
                assert properties_result["status"] == "success"

    @pytest.mark.asyncio
    async def test_job_monitoring_scenario(self, mock_env_vars, mock_runtime_service):
        """Test job monitoring scenario."""
        from qiskit_ibm_runtime_mcp_server.ibm_runtime import (
            list_my_jobs,
            get_job_status,
        )

        with patch(
            "qiskit_ibm_runtime_mcp_server.ibm_runtime.service", mock_runtime_service
        ):
            # 1. List recent jobs
            jobs_result = await list_my_jobs(10)
            assert jobs_result["status"] == "success"

            # 2. Monitor each job's status
            for job in jobs_result["jobs"]:
                status_result = await get_job_status(job["job_id"])
                assert status_result["status"] == "success"
                assert status_result["job_id"] == job["job_id"]
