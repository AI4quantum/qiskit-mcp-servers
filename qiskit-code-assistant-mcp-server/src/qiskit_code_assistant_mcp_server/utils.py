import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import httpx

from qiskit_code_assistant_mcp_server.constants import (
    QCA_REQUEST_TIMEOUT,
    QCA_TOOL_X_CALLER,
)


def _get_token_from_system():
    token = os.getenv("QISKIT_IBM_TOKEN")

    if not token:
        qiskit_file = Path.home() / ".qiskit" / "qiskit-ibm.json"
        if not qiskit_file.exists():
            raise Exception(
                f"Credentials file {qiskit_file} does not exist. Please set env var QISKIT_IBM_TOKEN to access the service, or save your IBM Quantum API token using QiskitRuntimeService. "
                "More info about saving your token using QiskitRuntimeService https://docs.quantum.ibm.com/api/qiskit-ibm-runtime/qiskit_ibm_runtime.QiskitRuntimeService#save_account"
            )

        with open(qiskit_file, "r") as _sc:
            creds = json.loads(_sc.read())

        token = creds.get("default-ibm-quantum", {}).get("token")
        if token is None:
            raise Exception(
                f"default-ibm-quantum not found in {qiskit_file}. Please set env var QISKIT_IBM_TOKEN to access the service, or save your IBM Quantum API token using QiskitRuntimeService. "
                "More info about saving your token using QiskitRuntimeService https://docs.quantum.ibm.com/api/qiskit-ibm-runtime/qiskit_ibm_runtime.QiskitRuntimeService#save_account"
            )

    return token


QISKIT_IBM_TOKEN = _get_token_from_system()

# Shared async client for better performance
_client: Optional[httpx.AsyncClient] = None


def get_http_client() -> httpx.AsyncClient:
    """Get or create the shared HTTP client."""
    global _client
    if _client is None or _client.is_closed:
        headers = {
            "x-caller": QCA_TOOL_X_CALLER,
            "Accept": "application/json",
            "Authorization": f"Bearer {QISKIT_IBM_TOKEN}",
        }
        _client = httpx.AsyncClient(
            headers=headers,
            timeout=httpx.Timeout(QCA_REQUEST_TIMEOUT),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )
    return _client


async def close_http_client():
    """Close the shared HTTP client."""
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
        _client = None


def get_error_message(response: httpx.Response) -> str:
    msg = "Unable to fetch Qiskit Code Assistant or Qiskit Code Assistant failed"

    if not response.is_success:
        try:
            json_msg = response.json() | {}
            msg = json_msg.get("detail", response.text)

            if response.status_code in [401, 403]:
                msg = f"Qiskit Code Assistant API Token is not authorized or is incorrect: {msg}"

        except Exception:
            msg = response.text

    return msg


async def make_qca_request(
    url: str,
    method: str,
    params: Optional[Dict[str, str]] = None,
    body: Optional[Dict[str, Any]] = None,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """Make an async request to the Qiskit Code Assistant with proper error handling and retry logic."""
    import asyncio

    client = get_http_client()
    last_exception: Optional[
        Union[httpx.TimeoutException, httpx.ConnectError, Exception]
    ] = None
    response = None

    for attempt in range(max_retries):
        try:
            response = await client.request(method, url, params=params, json=body)
            response.raise_for_status()
            return response.json()

        except httpx.TimeoutException as e:
            last_exception = e
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # Exponential backoff
                await asyncio.sleep(wait_time)
                continue

        except httpx.ConnectError as e:
            last_exception = e
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                await asyncio.sleep(wait_time)
                continue

        except httpx.HTTPStatusError as e:
            # Don't retry on HTTP errors (4xx, 5xx)
            return {"error": get_error_message(e.response)}

        except Exception as e:
            last_exception = e
            break

    # If we get here, all retries failed
    if response is not None:
        return {"error": get_error_message(response)}
    else:
        return {
            "error": f"Request failed after {max_retries} attempts: {str(last_exception)}"
        }
