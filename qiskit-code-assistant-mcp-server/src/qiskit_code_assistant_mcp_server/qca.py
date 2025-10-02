import logging
from typing import Any, Dict

from qiskit_code_assistant_mcp_server.constants import (
    QCA_TOOL_API_BASE,
    QCA_TOOL_MODEL_NAME,
    QCA_REQUEST_TIMEOUT,
)
from qiskit_code_assistant_mcp_server.utils import make_qca_request

logger = logging.getLogger(__name__)


async def qca_list_models() -> Dict[str, Any]:
    """List the available models from the Qiskit Code Assistant."""
    try:
        logger.info("Fetching available models from Qiskit Code Assistant")
        url = f"{QCA_TOOL_API_BASE}/v1/models"
        data = await make_qca_request(url, method="GET")

        if "error" in data:
            logger.error(f"Failed to list models: {data['error']}")
            return {"status": "error", "message": data["error"]}

        models = data.get("data", [])
        if len(models) <= 0:
            logger.warning("No models retrieved from Qiskit Code Assistant")
            return {"status": "error", "message": "No models retrieved."}
        else:
            logger.info(f"Retrieved {len(models)} models from Qiskit Code Assistant")
            return {"status": "success", "models": models}
    except Exception as e:
        logger.error(f"Exception in qca_list_models: {str(e)}")
        return {"status": "error", "message": f"Failed to list models: {str(e)}"}


async def qca_get_model(model_id: str) -> Dict[str, Any]:
    """Get the info for a model from the Qiskit Code Assistant.

    Args:
        model_id: The ID of the model to retrieve
    """
    if not model_id or not model_id.strip():
        return {
            "status": "error",
            "message": "model_id is required and cannot be empty",
        }

    try:
        logger.info(f"Fetching model info for model_id: {model_id}")
        url = f"{QCA_TOOL_API_BASE}/v1/model/{model_id}"
        data = await make_qca_request(url, method="GET")

        if "error" in data:
            logger.error(f"Failed to get model {model_id}: {data['error']}")
            return {"status": "error", "message": data["error"]}
        elif "id" not in data:
            logger.warning(f"Model {model_id} not retrieved - missing ID in response")
            return {"status": "error", "message": "Model not retrieved."}
        else:
            logger.info(f"Successfully retrieved model {model_id}")
            return {"status": "success", "model": data}
    except Exception as e:
        logger.error(f"Exception in qca_get_model for {model_id}: {str(e)}")
        return {"status": "error", "message": f"Failed to get model: {str(e)}"}


async def qca_get_model_disclaimer(model_id: str) -> Dict[str, Any]:
    """Get the disclaimer for a model from the Qiskit Code Assistant.

    Args:
        model_id: The ID of the model for which we want to retrieve the disclaimer
    """
    try:
        url = f"{QCA_TOOL_API_BASE}/v1/model/{model_id}/disclaimer"
        data = await make_qca_request(url, method="GET")

        if "error" in data:
            return {"status": "error", "message": data["error"]}
        elif "id" not in data:
            return {"status": "error", "message": "Model disclaimer not retrieved."}
        else:
            return {"status": "success", "disclaimer": data}
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to get model disclaimer: {str(e)}",
        }


async def qca_accept_model_disclaimer(
    model_id: str, disclaimer_id: str
) -> Dict[str, Any]:
    """
    Accept the disclaimer for an available model from the Qiskit Code Assistant.

    Args:
        model_id: The ID of the model for which we want to accept the disclaimer
        disclaimer_id: The ID of the disclaimer we want to accept

    Returns:
        Disclaimer acceptance status
    """
    try:
        url = f"{QCA_TOOL_API_BASE}/v1/model/{model_id}/disclaimer"
        body = {"disclaimer": disclaimer_id, "accepted": "true"}
        data = await make_qca_request(url, method="POST", body=body)

        if "error" in data:
            return {"status": "error", "message": data["error"]}
        elif "success" not in data:
            return {
                "status": "error",
                "message": "The response does not contain the acceptance result.",
            }
        else:
            return {"status": "success", "result": data}
    except Exception as e:
        return {"status": "error", "message": f"Failed to accept disclaimer: {str(e)}"}


async def qca_get_completion(prompt: str) -> Dict[str, Any]:
    """
    Get completion for writing, completing, and optimizing quantum code using Qiskit.

    Args:
        prompt: The prompt for code completion

    Returns:
        Code completion choices and metadata
    """
    if not prompt or not prompt.strip():
        return {"status": "error", "message": "prompt is required and cannot be empty"}

    if len(prompt.strip()) > 10000:  # Reasonable limit
        return {
            "status": "error",
            "message": "prompt is too long (max 10000 characters)",
        }

    try:
        logger.info(f"Requesting code completion for prompt (length: {len(prompt)})")
        url = f"{QCA_TOOL_API_BASE}/v1/completions"
        body = {"model": QCA_TOOL_MODEL_NAME, "prompt": prompt.strip()}
        data = await make_qca_request(url, method="POST", body=body)

        if "error" in data:
            logger.error(f"Failed to get completion: {data['error']}")
            return {"status": "error", "message": data["error"]}

        choices = data.get("choices")
        completion_id = data.get("id")

        if not choices:
            logger.warning("No choices returned for completion request")
            return {"status": "error", "message": "No choices for this prompt."}
        else:
            logger.info(
                f"Successfully generated completion with {len(choices)} choices (ID: {completion_id})"
            )
            return {
                "status": "success",
                "completion_id": completion_id,
                "choices": choices,
            }
    except Exception as e:
        logger.error(f"Exception in qca_get_completion: {str(e)}")
        return {"status": "error", "message": f"Failed to get completion: {str(e)}"}


async def qca_get_rag_completion(prompt: str) -> Dict[str, Any]:
    """
    Get RAG completion for answering conceptual or descriptive questions about Qiskit or Quantum.

    Args:
        prompt: The prompt for RAG-based completion

    Returns:
        RAG completion choices and metadata
    """
    try:
        url = f"{QCA_TOOL_API_BASE}/v1/completions"
        body = {"model": QCA_TOOL_MODEL_NAME, "prompt": prompt, "mode": "rag"}
        data = await make_qca_request(url, method="POST", body=body)

        if "error" in data:
            return {"status": "error", "message": data["error"]}

        choices = data.get("choices")
        if not choices:
            return {"status": "error", "message": "No choices for this prompt."}
        else:
            return {
                "status": "success",
                "completion_id": data.get("id"),
                "choices": choices,
            }
    except Exception as e:
        return {"status": "error", "message": f"Failed to get RAG completion: {str(e)}"}


async def qca_accept_completion(completion_id: str) -> Dict[str, Any]:
    """
    Accept a suggestion generated by the Qiskit Code Assistant.

    Args:
        completion_id: The ID of the completion to accept

    Returns:
        Completion acceptance status
    """
    try:
        url = f"{QCA_TOOL_API_BASE}/v1/completion/acceptance"
        body = {"completion": completion_id, "accepted": "true"}
        data = await make_qca_request(url, method="POST", body=body)

        if "error" in data:
            return {"status": "error", "message": data["error"]}

        result = data.get("result")
        if not result:
            return {
                "status": "error",
                "message": "No result for this completion acceptance.",
            }
        else:
            return {"status": "success", "result": result}
    except Exception as e:
        return {"status": "error", "message": f"Failed to accept completion: {str(e)}"}


async def qca_get_service_status() -> str:
    """
    Get current Qiskit Code Assistant service status.

    Returns:
        Service connection status and basic information
    """
    try:
        logger.info("Checking Qiskit Code Assistant service status")
        # Try to get models to test connectivity
        models_result = await qca_list_models()

        if models_result.get("status") == "success":
            model_count = len(models_result.get("models", []))
            status_info = {
                "connected": True,
                "api_base": QCA_TOOL_API_BASE,
                "model_name": QCA_TOOL_MODEL_NAME,
                "available_models": model_count,
                "timeout": QCA_REQUEST_TIMEOUT,
            }
            logger.info("Qiskit Code Assistant service is accessible")
        else:
            status_info = {
                "connected": False,
                "api_base": QCA_TOOL_API_BASE,
                "model_name": QCA_TOOL_MODEL_NAME,
                "error": models_result.get("message", "Unknown error"),
                "timeout": QCA_REQUEST_TIMEOUT,
            }
            logger.warning("Qiskit Code Assistant service is not accessible")

        return f"Qiskit Code Assistant Service Status: {status_info}"
    except Exception as e:
        logger.error(f"Failed to check service status: {str(e)}")
        return f"Qiskit Code Assistant Service Status: Error - {str(e)}"
