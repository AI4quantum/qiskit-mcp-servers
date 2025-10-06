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

"""
Qiskit Code Assistant MCP Server

A Model Context Protocol server that provides access to IBM Qiskit Code Assistant
for intelligent quantum code completion and assistance.

Dependencies:
- fastmcp
- httpx
- python-dotenv
"""

import logging

from fastmcp import FastMCP

from qiskit_code_assistant_mcp_server.constants import (
    QCA_MCP_DEBUG_LEVEL,
    validate_configuration,
)
from qiskit_code_assistant_mcp_server.utils import close_http_client
from qiskit_code_assistant_mcp_server.qca import (
    qca_accept_completion,
    qca_accept_model_disclaimer,
    qca_get_completion,
    qca_get_model,
    qca_get_rag_completion,
    qca_get_model_disclaimer,
    qca_get_service_status,
    qca_list_models,
)


# Configure logging
logging.basicConfig(level=getattr(logging, QCA_MCP_DEBUG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("Qiskit Code Assistant")

logger.info("Qiskit Code Assistant MCP Server initialized")

# Validate configuration on startup
if not validate_configuration():
    logger.error("Configuration validation failed - server may not work correctly")
else:
    logger.info("Configuration validation passed")


##################################################
## MCP Resources
## - https://modelcontextprotocol.io/docs/concepts/resources
##################################################

mcp.resource("qca://models", mime_type="application/json")(qca_list_models)
mcp.resource("qca://model/{model_id}", mime_type="application/json")(qca_get_model)
mcp.resource("qca://disclaimer/{model_id}", mime_type="application/json")(
    qca_get_model_disclaimer
)
mcp.resource("qca://status", mime_type="text/plain")(qca_get_service_status)


##################################################
## MCP Tools
## - https://modelcontextprotocol.io/docs/concepts/tools
##################################################

mcp.tool()(qca_accept_model_disclaimer)
mcp.tool()(qca_get_completion)
mcp.tool()(qca_get_rag_completion)
mcp.tool()(qca_accept_completion)


if __name__ == "__main__":
    import atexit

    logger.info("Starting Qiskit Code Assistant MCP Server")

    # Register cleanup function
    def cleanup():
        import asyncio

        try:
            asyncio.run(close_http_client())
            logger.info("HTTP client closed successfully")
        except Exception as e:
            logger.error(f"Error closing HTTP client: {e}")

    atexit.register(cleanup)

    try:
        mcp.run(transport="stdio")
    except KeyboardInterrupt:
        logger.info("Server interrupted, shutting down...")
    finally:
        cleanup()


# Assisted by watsonx Code Assistant
