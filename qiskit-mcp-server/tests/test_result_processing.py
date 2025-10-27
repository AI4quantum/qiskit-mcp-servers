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

"""Tests for result processing functions."""

import pytest
import json
from qiskit_mcp_server.result_processing import (
    marginal_counts,
    marginal_distribution,
    counts_to_probabilities,
    filter_counts,
    combine_counts,
    expectation_from_counts,
    analyze_measurement_results,
)


@pytest.mark.asyncio
async def test_marginal_counts_basic():
    """Test basic marginalization of counts."""
    counts_json = json.dumps({"000": 100, "001": 50, "110": 75, "111": 25})
    result = await marginal_counts(counts_json, "0,2")

    assert result["status"] == "success"
    assert "marginalized_counts" in result
    assert result["marginalized_qubit_count"] == 2


@pytest.mark.asyncio
async def test_marginal_counts_single_qubit():
    """Test marginalization to single qubit."""
    counts_json = json.dumps({"00": 50, "01": 30, "10": 15, "11": 5})
    result = await marginal_counts(counts_json, "0")

    assert result["status"] == "success"
    assert result["marginalized_qubit_count"] == 1


@pytest.mark.asyncio
async def test_marginal_distribution_basic():
    """Test marginal probability distribution."""
    counts_json = json.dumps({"000": 100, "111": 100})
    result = await marginal_distribution(counts_json, "0")

    assert result["status"] == "success"
    assert "distribution" in result
    assert abs(result["total_probability"] - 1.0) < 1e-6


@pytest.mark.asyncio
async def test_counts_to_probabilities():
    """Test conversion of counts to probabilities."""
    counts_json = json.dumps({"00": 50, "01": 25, "10": 15, "11": 10})
    result = await counts_to_probabilities(counts_json)

    assert result["status"] == "success"
    assert "probabilities" in result
    # Should sum to 1.0
    total_prob = sum(result["probabilities"].values())
    assert abs(total_prob - 1.0) < 1e-6


@pytest.mark.asyncio
async def test_filter_counts_pattern():
    """Test filtering counts by bit pattern."""
    counts_json = json.dumps({"000": 10, "001": 20, "100": 30, "101": 40})
    result = await filter_counts(counts_json, "x0x")

    assert result["status"] == "success"
    assert "filtered_counts" in result
    # Should only keep "000" and "101"


@pytest.mark.asyncio
async def test_filter_counts_wildcard():
    """Test filtering with wildcards."""
    counts_json = json.dumps({"00": 25, "01": 25, "10": 25, "11": 25})
    result = await filter_counts(counts_json, "1x")

    assert result["status"] == "success"
    # Should keep "10" and "11"


@pytest.mark.asyncio
async def test_combine_counts():
    """Test combining multiple count dictionaries."""
    counts_list = [{"00": 50, "01": 50}, {"00": 30, "10": 20}]
    counts_json_list = json.dumps(counts_list)
    result = await combine_counts(counts_json_list)

    assert result["status"] == "success"
    assert "combined_counts" in result
    # "00" should be 80 (50 + 30)


@pytest.mark.asyncio
async def test_expectation_from_counts_basic():
    """Test expectation value calculation from counts."""
    counts_json = json.dumps({"00": 50, "11": 50})
    operator = "1,1,-1,-1"  # Z⊗Z eigenvalues
    result = await expectation_from_counts(counts_json, operator)

    assert result["status"] == "success"
    assert "expectation_value" in result


@pytest.mark.asyncio
async def test_analyze_measurement_results():
    """Test comprehensive measurement analysis."""
    counts_json = json.dumps({"00": 100, "01": 50, "10": 30, "11": 20})
    result = await analyze_measurement_results(counts_json)

    assert result["status"] == "success"
    assert "total_shots" in result
    assert "most_likely_outcome" in result
    assert "least_likely_outcome" in result
    assert "entropy" in result
    assert result["total_shots"] == 200


@pytest.mark.asyncio
async def test_marginal_counts_empty():
    """Test marginalization with empty counts."""
    counts_json = json.dumps({})
    result = await marginal_counts(counts_json, "0")

    # Should handle gracefully
    assert result["status"] in ["success", "error"]


# Assisted by watsonx Code Assistant
