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

"""Result processing utilities for manipulating measurement outcomes."""

import logging
from typing import Any, Dict
import json

logger = logging.getLogger(__name__)


# ============================================================================
# Result Processing
# ============================================================================


async def marginal_counts(counts_json: str, indices: str) -> Dict[str, Any]:
    """Marginalize counts to specific qubit indices.

    Args:
        counts_json: JSON string of counts (e.g., '{"000": 100, "111": 200}')
        indices: Comma-separated qubit indices to keep (e.g., "0,2")

    Returns:
        Marginalized counts
    """
    try:
        from qiskit.result import marginal_counts as qiskit_marginal_counts

        counts = json.loads(counts_json)
        indices_list = [int(i.strip()) for i in indices.split(",")]

        # Convert to proper counts format
        marginalized = qiskit_marginal_counts(counts, indices_list)

        return {
            "status": "success",
            "message": f"Marginalized to qubits {indices_list}",
            "marginalized_counts": dict(marginalized),
            "original_qubit_count": len(list(counts.keys())[0]) if counts else 0,
            "marginalized_qubit_count": len(indices_list),
        }
    except Exception as e:
        logger.error(f"Failed to marginalize counts: {e}")
        return {"status": "error", "message": f"Failed to marginalize counts: {str(e)}"}


async def marginal_distribution(counts_json: str, indices: str) -> Dict[str, Any]:
    """Compute marginal probability distribution.

    Args:
        counts_json: JSON string of counts
        indices: Comma-separated qubit indices to keep

    Returns:
        Marginal probability distribution
    """
    try:
        from qiskit.result import marginal_distribution as qiskit_marginal_distribution

        counts = json.loads(counts_json)
        indices_list = [int(i.strip()) for i in indices.split(",")]

        marginalized = qiskit_marginal_distribution(counts, indices_list)

        # Convert counts to probabilities
        total_shots = sum(marginalized.values())
        probabilities = {
            key: count / total_shots for key, count in marginalized.items()
        }

        return {
            "status": "success",
            "message": f"Computed marginal distribution for qubits {indices_list}",
            "distribution": probabilities,
            "total_probability": sum(probabilities.values()),
        }
    except Exception as e:
        logger.error(f"Failed to compute marginal distribution: {e}")
        return {
            "status": "error",
            "message": f"Failed to compute marginal distribution: {str(e)}",
        }


async def counts_to_probabilities(counts_json: str) -> Dict[str, Any]:
    """Convert counts to probability distribution.

    Args:
        counts_json: JSON string of counts

    Returns:
        Probability distribution
    """
    try:
        counts = json.loads(counts_json)

        total_shots = sum(counts.values())
        if total_shots == 0:
            return {
                "status": "error",
                "message": "Total shots is zero",
            }

        probabilities = {
            outcome: count / total_shots for outcome, count in counts.items()
        }

        return {
            "status": "success",
            "message": f"Converted {total_shots} shots to probabilities",
            "probabilities": probabilities,
            "total_shots": total_shots,
            "num_outcomes": len(probabilities),
        }
    except Exception as e:
        logger.error(f"Failed to convert to probabilities: {e}")
        return {
            "status": "error",
            "message": f"Failed to convert to probabilities: {str(e)}",
        }


async def filter_counts(counts_json: str, pattern: str) -> Dict[str, Any]:
    """Filter counts by bit pattern.

    Args:
        counts_json: JSON string of counts
        pattern: Bit pattern to match (use 'x' for wildcard, e.g., "1x0")

    Returns:
        Filtered counts
    """
    try:
        counts = json.loads(counts_json)

        def matches_pattern(outcome: str, pattern: str) -> bool:
            if len(outcome) != len(pattern):
                return False
            for o, p in zip(outcome, pattern):
                if p != "x" and o != p:
                    return False
            return True

        filtered = {
            outcome: count
            for outcome, count in counts.items()
            if matches_pattern(outcome, pattern)
        }

        return {
            "status": "success",
            "message": f"Filtered counts by pattern '{pattern}'",
            "filtered_counts": filtered,
            "original_count": len(counts),
            "filtered_count": len(filtered),
            "total_shots": sum(filtered.values()),
        }
    except Exception as e:
        logger.error(f"Failed to filter counts: {e}")
        return {"status": "error", "message": f"Failed to filter counts: {str(e)}"}


async def combine_counts(counts_list_json: str) -> Dict[str, Any]:
    """Combine multiple count dictionaries.

    Args:
        counts_list_json: JSON array of count dicts (e.g., '[{"00": 10}, {"00": 5, "11": 3}]')

    Returns:
        Combined counts
    """
    try:
        counts_list = json.loads(counts_list_json)

        combined: dict[str, int] = {}
        for counts in counts_list:
            for outcome, count in counts.items():
                combined[outcome] = combined.get(outcome, 0) + count

        return {
            "status": "success",
            "message": f"Combined {len(counts_list)} count dictionaries",
            "combined_counts": combined,
            "total_shots": sum(combined.values()),
            "num_outcomes": len(combined),
        }
    except Exception as e:
        logger.error(f"Failed to combine counts: {e}")
        return {"status": "error", "message": f"Failed to combine counts: {str(e)}"}


async def expectation_from_counts(
    counts_json: str, observable: str = "Z"
) -> Dict[str, Any]:
    """Calculate expectation value from measurement counts.

    Args:
        counts_json: JSON string of counts
        observable: Observable to measure (Z, X, Y, ZZ, etc.)

    Returns:
        Expectation value
    """
    try:
        counts = json.loads(counts_json)

        total_shots = sum(counts.values())
        if total_shots == 0:
            return {"status": "error", "message": "Total shots is zero"}

        # For single-qubit Z observable
        if observable == "Z":
            exp_val = 0.0
            for outcome, count in counts.items():
                # Assume last bit is the measured qubit
                if outcome[-1] == "0":
                    exp_val += count
                else:
                    exp_val -= count
            exp_val /= total_shots

            return {
                "status": "success",
                "message": f"Calculated expectation value for {observable}",
                "expectation_value": exp_val,
                "observable": observable,
                "total_shots": total_shots,
            }

        # For multi-qubit ZZ observable
        elif observable.startswith("Z") and len(observable) > 1:
            num_z = len(observable)
            exp_val = 0.0

            for outcome, count in counts.items():
                # Count parity of the relevant bits
                parity = sum(int(outcome[-(i + 1)]) for i in range(num_z)) % 2
                if parity == 0:
                    exp_val += count
                else:
                    exp_val -= count

            exp_val /= total_shots

            return {
                "status": "success",
                "message": f"Calculated expectation value for {observable}",
                "expectation_value": exp_val,
                "observable": observable,
                "total_shots": total_shots,
            }

        else:
            return {
                "status": "error",
                "message": f"Observable {observable} not supported. Use Z or ZZ.",
            }

    except Exception as e:
        logger.error(f"Failed to calculate expectation value: {e}")
        return {
            "status": "error",
            "message": f"Failed to calculate expectation value: {str(e)}",
        }


async def analyze_measurement_results(counts_json: str) -> Dict[str, Any]:
    """Perform comprehensive analysis of measurement results.

    Args:
        counts_json: JSON string of counts

    Returns:
        Analysis including entropy, most probable state, etc.
    """
    try:
        import math

        counts = json.loads(counts_json)

        total_shots = sum(counts.values())
        if total_shots == 0:
            return {"status": "error", "message": "Total shots is zero"}

        # Find most probable state
        most_probable = max(counts.items(), key=lambda x: x[1])

        # Calculate Shannon entropy
        probabilities = [c / total_shots for c in counts.values()]
        entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)

        # Calculate purity (sum of p^2)
        purity = sum(p**2 for p in probabilities)

        return {
            "status": "success",
            "message": "Analyzed measurement results",
            "total_shots": total_shots,
            "num_outcomes": len(counts),
            "most_probable_state": most_probable[0],
            "most_probable_count": most_probable[1],
            "most_probable_probability": most_probable[1] / total_shots,
            "shannon_entropy": entropy,
            "purity": purity,
            "all_probabilities": {k: v / total_shots for k, v in counts.items()},
        }
    except Exception as e:
        logger.error(f"Failed to analyze results: {e}")
        return {"status": "error", "message": f"Failed to analyze results: {str(e)}"}


# Assisted by watsonx Code Assistant
