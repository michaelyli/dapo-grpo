"""Reward functions for math verification using math_verify and boxed extraction."""

import os

from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig


def last_boxed_only_string(string):
    """Extract the last \\boxed{...} expression from a string."""
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def compute_score(model_output, ground_truth, timeout_score=0.0):
    """Score a single model output against ground truth using math_metric."""
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        ret_score, _ = verify_func([ground_truth_boxed], [model_output])
    except TimeoutException:
        rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "?"))
        print(f"[rank {rank}] TIMEOUT in math_verify for ground_truth={ground_truth[:100]}, "
              f"model_output_len={len(model_output)}")
        ret_score = timeout_score
    except BaseException:
        ret_score = 0.0
    return float(ret_score)


def accuracy_reward(prompts, completions, **kwargs):
    """Reward function using math_verify for mathematical answer verification.

    Args:
        prompts: List of prompt strings (not used but kept for API consistency)
        completions: List of completion strings containing mathematical answers
        **kwargs: Must contain 'solution' key with ground truth answers

    Returns:
        List of float rewards (1.0 for correct, 0.0 for incorrect)
    """
    contents = completions
    rewards = []
    solution = kwargs["solution"]

    for content, sol in zip(contents, solution):
        last_boxed = last_boxed_only_string(content)
        parse_content = last_boxed if last_boxed is not None else content
        reward = compute_score(parse_content, sol)
        rewards.append(reward)

    return rewards
