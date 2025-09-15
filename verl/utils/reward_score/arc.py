

import re


def extract_solution(solution_str):

    # Optimization: Regular expression matching on very long strings can be slow.
    # For math problems, the final answer is usually at the end.
    # We only match on the last 300 characters, which is a safe approximation for 300 tokens.
        # this also tests the formatting of the model
    solutions = re.findall("```(.*?)```", solution_str, re.DOTALL)
    if len(solutions) == 0:
        final_answer = None
    else:
        # take the last solution
        final_answer = solutions[0]
    return final_answer


def compute_score(solution_str, ground_truth, score=1.0):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual
    Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    # breakpoint()
    if answer is None:
        return 0
    else:
        if answer.strip() == ground_truth.strip():
            return score
        else:
            return 0







