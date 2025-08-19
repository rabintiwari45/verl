# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Preprocess the ARC BARC dataset to Parquet format
"""

import re
import os
import datasets
import argparse


data_source = "barc"


def extract_text_between_backticks(text: str) -> str:
    """
    Extracts text between triple backticks (```) from a string.

    Args:
        text: The input string.

    Returns:
        A string with the first text found between triple backticks.
        Returns an empty string if no triple backticks are found.
    """
    item = re.findall(r'```\n(.*?)```', text, re.DOTALL)
    if item:
        return item[0]
    return ""


def extract_solution(solution_str: str) -> str:
    """
    Extracts the final numeric solution from a solution string with '#### ' prefix.

    Args:
        solution_str: The input solution string.

    Returns:
        A string containing the numeric solution.
    """
    solution = re.search(r"#### (\-?[0-9\.\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split('#### ')[1].replace(',', '')
    return final_solution


instruction_following = "Let's think step by step and output the final answer within '```...```'."


def make_map_fn(split: str):
    """
    Returns a processing function for mapping over dataset items.

    Args:
        split: Dataset split name ('train' or 'test').

    Returns:
        Function to process each dataset example.
    """

    def process_fn(example, idx):
        question_raw = example.get("messages")[0].get("content") + " " + example.get("messages")[1].get("content")
        question = question_raw + ' ' + instruction_following

        answer_raw = example.get("messages")[-1].get("content")
        solution = extract_text_between_backticks(answer_raw)

        data = {
            "data_source": data_source,
            "prompt": [{
                "role": "user",
                "content": question,
            }],
            "ability": "reasoning",
            "reward_model": {
                "style": "rule",
                "ground_truth": solution
            },
            "extra_info": {
                'split': split,
                'index': idx,
            }
        }
        return data

    return process_fn


def main(local_dir: str):
    dataset = datasets.load_dataset(
        "barc0/transduction_100k-gpt4-description-gpt4omini-code_generated_problems_messages_format_0.3"
    )

    # Select subset for training and testing
    train_dataset = dataset['train_sft'].select(range(128))
    test_dataset = dataset['test_sft'].select(range(50))

    # Apply mapping
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    # Remove original messages column
    train_dataset = train_dataset.remove_columns(['messages'])
    test_dataset = test_dataset.remove_columns(['messages'])

    # Save to parquet
    os.makedirs(local_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess ARC BARC dataset to Parquet format")
    parser.add_argument("--output_dir", type=str, default="/content/verl/data", help="Output directory for parquet files")
    args = parser.parse_args()

    main(args.output_dir)
