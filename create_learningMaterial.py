"""
SPECIFICATION, do not remove this documentation block.
This `.py` script expects as input a path to an input JSON file. The input JSON file has the following format:

```json
{
    "Domains": {
        "Domain 1": [
            {
                "task-statement": "Task Statement 1.1 ..."
            },
            {
                "task-statement": "Task Statement 1.2 ..."
            },
            {
                "task-statement": "Task Statement 1.3 ..."
            },
            {
                "task-statement": "Task Statement 1.4 ..."
            }
        ],
        ...
        "Domain 4": [
            {
                "task-statement": "Task Statement 4.1 ..."
            },
            {
                "task-statement": "Task Statement 4.2 ..."
            },
            {
                "task-statement": "Task Statement 4.3 ..."
            }
        ]
    },
    "Technologies": [ "APIs", "Benefits of migrating to the AWS Cloud", ... ],
    "AWS-Services": [
        "Analytics",
        "Amazon Athena",
        "Amazon EMR",
        ...
    ]
}
```

The script

* uses: LM Studio for prompting an LLM (notice the lm-studio.utils.py), various libraries such as `json`, `os`, `argparse`, `time`, `datetime`, etc.

to accomplish the following:

1. Iterate over all domains and their associated task statements. It prompts an LLM (e.g., GPT-4) to generate learning material for each task statement. Prompt: *You are an expert AWS cloud architect and trainer. Create a concise and informative learning material for the following task statement: {task-statement}. The learning material should be a comprehensive answer to the task statement.*

2. Iterate over all technologies and prompt the LLM to generate learning material for each technology. Prompt: *You are an expert AWS expert and trainer. Create a concise and informative learning material for the following technology: {technology}. The learning material should cover key concepts, use cases, and best practices.*

3. Iterate over all AWS‑services and prompt the LLM to generate learning material for each AWS service. Prompt: *You are an expert AWS expert and trainer. Create a concise and informative learning material for the following AWS service: {aws-service}. The learning material should cover key concepts, use cases, and best practices.*

Store the generated learning materials in the folder `LearningMaterials` in an output JSON file with the following format:

```json
{
    "Domains": {
        "Domain 1": [
            {
                "task-statement": "Task Statement 1.1 ...",
                "learning-material": "Generated learning material for Task Statement 1.1 ..."
            },
            ...
        ],
        ...
    },
    "Technologies": [
        {
            "technology": "APIs",
            "learning-material": "Generated learning material for APIs ..."
        },
        ...
    ],
    "AWS-Services": [
        {
            "aws-service": "Analytics",
            "learning-material": "Generated learning material for Analytics ..."
        },
        ...
    ]
}
```

The name of the output JSON file is the same as the name of the input JSON file.


"""

import json
import os
import argparse
import time
from tqdm import tqdm
from lm_studio_utils import prompt_nemotron

# Prompt templates
TASK_STATEMENT_PROMPT = "You are an expert AWS cloud architect and trainer. Create a concise and informative learning material for the following task statement: {task_statement}. The learning material should be a comprehensive answer to the task statement."

TECHNOLOGY_PROMPT = "You are an expert AWS expert and trainer. Create a concise and informative learning material for the following technology: {technology}. The learning material should cover key concepts, use cases, and best practices."

AWS_SERVICE_PROMPT = "You are an expert AWS expert and trainer. Create a concise and informative learning material for the following AWS service: {aws_service}. The learning material should cover key concepts, use cases, and best practices."


def load_input(path: str) -> dict:
    """Load and return JSON from input file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_or_create_output(output_path: str, input_data: dict) -> dict:
    """Load existing output file for resume, or create skeleton structure."""
    if os.path.exists(output_path):
        print(f"Resuming from existing output file: {output_path}")
        with open(output_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Create skeleton structure
    output_data = {
        "Domains": {},
        "Technologies": [],
        "AWS-Services": []
    }

    # Copy domain structure
    for domain_name, tasks in input_data.get("Domains", {}).items():
        output_data["Domains"][domain_name] = [
            {"task-statement": task["task-statement"]}
            for task in tasks
        ]

    return output_data


def save_output(output_path: str, data: dict) -> None:
    """Write JSON to output file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def generate_with_retry(prompt: str, max_retries: int = 3) -> str | None:
    """Call prompt_nemotron with retry logic."""
    for attempt in range(max_retries):
        try:
            response = prompt_nemotron(prompt, include_reasoning=False)
            return response
        except Exception as e:
            print(f"  Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
    return None


def process_domains(input_data: dict, output_data: dict, output_path: str, test_mode: bool = False) -> None:
    """Process all domains and task statements."""
    print("\n=== Processing Domains ===")

    domains = list(input_data.get("Domains", {}).items())
    if test_mode:
        domains = domains[:1]  # Only first domain in test mode

    for domain_name, tasks in domains:
        print(f"\nDomain: {domain_name}")

        # Ensure domain exists in output
        if domain_name not in output_data["Domains"]:
            output_data["Domains"][domain_name] = [
                {"task-statement": task["task-statement"]}
                for task in tasks
            ]

        if test_mode:
            tasks = tasks[:1]  # Only first task in test mode

        for i, task in enumerate(tqdm(tasks, desc=f"  {domain_name}", leave=True)):
            task_statement = task["task-statement"]
            output_task = output_data["Domains"][domain_name][i]

            # Skip if already processed
            if "learning-material" in output_task:
                continue

            prompt = TASK_STATEMENT_PROMPT.format(task_statement=task_statement)
            response = generate_with_retry(prompt)

            if response:
                output_task["learning-material"] = response
            else:
                output_task["learning-material"] = "[FAILED TO GENERATE]"

            save_output(output_path, output_data)


def process_technologies(input_data: dict, output_data: dict, output_path: str, test_mode: bool = False) -> None:
    """Process all technologies."""
    print("\n=== Processing Technologies ===")

    technologies = input_data.get("Technologies", [])
    if test_mode:
        technologies = technologies[:1]  # Only first technology in test mode

    # Build lookup of already processed technologies
    processed = {item["technology"] for item in output_data["Technologies"] if "learning-material" in item}

    for tech in tqdm(technologies, desc="  Technologies", leave=True):
        if tech in processed:
            continue

        prompt = TECHNOLOGY_PROMPT.format(technology=tech)
        response = generate_with_retry(prompt)

        if response:
            output_data["Technologies"].append({
                "technology": tech,
                "learning-material": response
            })
        else:
            output_data["Technologies"].append({
                "technology": tech,
                "learning-material": "[FAILED TO GENERATE]"
            })

        save_output(output_path, output_data)


def process_aws_services(input_data: dict, output_data: dict, output_path: str, test_mode: bool = False) -> None:
    """Process all AWS services."""
    print("\n=== Processing AWS Services ===")

    services = input_data.get("AWS-Services", [])
    if test_mode:
        services = services[:1]  # Only first service in test mode

    # Build lookup of already processed services
    processed = {item["aws-service"] for item in output_data["AWS-Services"] if "learning-material" in item}

    for service in tqdm(services, desc="  AWS Services", leave=True):
        if service in processed:
            continue

        prompt = AWS_SERVICE_PROMPT.format(aws_service=service)
        response = generate_with_retry(prompt)

        if response:
            output_data["AWS-Services"].append({
                "aws-service": service,
                "learning-material": response
            })
        else:
            output_data["AWS-Services"].append({
                "aws-service": service,
                "learning-material": "[FAILED TO GENERATE]"
            })

        save_output(output_path, output_data)


def main():
    parser = argparse.ArgumentParser(
        description="Generate AWS learning materials from input JSON using LM Studio"
    )
    parser.add_argument("input_path", help="Path to input JSON file")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: process only 1 domain task, 1 technology, 1 service")
    args = parser.parse_args()

    # Derive output path
    filename = os.path.basename(args.input_path)
    if args.test:
        # Use different output file for test mode
        name, ext = os.path.splitext(filename)
        filename = f"{name}_test{ext}"
    output_path = os.path.join("LearningMaterials", filename)

    print(f"Input: {args.input_path}")
    print(f"Output: {output_path}")
    if args.test:
        print("Mode: TEST (1 domain task, 1 technology, 1 service)")

    # Load input and output
    input_data = load_input(args.input_path)
    output_data = load_or_create_output(output_path, input_data)

    # Process all sections
    process_domains(input_data, output_data, output_path, test_mode=args.test)
    process_technologies(input_data, output_data, output_path, test_mode=args.test)
    process_aws_services(input_data, output_data, output_path, test_mode=args.test)

    print(f"\nDone! Output saved to {output_path}")


if __name__ == "__main__":
    main()
