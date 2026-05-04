#!/usr/bin/env python3
"""
Batch runner: execute all ray tracing scenarios for NKB mode.
"""

import argparse
import importlib
import os
import re
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

MODEL_NAME = "kimi-k2.5"
OUTPUT_DIR = r"F:\code\wirelessagent\run_results\batch_run\nkb\kimi-k2.5"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SCENARIOS = {
    "TJU_north": {
        "ray_tracing_csv": r"F:\code\wirelessagent\ray_tracing_results\ray_tracing_results_north.csv",
        "export_file": os.path.join(OUTPUT_DIR, f"network_slicing_results_TJU_north_{MODEL_NAME}.csv"),
    },
    "TJU_south": {
        "ray_tracing_csv": r"F:\code\wirelessagent\ray_tracing_results\ray_tracing_results_south.csv",
        "export_file": os.path.join(OUTPUT_DIR, f"network_slicing_results_TJU_south_{MODEL_NAME}.csv"),
    },
    "TJU_east": {
        "ray_tracing_csv": r"F:\code\wirelessagent\ray_tracing_results\ray_tracing_results_east.csv",
        "export_file": os.path.join(OUTPUT_DIR, f"network_slicing_results_TJU_east_{MODEL_NAME}.csv"),
    },
    "TJU_west": {
        "ray_tracing_csv": r"F:\code\wirelessagent\ray_tracing_results\ray_tracing_results_west.csv",
        "export_file": os.path.join(OUTPUT_DIR, f"network_slicing_results_TJU_west_{MODEL_NAME}.csv"),
    },
    "TJU_gym": {
        "ray_tracing_csv": r"F:\code\wirelessagent\ray_tracing_results\ray_tracing_results_gym.csv",
        "export_file": os.path.join(OUTPUT_DIR, f"network_slicing_results_TJU_gym_{MODEL_NAME}.csv"),
    },
}

NUM_USERS = 30


def run_scenario(scenario_name: str, num_users: int = NUM_USERS) -> None:
    if scenario_name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_name}, available: {list(SCENARIOS.keys())}")

    scenario = SCENARIOS[scenario_name]
    source_file = os.path.join(project_root, "no_knowledge_base", "WA_NKB.py")

    with open(source_file, "r", encoding="utf-8") as f:
        content = f.read()
    original_content = content

    ray_tracing_path = scenario["ray_tracing_csv"].replace("\\", "\\\\")
    export_path = scenario["export_file"].replace("\\", "\\\\")

    content = re.sub(r'ray_tracing_csv = r"[^"]*"', f'ray_tracing_csv = r"{ray_tracing_path}"', content)
    content = re.sub(r'get_llm\("[^"]*"\)', f'get_llm("{MODEL_NAME}")', content)
    content = re.sub(
        r'main\(num_users=\d+, export_file=r"[^"]*"\)',
        f'main(num_users={num_users}, export_file=r"{export_path}")',
        content,
    )

    with open(source_file, "w", encoding="utf-8") as f:
        f.write(content)

    try:
        import no_knowledge_base.WA_NKB as wa_module
        importlib.reload(wa_module)

        wa_module.reset_token_stats()
        wa_module.reset_network_state()
        wa_module.main(num_users=num_users, export_file=scenario["export_file"])

        print(f"\n[OK] {scenario_name} done. Output: {scenario['export_file']}")
    finally:
        with open(source_file, "w", encoding="utf-8") as f:
            f.write(original_content)


def run_all_scenarios(num_users: int = NUM_USERS) -> None:
    for scenario_name in SCENARIOS.keys():
        run_scenario(scenario_name, num_users)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run network slicing scenarios (NKB)")
    parser.add_argument("--scenario", type=str, choices=list(SCENARIOS.keys()), help="Run one scenario only")
    parser.add_argument("--num-users", type=int, default=NUM_USERS, help=f"User count (default: {NUM_USERS})")
    args = parser.parse_args()

    if args.scenario:
        run_scenario(args.scenario, args.num_users)
    else:
        run_all_scenarios(args.num_users)


if __name__ == "__main__":
    main()
