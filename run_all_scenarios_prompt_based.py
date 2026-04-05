#!/usr/bin/env python3
"""
运行脚本：依次处理5个ray_tracing_results并保存结果 (Prompt Based版本)

使用方法:
    python run_all_scenarios_prompt_based.py

或者单独运行某个场景:
    python run_all_scenarios_prompt_based.py --scenario TJU_north

指定模型:
    python run_all_scenarios_prompt_based.py --model minimax-m2.5
"""

import argparse
import sys
import os
import importlib

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 模型名称（用于输出文件名）- 默认使用 glm-5
MODEL_NAME = "minimax-m2.5"

# 输出目录 (会在main()中根据MODEL_NAME动态设置)
OUTPUT_DIR = r"F:\code\wirelessagent\run_results\batch_run\prompt_based\minimax-m2.5"

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 5个场景配置
SCENARIOS = {
    "TJU_north": {
        "ray_tracing_csv": r"F:\code\wirelessagent\ray_tracing_results\ray_tracing_results_north.csv",
        "export_file": os.path.join(OUTPUT_DIR, f"network_slicing_results_TJU_north_{MODEL_NAME}.csv")
    },
    "TJU_south": {
        "ray_tracing_csv": r"F:\code\wirelessagent\ray_tracing_results\ray_tracing_results_south.csv",
        "export_file": os.path.join(OUTPUT_DIR, f"network_slicing_results_TJU_south_{MODEL_NAME}.csv")
    },
    # "TJU_east": {
    #     "ray_tracing_csv": r"F:\code\wirelessagent\ray_tracing_results\ray_tracing_results_east.csv",
    #     "export_file": os.path.join(OUTPUT_DIR, f"network_slicing_results_TJU_east_{MODEL_NAME}.csv")
    # },
    # "TJU_west": {
    #     "ray_tracing_csv": r"F:\code\wirelessagent\ray_tracing_results\ray_tracing_results_west.csv",
    #     "export_file": os.path.join(OUTPUT_DIR, f"network_slicing_results_TJU_west_{MODEL_NAME}.csv")
    # },
    # "TJU_gym": {
    #     "ray_tracing_csv": r"F:\code\wirelessagent\ray_tracing_results\ray_tracing_results_gym.csv",
    #     "export_file": os.path.join(OUTPUT_DIR, f"network_slicing_results_TJU_gym_{MODEL_NAME}.csv")
    # }
}

# 用户数量
NUM_USERS = 30


def run_scenario(scenario_name, num_users=NUM_USERS):
    """运行单个场景"""
    if scenario_name not in SCENARIOS:
        raise ValueError(f"未知场景: {scenario_name}, 可用: {list(SCENARIOS.keys())}")

    scenario = SCENARIOS[scenario_name]

    # 导入并重载模块
    import Prompt_Based
    importlib.reload(Prompt_Based)

    # 设置模型名称并重新初始化LLM
    Prompt_Based.MODEL_NAME = MODEL_NAME
    Prompt_Based.llm = Prompt_Based.get_llm(MODEL_NAME)

    # 重置全局状态
    Prompt_Based.reset_network_state()

    # 直接调用 main 函数（通过参数传递）
    Prompt_Based.main(
        num_users=num_users,
        export_file=scenario["export_file"],
        ray_tracing_csv=scenario["ray_tracing_csv"]
    )
    print(f"\n[OK] {scenario_name} 完成! 结果已保存到: {scenario['export_file']}")


def run_all_scenarios(num_users=NUM_USERS):
    """依次运行所有场景"""
    print("=" * 60)
    print(f"开始运行所有场景... (模型: {MODEL_NAME})")
    print("=" * 60)

    for i, scenario_name in enumerate(SCENARIOS.keys(), 1):
        print(f"\n{'=' * 60}")
        print(f"场景 {i}/5: {scenario_name}")
        print(f"{'=' * 60}")
        print(f"输入: {SCENARIOS[scenario_name]['ray_tracing_csv']}")
        print(f"输出: {SCENARIOS[scenario_name]['export_file']}")
        print("-" * 60)

        run_scenario(scenario_name, num_users)

    print("\n" + "=" * 60)
    print("所有场景运行完成!")
    print("=" * 60)


def main():
    global MODEL_NAME, OUTPUT_DIR

    parser = argparse.ArgumentParser(description="运行网络切片场景 (Prompt Based版本)")
    parser.add_argument(
        "--scenario",
        type=str,
        choices=list(SCENARIOS.keys()),
        help="指定要运行的场景，默认运行所有场景"
    )
    parser.add_argument(
        "--num-users",
        type=int,
        default=NUM_USERS,
        help=f"用户数量 (默认: {NUM_USERS})"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help=f"LLM模型名称 (默认: {MODEL_NAME})"
    )

    args = parser.parse_args()

    # 更新全局 MODEL_NAME
    MODEL_NAME = args.model

    # 更新输出目录为模型特定目录
    OUTPUT_DIR = os.path.join(r"F:\code\wirelessagent\run_results\batch_run\prompt_based", MODEL_NAME)

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 更新输出目录和文件名中的模型名称
    for key in SCENARIOS:
        SCENARIOS[key]["export_file"] = os.path.join(
            OUTPUT_DIR,
            f"network_slicing_results_{key}_{MODEL_NAME}.csv"
        )

    if args.scenario:
        run_scenario(args.scenario, args.num_users)
    else:
        run_all_scenarios(args.num_users)


if __name__ == "__main__":
    main()
