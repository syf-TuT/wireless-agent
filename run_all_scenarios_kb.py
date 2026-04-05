#!/usr/bin/env python3
"""
运行脚本：依次处理5个ray_tracing_results并保存结果 (KB版本)

使用方法:
    python run_all_scenarios_kb.py

或者单独运行某个场景:
    python run_all_scenarios_kb.py --scenario TJU_north
"""

import argparse
import sys
import os
import importlib

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 添加 with_knowledge_base 目录到Python路径 (用于导入 rag_system)
kb_dir = os.path.join(project_root, "with_knowledge_base")
sys.path.insert(0, kb_dir)

# 模型名称（用于输出文件名）
MODEL_NAME = "minimax-m2.5"

# 输出目录
OUTPUT_DIR = r"F:\code\wirelessagent\run_results\batch_run\kb\minimax-m2.5"

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 5个场景配置
SCENARIOS = {
    # "TJU_north": {
    #     "ray_tracing_csv": r"F:\code\wirelessagent\ray_tracing_results\ray_tracing_results_north.csv",
    #     "export_file": os.path.join(OUTPUT_DIR, f"network_slicing_results_TJU_north_{MODEL_NAME}.csv")
    # },
    # "TJU_south": {
    #     "ray_tracing_csv": r"F:\code\wirelessagent\ray_tracing_results\ray_tracing_results_south.csv",
    #     "export_file": os.path.join(OUTPUT_DIR, f"network_slicing_results_TJU_south_{MODEL_NAME}.csv")
    # },
    # "TJU_east": {
    #     "ray_tracing_csv": r"F:\code\wirelessagent\ray_tracing_results\ray_tracing_results_east.csv",
    #     "export_file": os.path.join(OUTPUT_DIR, f"network_slicing_results_TJU_east_{MODEL_NAME}.csv")
    # },
    "TJU_west": {
        "ray_tracing_csv": r"F:\code\wirelessagent\ray_tracing_results\ray_tracing_results_west.csv",
        "export_file": os.path.join(OUTPUT_DIR, f"network_slicing_results_TJU_west_{MODEL_NAME}.csv")
    },
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

    # 读取源文件并替换路径
    source_file = os.path.join(project_root, "with_knowledge_base", "WA_DS_V3_KB.py")
    with open(source_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 保存原始内容
    original_content = content

    # 替换ray_tracing_csv路径和export_file
    import re

    # 转义路径中的反斜杠
    ray_tracing_path = scenario["ray_tracing_csv"].replace("\\", "\\\\")
    export_path = scenario["export_file"].replace("\\", "\\\\")

    # 替换 ray_tracing_csv 变量
    content = re.sub(
        r'ray_tracing_csv = r"[^"]*"',
        f'ray_tracing_csv = r"{ray_tracing_path}"',
        content
    )

    # 替换 llm 模型选择
    content = re.sub(
        r'get_llm\("[^"]*"\)',
        f'get_llm("{MODEL_NAME}")',
        content
    )

    # 替换 main() 调用中的 export_file 参数
    content = re.sub(
        r'main\(num_users=\d+, export_file=r"[^"]*"\)',
        f'main(num_users={num_users}, export_file=r"{export_path}")',
        content
    )

    # 写回文件
    with open(source_file, 'w', encoding='utf-8') as f:
        f.write(content)

    try:
        # 运行修改后的脚本
        # 注意：需要使用 importlib.reload() 确保每次都重新加载模块，以清空全局状态
        import with_knowledge_base.WA_DS_V3_KB as wa_module
        importlib.reload(wa_module)
        WA_DS_V3_KB = wa_module

        # 清空全局状态
        WA_DS_V3_KB.reset_token_stats()
        WA_DS_V3_KB.reset_network_state()

        WA_DS_V3_KB.main(num_users=num_users, export_file=scenario["export_file"])
        print(f"\n✓ {scenario_name} 完成! 结果已保存到: {scenario['export_file']}")
    finally:
        # 恢复原始内容
        with open(source_file, 'w', encoding='utf-8') as f:
            f.write(original_content)


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
    parser = argparse.ArgumentParser(description="运行网络切片场景 (KB版本)")
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

    args = parser.parse_args()

    if args.scenario:
        run_scenario(args.scenario, args.num_users)
    else:
        run_all_scenarios(args.num_users)


if __name__ == "__main__":
    main()
