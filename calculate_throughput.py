"""
Rule-Based 吞吐量计算脚本
基于 docs/Rule_Based_Throughput_Calculation.md 规格实现
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 切片参数
# =============================================================================
SLICE_PARAMS = {
    'eMBB': {
        'B': 90,      # 总带宽 MHz
        'B_min': 6,   # 最小带宽 MHz
        'B_max': 20,  # 最大带宽 MHz
        'R_min': 100, # 最小速率 Mbps
        'R_max': 400, # 最大速率 Mbps
    },
    'URLLC': {
        'B': 30,
        'B_min': 1,
        'B_max': 5,
        'R_min': 1,
        'R_max': 100,
    },
    'mMTC': {
        'B': 10,
        'B_min': 0.1,
        'B_max': 1.0,
        'R_min': 0.01,
        'R_max': 1.0,
    },
}

ALPHA = 10.0  # 增益因子


def calculate_rate(b, q, alpha=ALPHA):
    """计算速率: rate = alpha * b * log10(1 + 10^(q/10))"""
    return alpha * b * np.log10(1 + 10 ** (q / 10))


def is_allocation_feasible(B_alloc, Q, params):
    """检查资源分配是否满足约束"""
    B, B_min, B_max, R_min, R_max = params['B'], params['B_min'], params['B_max'], params['R_min'], params['R_max']

    # 总带宽约束
    if abs(np.sum(B_alloc) - B) > 1e-6:
        return False

    # 带宽边界约束
    if np.any(B_alloc < B_min - 1e-6) or np.any(B_alloc > B_max + 1e-6):
        return False

    # 速率约束
    rates = np.array([calculate_rate(b, q) for b, q in zip(B_alloc, Q)])
    if np.any(rates < R_min - 1e-6) or np.any(rates > R_max + 1e-6):
        return False

    return True


def objective(B_alloc):
    """目标函数: 最大化比例公平性 -> 最小化 -sum(log(rate))"""
    rates = np.array([calculate_rate(b, q) for b, q in zip(B_alloc, Q_global)])
    return -np.sum(np.log(rates + 1e-10))


Q_global = None  # 全局变量用于目标函数


def solve_resource_allocation(B, M, Q, params):
    """
    求解资源分配优化问题

    参数:
        B: 总带宽 (MHz)
        M: 用户数
        Q: CQI 数组 (长度 M)
        params: 切片参数字典

    返回:
        B_alloc: 分配的带宽数组
        rates: 速率数组
        total_rate: 总吞吐量
    """
    global Q_global
    Q_global = Q.copy()

    B_min, B_max, R_min, R_max = params['B_min'], params['B_max'], params['R_min'], params['R_max']

    # 检查可行性：每个用户至少分配B_min，总需求 M*B_min <= B
    if M * B_min > B:
        # 不可行：返回最小分配
        B_alloc = np.full(M, B / M)
        rates = np.array([calculate_rate(b, q) for b, q in zip(B_alloc, Q)])
        return B_alloc, rates, np.sum(rates)

    # 初始化：每个用户分配B_min
    initial_allocation = np.full(M, B_min)
    remaining_bandwidth = B - M * B_min

    # 计算达到R_min所需的最小额外带宽
    min_extra_bandwidth = 0
    for i in range(M):
        # 找到刚好达到R_min的带宽
        for b_test in np.linspace(0, B_max - B_min, 1000):
            if calculate_rate(B_min + b_test, Q[i]) >= R_min:
                min_extra_bandwidth += b_test
                break

    # 如果剩余带宽足够满足所有用户的R_min
    if remaining_bandwidth >= min_extra_bandwidth:
        # 按CQI权重分配额外带宽
        extra_bw = remaining_bandwidth - min_extra_bandwidth
        cqi_weights = Q / np.sum(Q)
        extra_per_user = extra_bw * cqi_weights

        # 分配额外带宽，但不能超过B_max
        for i in range(M):
            possible_extra = min(extra_per_user[i], B_max - B_min)
            initial_allocation[i] += possible_extra

    # 使用SLSQP优化
    bounds = [(B_min, B_max) for _ in range(M)]

    # 确保初始值在边界内
    initial_allocation = np.clip(initial_allocation, B_min, B_max)

    result = minimize(
        objective,
        initial_allocation,
        method='SLSQP',
        bounds=bounds,
        constraints=[
            {'type': 'eq', 'fun': lambda x: np.sum(x) - B}  # 总带宽约束
        ],
        options={'maxiter': 1000, 'ftol': 1e-9}
    )

    if result.success:
        B_alloc = result.x
    else:
        # 优化失败，使用按CQI权重分配
        B_alloc = np.full(M, B / M)

    # 确保边界约束
    B_alloc = np.clip(B_alloc, B_min, B_max)

    # 重新归一化以满足总带宽约束
    B_alloc = B_alloc * (B / np.sum(B_alloc))

    rates = np.array([calculate_rate(b, q) for b, q in zip(B_alloc, Q)])

    # 应用速率约束
    rates = np.clip(rates, R_min, R_max)

    return B_alloc, rates, np.sum(rates)


def calculate_for_n_users(df, n):
    """计算前n个用户的总吞吐量

    按RX_ID排序后取前n个用户，然后按切片类型分组处理
    """
    result = {
        'n_users': n,
        'total_throughput': 0,
        'bandwidth_utilization': 0,
        'max_rate': 0,
        'slices': {}
    }

    total_bandwidth = 130  # eMBB(90) + URLLC(30) + mMTC(10)

    # 按RX_ID排序并取前n个用户
    df_sorted = df.sort_values('RX_ID').head(n)

    for slice_type in ['eMBB', 'URLLC', 'mMTC']:
        # 从前n个用户中筛选该切片类型
        slice_df = df_sorted[df_sorted['Request_Label'] == slice_type]

        if len(slice_df) == 0:
            continue

        Q = slice_df['CQI'].values
        M = len(Q)
        params = SLICE_PARAMS[slice_type]
        B = params['B']

        B_alloc, rates, total_rate = solve_resource_allocation(B, M, Q, params)

        used_bandwidth = np.sum(B_alloc)
        util = used_bandwidth / params['B'] * 100

        result['slices'][slice_type] = {
            'users': len(B_alloc),
            'allocated_bandwidth': used_bandwidth,
            'total_bandwidth': params['B'],
            'bandwidth_utilization': util,
            'rates': rates.tolist(),
            'total_rate': total_rate,
            'max_rate': np.max(rates),
        }
        result['total_throughput'] += total_rate
        result['max_rate'] = max(result['max_rate'], np.max(rates))

    # 计算总带宽利用率
    total_allocated = sum(
        result['slices'][s]['allocated_bandwidth']
        for s in result['slices']
    )
    result['bandwidth_utilization'] = total_allocated / total_bandwidth * 100

    return result


def main():
    # 读取数据
    data_path = 'ray_tracing_results/ray_tracing_results_south.csv'
    df = pd.read_csv(data_path)

    print("=" * 80)
    print("Rule-Based Throughput Calculation Results")
    print("=" * 80)
    print(f"Total users: {len(df)}")
    print(f"eMBB users: {len(df[df['Request_Label'] == 'eMBB'])}")
    print(f"URLLC users: {len(df[df['Request_Label'] == 'URLLC'])}")
    print(f"mMTC users: {len(df[df['Request_Label'] == 'mMTC'])}")
    print()

    # 计算不同用户数的结果
    user_counts = [5, 10, 15, 20, 25, 30]

    print("-" * 80)
    print(f"{'Users':^8} | {'Throughput(Mbps)':^18} | {'MaxRate(Mbps)':^15} | {'BW Util(%)':^12}")
    print("-" * 80)

    all_results = []
    for n in user_counts:
        if n > len(df):
            print(f"{n:^8} | {'Insufficient data':^18} | {'-':^15} | {'-':^12}")
            continue

        result = calculate_for_n_users(df, n)

        print(f"{result['n_users']:^8} | {result['total_throughput']:^18.2f} | "
              f"{result['max_rate']:^15.2f} | {result['bandwidth_utilization']:^12.2f}")

        all_results.append(result)

    print("-" * 80)
    print()

    # 详细结果
    print("=" * 80)
    print("Detailed Slice Results")
    print("=" * 80)

    for result in all_results:
        print(f"\n>>> Users: {result['n_users']}")
        print(f"Total Throughput: {result['total_throughput']:.2f} Mbps")
        print(f"Max Rate: {result['max_rate']:.2f} Mbps")
        print(f"Bandwidth Utilization: {result['bandwidth_utilization']:.2f}%")

        for slice_type, data in result['slices'].items():
            print(f"\n  [{slice_type}]")
            print(f"    Users: {data['users']}")
            print(f"    Allocated BW: {data['allocated_bandwidth']:.2f} MHz / {data['total_bandwidth']} MHz")
            print(f"    BW Utilization: {data['bandwidth_utilization']:.2f}%")
            print(f"    Total Rate: {data['total_rate']:.2f} Mbps")
            print(f"    Max Rate: {data['max_rate']:.2f} Mbps")

    # 返回结果供进一步使用
    return all_results


if __name__ == '__main__':
    results = main()