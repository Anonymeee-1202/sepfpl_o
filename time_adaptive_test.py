"""
时间适应隐私预算分配测试脚本

该脚本独立生成时间适应隐私预算分配序列，包括：
- eps_t: 每轮的隐私预算
- sigma_t: 每轮的 RDP 噪声系数
- std_t: 每轮的最终噪声标准差

不依赖其他模块，可以独立运行。
"""

import math
from typing import List, Tuple


def generate_time_adaptive_sequences(
    total_rounds: int,
    rdp_eps_tot: float,
    rdp_alpha: float = 2.0,
    rdp_p: float = 1.05,
    sensitivity: float = 1.0
) -> Tuple[List[float], List[float], List[float]]:
    """
    生成时间适应隐私预算分配序列
    
    参数:
        total_rounds (int): 总训练轮数
        rdp_eps_tot (float): 总隐私预算
        rdp_alpha (float): RDP 阶数 α，控制隐私保证的严格程度（默认 2.0）
        rdp_p (float): 时间适应幂次参数，控制隐私预算的分配策略（默认 1.05）
                      p > 1 表示后期轮次分配更多隐私预算，有利于模型收敛
        sensitivity (float): 敏感度，表示单个样本对梯度的影响上限（默认 1.0）
    
    返回:
        eps_t_list (List[float]): 每轮的隐私预算序列
        sigma_t_list (List[float]): 每轮的 RDP 噪声系数序列
        std_t_list (List[float]): 每轮的最终噪声标准差序列
    
    公式说明:
        1. 时间适应分配公式: ε_t = ε_tot * (t^p) / (sum_{j=1}^T j^p)
           其中 t 是当前轮次，T 是总轮数，p 是幂次参数
        
        2. RDP 高斯机制公式: σ = sqrt(α / (2 * ε_α))
           其中 α 是 RDP 阶数，ε_α 是隐私预算
        
        3. 最终噪声标准差: std_t = σ_t * sensitivity
           其中 sensitivity 是敏感度
    """
    # 计算归一化分母: sum_{j=1}^T j^p
    denominator = sum(j ** rdp_p for j in range(1, total_rounds + 1))
    
    # 初始化结果列表
    eps_t_list = []
    sigma_t_list = []
    std_t_list = []
    
    # 预计算每轮的隐私预算和对应的噪声标准差
    for t in range(1, total_rounds + 1):
        # 计算第 t 轮的隐私预算
        # 时间适应分配公式: ε_t = ε_tot * (t^p) / (sum_{j=1}^T j^p)
        eps_t = rdp_eps_tot * (t ** rdp_p) / denominator
        eps_t_list.append(eps_t)
        
        # 根据该轮的隐私预算计算对应的噪声标准差
        # 使用相同的 RDP 公式: σ = sqrt(α / (2 * ε_α))
        sigma_t = math.sqrt(rdp_alpha / (2.0 * eps_t))
        sigma_t_list.append(sigma_t)
        
        # 最终的噪声标准差 = RDP 噪声系数 × 敏感度
        std_t = sigma_t * sensitivity
        std_t_list.append(std_t)
    
    return eps_t_list, sigma_t_list, std_t_list


def print_sequences(
    eps_t_list: List[float],
    sigma_t_list: List[float],
    std_t_list: List[float],
    rdp_eps_tot: float,
    show_all: bool = False,
    max_display: int = 10
):
    """
    打印生成的序列
    
    参数:
        eps_t_list (List[float]): 每轮的隐私预算序列
        sigma_t_list (List[float]): 每轮的 RDP 噪声系数序列
        std_t_list (List[float]): 每轮的最终噪声标准差序列
        rdp_eps_tot (float): 总隐私预算（用于验证）
        show_all (bool): 是否显示所有轮次（默认 False，只显示前 N 个和后 N 个）
        max_display (int): 当 show_all=False 时，显示前 N 个和后 N 个轮次（默认 10）
    """
    total_rounds = len(eps_t_list)
    
    print(f"\n{'='*80}")
    print(f"时间适应隐私预算分配序列 (总轮数: {total_rounds})")
    print(f"{'='*80}")
    print(f"{'Round':<8} {'eps_t':<15} {'sigma_t':<15} {'std_t':<15}")
    print(f"{'-'*80}")
    
    if show_all or total_rounds <= max_display * 2:
        # 显示所有轮次
        for t in range(total_rounds):
            print(f"{t+1:<8} {eps_t_list[t]:<15.6f} {sigma_t_list[t]:<15.6f} {std_t_list[t]:<15.6f}")
    else:
        # 显示前 N 个和后 N 个轮次
        for t in range(max_display):
            print(f"{t+1:<8} {eps_t_list[t]:<15.6f} {sigma_t_list[t]:<15.6f} {std_t_list[t]:<15.6f}")
        
        print(f"{'...':<8} {'...':<15} {'...':<15} {'...':<15}")
        
        for t in range(total_rounds - max_display, total_rounds):
            print(f"{t+1:<8} {eps_t_list[t]:<15.6f} {sigma_t_list[t]:<15.6f} {std_t_list[t]:<15.6f}")
    
    print(f"{'='*80}")
    
    # 打印统计信息
    print(f"\n统计信息:")
    print(f"  eps_t 范围: [{min(eps_t_list):.6f}, {max(eps_t_list):.6f}]")
    print(f"  sigma_t 范围: [{min(sigma_t_list):.6f}, {max(sigma_t_list):.6f}]")
    print(f"  std_t 范围: [{min(std_t_list):.6f}, {max(std_t_list):.6f}]")
    print(f"  eps_t 总和: {sum(eps_t_list):.6f} (期望: {rdp_eps_tot:.6f})")
    print(f"  eps_t 总和误差: {abs(sum(eps_t_list) - rdp_eps_tot):.6f}")


def main():
    """
    主函数：演示如何使用时间适应隐私预算分配
    """
    # 默认参数配置
    total_rounds = 40
    rdp_eps_tot = 0.4
    rdp_alpha = 2.0
    rdp_p = 0.1
    sensitivity = 1.0
    
    print("时间适应隐私预算分配测试")
    print(f"\n参数配置:")
    print(f"  总轮数 (total_rounds): {total_rounds}")
    print(f"  总隐私预算 (rdp_eps_tot): {rdp_eps_tot}")
    print(f"  RDP 阶数 (rdp_alpha): {rdp_alpha}")
    print(f"  时间适应幂次 (rdp_p): {rdp_p}")
    print(f"  敏感度 (sensitivity): {sensitivity}")
    
    # 生成序列
    eps_t_list, sigma_t_list, std_t_list = generate_time_adaptive_sequences(
        total_rounds=total_rounds,
        rdp_eps_tot=rdp_eps_tot,
        rdp_alpha=rdp_alpha,
        rdp_p=rdp_p,
        sensitivity=sensitivity
    )
    
    # 打印结果
    print_sequences(eps_t_list, sigma_t_list, std_t_list, rdp_eps_tot, show_all=False, max_display=5)
    
    # 示例：不同参数配置的对比
    print(f"\n\n{'='*80}")
    print("不同 rdp_p 参数的影响对比")
    print(f"{'='*80}")
    
    p_values = [0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
    for p in p_values:
        eps_list, _, std_list = generate_time_adaptive_sequences(
            total_rounds=total_rounds,
            rdp_eps_tot=rdp_eps_tot,
            rdp_alpha=rdp_alpha,
            rdp_p=p,
            sensitivity=sensitivity
        )
        print(f"\nrdp_p = {p}:")
        print(f"  第1轮 eps_t: {eps_list[0]:.6f}, std_t: {std_list[0]:.6f}")
        print(f"  第{total_rounds}轮 eps_t: {eps_list[-1]:.6f}, std_t: {std_list[-1]:.6f}")
        print(f"  增长比例: {eps_list[-1]/eps_list[0]:.2f}x")


if __name__ == "__main__":
    main()

