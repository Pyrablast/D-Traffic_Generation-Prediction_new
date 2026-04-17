# --- START OF FILE visualize_burst.py (Synthetic Concept Version) ---
import numpy as np
import matplotlib.pyplot as plt

def generate_synthetic_micro_traffic(seconds=100, samples_per_sec=10):
    """生成包含微突发的高频虚拟流量 (用于论证边缘节点预处理)"""
    total_samples = seconds * samples_per_sec
    time_x = np.linspace(0, seconds, total_samples)
    
    # 基础平滑流量 (模拟卫星过境时的宏观流量波峰)
    base_traffic = 50 + 30 * np.sin(2 * np.pi * time_x / 100) 
    
    # 注入微突发 (Micro-bursts): 使用重尾分布模拟网络拥塞尖刺
    burst_noise = np.random.pareto(a=2.5, size=total_samples) * 8.0
    
    high_freq_traffic = base_traffic + burst_noise
    return time_x, high_freq_traffic

def plot_edge_node_concept():
    print("正在生成虚拟的高频微突发数据，论证边缘节点截获能力...")
    seconds_to_plot = 100
    samples_per_sec = 10 # 100ms 采样率
    
    # 生成高频数据
    x_100ms, high_freq_traffic = generate_synthetic_micro_traffic(seconds_to_plot, samples_per_sec)
    
    # 执行 Pooling
    macro_steps = seconds_to_plot
    reshaped_traffic = high_freq_traffic.reshape(macro_steps, samples_per_sec)
    
    # 1. 我们的方案: Max-pooling (捕获微突发)
    max_pooled_traffic = np.max(reshaped_traffic, axis=1)
    # 2. 传统方案: Average-pooling (平滑掉了突发)
    avg_pooled_traffic = np.mean(reshaped_traffic, axis=1)
    
    # 绘图
    plt.figure(figsize=(14, 6))
    
    # 画底层 100ms 极高频原始数据
    plt.plot(x_100ms, high_freq_traffic, color='lightgray', alpha=0.8, linewidth=1, label='Raw 100ms Micro-burst Traffic')
    
    # 画传统 Average-pooling
    x_1s = np.arange(macro_steps)
    plt.step(x_1s, avg_pooled_traffic, where='post', color='seagreen', alpha=0.9, linewidth=2, linestyle='--', label='Traditional Avg-pooling (Misses Bursts)')
    
    # 画你的 Max-pooling
    plt.step(x_1s, max_pooled_traffic, where='post', color='crimson', alpha=0.9, linewidth=2, label='Our Edge Max-pooling (Captures Bursts)')
    
    plt.title("Edge Node Preprocessing: Capturing Micro-bursts via Max-pooling", fontsize=16, fontweight='bold')
    plt.xlabel("Local Processing Time", fontsize=14)
    plt.ylabel("Traffic Volume (Mbps)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(fontsize=12, loc='upper left')
    plt.tight_layout()
    plt.savefig('edge_node_pooling_concept.png', dpi=300)
    print(">> 理论论证图已生成：edge_node_pooling_concept.png (可以直接放进论文！)")

if __name__ == "__main__":
    plot_edge_node_concept()