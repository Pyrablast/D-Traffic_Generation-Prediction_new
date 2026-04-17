import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def simulate_edge_node(csv_file, sat_idx=0, num_nodes=66):
    print("正在加载 100ms 级高频流量矩阵...")
    df = pd.read_csv(csv_file, header=None)
    raw_data = df.values
    
    total_steps = raw_data.shape[0] // num_nodes
    raw_data = raw_data[:total_steps * num_nodes, :]
    
    # Reshape 为 [Time(100ms), Source, Dest]
    traffic_tensor = raw_data.reshape(total_steps, num_nodes, num_nodes)
    
    # 获取 0 号卫星在 100ms 级别的出流量总和
    high_freq_traffic = np.sum(traffic_tensor[:, sat_idx, :], axis=1) # 长度为 6000
    
    # ==========================================
    # 模拟边缘节点 (Satellite Edge) 的本地动作
    # ==========================================
    # 每 1 秒 (10 个 100ms 采样点) 作为一个宏观周期
    macro_steps = total_steps // 10
    
    # 1. 你的方案: Max-pooling (捕获微突发)
    max_pooled_traffic = np.max(high_freq_traffic[:macro_steps*10].reshape(macro_steps, 10), axis=1)
    
    # 2. 传统方案: Average-pooling (平滑掉了突发)
    avg_pooled_traffic = np.mean(high_freq_traffic[:macro_steps*10].reshape(macro_steps, 10), axis=1)
    
    # ==========================================
    # 绘图展示对比
    # ==========================================
    # 为了看得清楚，我们只画前 100 秒 (1000 个 100ms 采样点)
    plot_sec = min(100, macro_steps)
    
    plt.figure(figsize=(16, 6))
    
    # 画底层 100ms 极高频原始数据 (灰色背景细线)
    x_100ms = np.arange(plot_sec * 10) / 10.0  # 换算成秒
    plt.plot(x_100ms, high_freq_traffic[:plot_sec*10], color='lightgray', alpha=0.8, linewidth=1, label='Raw 100ms Micro-traffic')
    
    # 画传统 Average-pooling 的结果 (绿色线)
    x_1s = np.arange(plot_sec)
    plt.step(x_1s, avg_pooled_traffic[:plot_sec], where='post', color='green', alpha=0.7, linewidth=2, label='Traditional Avg-pooling (1s)')
    
    # 画你的 Max-pooling 方案的结果 (红色线)
    plt.step(x_1s, max_pooled_traffic[:plot_sec], where='post', color='red', linewidth=2, label='Your Max-pooling (1s)')
    
    plt.title(f"Edge Node Preprocessing: Max-pooling vs Avg-pooling (Satellite #{sat_idx})", fontsize=14)
    plt.xlabel("Time (Seconds)", fontsize=12)
    plt.ylabel("Traffic Volume", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12, loc='upper left')
    plt.tight_layout()
    plt.savefig('edge_node_pooling_comparison.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    simulate_edge_node("traffic_matrix(Iridium).csv", sat_idx=0)