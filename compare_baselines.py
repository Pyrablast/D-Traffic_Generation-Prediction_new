import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Subset

from dataset import LEOSatelliteDataset
from train import train_cloud_model, CONFIG, SatelliteEdgeNode, calculate_metrics

def run_simulation_mode(model, dataset, test_indices, mode="Traditional"):
    """统一的仿真引擎，支持两种模式切换"""
    model.eval()
    
    # 实例化边缘节点
    edge_nodes = [SatelliteEdgeNode(i, CONFIG['trigger_margin'], CONFIG['absolute_tolerance_mbps'], 
                                    dataset.flow_min, dataset.flow_max) for i in range(66)]
    
    timer_since_last_calc = 0
    total_inferences = 0        
    all_truths, all_preds = [], []
    
    with torch.no_grad():
        for step, idx in enumerate(test_indices):
            X_t, A_t, Y_t = dataset[idx] 
            actual_traffic_now = Y_t[0].numpy()
            
            # 本地检查
            alarms = sum([1 for node_idx, edge in enumerate(edge_nodes) if edge.check_trigger(actual_traffic_now[node_idx])])
            
            # --- 核心模式区分 ---
            if mode == "Traditional":
                # 传统时间驱动：每 1 秒强制重算 1 次！无视触发！
                trigger_condition = True
            else:
                # 事件触发：共识阈值 >= 3 颗卫星 或 达到 60秒生命周期
                trigger_condition = (alarms >= 3) or (timer_since_last_calc >= CONFIG['max_macro_cycle'])
            
            if trigger_condition:
                input_X = X_t.unsqueeze(0).to(CONFIG['device'])
                input_A = A_t.unsqueeze(0).to(CONFIG['device'])
                predicted_envelopes = model(input_X, input_A).squeeze(0).cpu().numpy() 
                
                for i, edge in enumerate(edge_nodes):
                    edge.update_envelope(predicted_envelopes[:, i, :])
                
                timer_since_last_calc = 0
                total_inferences += 1
            else:
                timer_since_last_calc += 1
                
            current_preds = np.array([edge.step_and_get_prediction() for edge in edge_nodes])
            all_preds.append(current_preds)
            all_truths.append(actual_traffic_now)
            
    # 计算指标
    all_truths, all_preds = np.array(all_truths), np.array(all_preds)
    mae_norm, mse_norm, mae_real, rmse_real, wape = calculate_metrics(all_truths, all_preds, dataset.flow_min, dataset.flow_max)
    
    return total_inferences, wape

def plot_baseline_comparison(trad_inf, trad_wape, event_inf, event_wape):
    """画双轴对比柱状图"""
    labels = ['Time-Driven (Baseline)', 'Event-Triggered (Ours)']
    inferences = [trad_inf, event_inf]
    wapes = [trad_wape, event_wape]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 柱状图参数
    x = np.arange(len(labels))
    width = 0.35
    
    # 左轴：算力消耗 (推断次数)
    rects1 = ax1.bar(x - width/2, inferences, width, label='Compute Cost (Inferences)', color='royalblue')
    ax1.set_ylabel('Total Inferences (Compute Cost)', fontsize=14, fontweight='bold', color='royalblue')
    ax1.tick_params(axis='y', labelcolor='royalblue')
    ax1.set_ylim(0, max(inferences) * 1.2)
    
    # 右轴：预测误差 (WAPE)
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, wapes, width, label='Prediction Error (WAPE %)', color='crimson')
    ax2.set_ylabel('Weighted Avg Percentage Error (WAPE %)', fontsize=14, fontweight='bold', color='crimson')
    ax2.tick_params(axis='y', labelcolor='crimson')
    ax2.set_ylim(0, max(wapes) * 1.5)
    
    # 在柱子上打上数字标签
    for rect in rects1:
        height = rect.get_height()
        ax1.annotate(f'{int(height)}', xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=12, fontweight='bold')
    for rect in rects2:
        height = rect.get_height()
        ax2.annotate(f'{height:.2f}%', xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=14, fontweight='bold')
    plt.title('Performance Comparison: Compute Cost vs. Prediction Error', fontsize=16, fontweight='bold')
    
    # 合并图例
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', fontsize=12)
    
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('baseline_comparison.png', dpi=300)
    print("\n>> 对比图已生成：baseline_comparison.png")

if __name__ == "__main__":
    # 1. 加载数据与训练模型
    print("Loading Data and Pre-training Model...")
    dataset = LEOSatelliteDataset(CONFIG['traffic_file'], CONFIG['location_file'], history_len=CONFIG['history_len'], pred_len=CONFIG['pred_len'])
    train_size = int(0.8 * len(dataset))
    test_indices = list(range(train_size, len(dataset)))
    
    model = train_cloud_model(dataset, list(range(0, train_size)))
    
    # 2. 跑 Baseline (传统每秒推断)
    print("\n[Running Mode 1]: Time-Driven (Traditional Baseline)")
    trad_inf, trad_wape = run_simulation_mode(model, dataset, test_indices, mode="Traditional")
    
    # 3. 跑 我们的方法 (事件触发)
    print("\n[Running Mode 2]: Event-Triggered (Proposed Method)")
    event_inf, event_wape = run_simulation_mode(model, dataset, test_indices, mode="Event-Triggered")
    
    # 4. 画图
    print(f"\n--- 总结 ---")
    print(f"传统算力消耗: {trad_inf} 次 | 误差: {trad_wape:.2f}%")
    print(f"事件触发算力: {event_inf} 次 | 误差: {event_wape:.2f}%")
    print(f"算力节约: {((trad_inf - event_inf)/trad_inf)*100:.2f}% !!!")
    plot_baseline_comparison(trad_inf, trad_wape, event_inf, event_wape)