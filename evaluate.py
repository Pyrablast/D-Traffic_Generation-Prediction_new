import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from dataset import LEOSatelliteDataset
from model import ST_GAGCN

# ==========================================
# 1. 仿真系统配置 
# ==========================================
CONFIG = {
    'traffic_file': 'traffic_matrix(Iridium).csv',
    'location_file': '经纬度(Iridium)new.csv',
    'history_len': 12,   
    'pred_len': 60,      
    'hidden_dim': 64,
    'heads': 4,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    
    'max_macro_cycle': 60, 
    'trigger_margin': 1.1,          
    'absolute_tolerance_mbps': 15.0 
}

# ==========================================
# 2. 边缘节点实体类 (支持双边触发)
# ==========================================
class SatelliteEdgeNode:
    def __init__(self, node_id, margin, abs_tol_mbps, flow_min, flow_max):
        self.node_id = node_id
        self.margin = margin
        self.abs_tol_mbps = abs_tol_mbps
        self.flow_min = flow_min
        self.flow_max = flow_max
        self.threshold_envelope = None  
        self.current_idx = 0            
        
    def update_envelope(self, new_envelope):
        self.threshold_envelope = new_envelope
        self.current_idx = 0
        
    def _denormalize(self, value):
        return value * (self.flow_max - self.flow_min) + self.flow_min

    def check_trigger(self, actual_traffic_1s):
        """返回触发类型: 'upper', 'lower', 'none', 'timeout'"""
        if self.threshold_envelope is None or self.current_idx >= len(self.threshold_envelope):
            return 'timeout'
            
        pred_out = max(self.threshold_envelope[self.current_idx][0], 0.0)
        pred_in  = max(self.threshold_envelope[self.current_idx][1], 0.0)
        
        real_pred_out = self._denormalize(pred_out)
        real_pred_in  = self._denormalize(pred_in)
        real_actual_out = self._denormalize(actual_traffic_1s[0])
        real_actual_in  = self._denormalize(actual_traffic_1s[1])
        
        # 上限与下限计算
        upper_safe_out = real_pred_out * self.margin + self.abs_tol_mbps
        upper_safe_in  = real_pred_in * self.margin + self.abs_tol_mbps
        
        lower_margin = 2.0 - self.margin
        lower_safe_out = max(0.0, real_pred_out * lower_margin - self.abs_tol_mbps)
        lower_safe_in  = max(0.0, real_pred_in * lower_margin - self.abs_tol_mbps)
        
        # 双边判定
        if real_actual_out > upper_safe_out or real_actual_in > upper_safe_in:
            return 'upper'
        elif real_actual_out < lower_safe_out or real_actual_in < lower_safe_in:
            return 'lower'
        else:
            return 'none'
            
    def step_and_get_prediction(self):
        if self.threshold_envelope is None or self.current_idx >= len(self.threshold_envelope):
            return [0.0, 0.0]
        pred_out = max(self.threshold_envelope[self.current_idx][0], 0.0)
        pred_in  = max(self.threshold_envelope[self.current_idx][1], 0.0)
        self.current_idx += 1 
        return [pred_out, pred_in]

# ==========================================
# 3. 核心仿真引擎 (收集全网数据)
# ==========================================
def run_simulation(model, dataset, test_indices):
    print("\n--- 启动云边协同双边触发仿真 (加载预训练权重) ---")
    model.eval()
    edge_nodes = [SatelliteEdgeNode(i, CONFIG['trigger_margin'], CONFIG['absolute_tolerance_mbps'], dataset.flow_min, dataset.flow_max) for i in range(66)]
    
    timer_since_last_calc = 0
    total_inferences = 0        
    trigger_timestamps = [] 
    
    history_real_traffic = []   
    history_upper_envelope = []  
    history_lower_envelope = [] 
    history_pure_preds = []     
    
    with torch.no_grad():
        for step, idx in enumerate(test_indices):
            X_t, A_t, Y_t = dataset[idx] 
            actual_traffic_now = Y_t[0].numpy() 
            
            real_out_flow = actual_traffic_now[:, 0] * (dataset.flow_max - dataset.flow_min) + dataset.flow_min
            history_real_traffic.append(real_out_flow)
            
            # 1. 收集本地报警状态
            alarms_upper = 0
            alarms_lower = 0
            is_timeout = False
            
            for node_idx, edge in enumerate(edge_nodes):
                status = edge.check_trigger(actual_traffic_now[node_idx])
                if status == 'upper': alarms_upper += 1
                elif status == 'lower': alarms_lower += 1
                elif status == 'timeout': is_timeout = True
                
            total_alarms = alarms_upper + alarms_lower
            
            # 2. 全局共识触发机制 (共识阈值设为 5，或发生超时)
            if total_alarms >= 5 or is_timeout or timer_since_last_calc >= CONFIG['max_macro_cycle']:
                
                # 判断全局触发原因
                if is_timeout or timer_since_last_calc >= CONFIG['max_macro_cycle']:
                    reason = 'timeout'
                elif alarms_upper >= alarms_lower:
                    reason = 'upper'  
                else:
                    reason = 'lower'  
                    
                trigger_timestamps.append((step, reason))
                
                input_X = X_t.unsqueeze(0).to(CONFIG['device'])
                input_A = A_t.unsqueeze(0).to(CONFIG['device'])
                predicted_envelopes = model(input_X, input_A).squeeze(0).cpu().numpy() 
                for i, edge in enumerate(edge_nodes):
                    edge.update_envelope(predicted_envelopes[:, i, :])
                timer_since_last_calc = 0
                total_inferences += 1
            else:
                timer_since_last_calc += 1
                
            # 3. 记录全网当前使用的上下包络线
            step_env_up = []
            step_env_low = []
            for i, edge in enumerate(edge_nodes):
                if edge.threshold_envelope is not None and edge.current_idx < len(edge.threshold_envelope):
                    current_pred_norm = max(edge.threshold_envelope[edge.current_idx][0], 0.0)
                    real_pred = current_pred_norm * (dataset.flow_max - dataset.flow_min) + dataset.flow_min
                    
                    upper_safe = real_pred * CONFIG['trigger_margin'] + CONFIG['absolute_tolerance_mbps']
                    lower_margin = 2.0 - CONFIG['trigger_margin']
                    lower_safe = max(0.0, real_pred * lower_margin - CONFIG['absolute_tolerance_mbps'])
                else:
                    upper_safe = history_upper_envelope[-1][i] if len(history_upper_envelope)>0 else 0.0
                    lower_safe = history_lower_envelope[-1][i] if len(history_lower_envelope)>0 else 0.0
                    
                step_env_up.append(upper_safe)
                step_env_low.append(lower_safe)
                
            history_upper_envelope.append(step_env_up)
            history_lower_envelope.append(step_env_low)
            
            current_preds = np.array([edge.step_and_get_prediction() for edge in edge_nodes])
            history_pure_preds.append(current_preds)
            
    trad_inf = len(test_indices)
    saved = (1 - total_inferences / trad_inf) * 100
    print(f"仿真完成! 传统推理: {trad_inf}次 | 双边事件触发: {total_inferences}次 | 算力节约: {saved:.2f}%")
    
    return (np.array(history_real_traffic), 
            np.array(history_upper_envelope), 
            np.array(history_lower_envelope),
            np.array(history_pure_preds), 
            trigger_timestamps)

# ==========================================
# 4. 独立画图模块
# ==========================================
def plot_satellite_dashboard(sat_idx, real_traffic, env_upper, env_lower, pure_preds, triggers, flow_min, flow_max):
    os.makedirs('results_images', exist_ok=True)
    time_steps = len(real_traffic)
    
    sat_real = real_traffic[:, sat_idx]
    sat_up = env_upper[:, sat_idx]
    sat_low = env_lower[:, sat_idx]
    sat_pred_real = pure_preds[:, sat_idx, 0] * (flow_max - flow_min) + flow_min
    
    # ------ 图 1: 管状安全包络线与双色事件触发 ------
    plt.figure(figsize=(15, 6))
    plt.plot(range(time_steps), sat_real, label='Actual Traffic Demand', color='dodgerblue', linewidth=1.5, zorder=3)
    
    # 画上下限 
    plt.step(range(time_steps), sat_up, where='post', label='Upper Boundary (Congestion Limit)', color='crimson', linewidth=2, linestyle='--', zorder=4)
    plt.step(range(time_steps), sat_low, where='post', label='Lower Boundary (Waste Limit)', color='seagreen', linewidth=2, linestyle='-.', zorder=4)
    
    # 填充"安全管道" 
    plt.fill_between(range(time_steps), sat_low, sat_up, color='lightgray', alpha=0.3, step='post', label='Safe Tube (No Trigger Zone)')
    
    plt.plot([], [], color='red', linestyle=':', linewidth=1.5, label='Upper Trigger (Expand Bandwidth)')
    plt.plot([], [], color='green', linestyle=':', linewidth=1.5, label='Lower Trigger (Reclaim Bandwidth)')
    plt.plot([], [], color='orange', linestyle='-.', linewidth=1.5, label='Timeout Trigger (Cycle Reset)')
    
    # 画三色触发线 (不再传 label 参数)
    for t, reason in triggers:
        if reason == 'upper':
            color, ls = 'red', ':'
        elif reason == 'lower':
            color, ls = 'green', ':'
        else:
            color, ls = 'orange', '-.'
        plt.axvline(x=t, color=color, linestyle=ls, linewidth=1.5, zorder=1)
        
    plt.title(f'Tube-based Event-Triggered Control on Satellite #{sat_idx}', fontsize=16, fontweight='bold')
    plt.xlabel('Simulation Time (Seconds)', fontsize=14)
    plt.ylabel('Traffic Volume (Mbps)', fontsize=14)
    plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'results_images/Sat_{sat_idx:02d}_Tube_Envelope.png', dpi=300)
    plt.close()
    
    # ------ 图 2: 真实拥塞 vs 盲预测追踪 ------
    plt.figure(figsize=(14, 6))
    plt.plot(range(time_steps), sat_real, label='Ground Truth (Real Congestion)', marker='.', color='tab:blue', linewidth=1.5, markersize=8)
    plt.plot(range(time_steps), sat_pred_real, label='ST-GAGCN Blind Prediction (60-Step)', marker='x', color='tab:orange', linewidth=1.5, markersize=6)
    
    plt.title(f'Real-time Prediction Tracking for Satellite #{sat_idx}', fontsize=16, fontweight='bold')
    plt.xlabel('Simulation Time Steps (Seconds)', fontsize=14)
    plt.ylabel('Traffic Volume (Mbps)', fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='-', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'results_images/Sat_{sat_idx:02d}_Tracking.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    dataset = LEOSatelliteDataset(CONFIG['traffic_file'], CONFIG['location_file'], history_len=CONFIG['history_len'], pred_len=CONFIG['pred_len'])
    train_size = int(0.8 * len(dataset))
    test_indices = list(range(train_size, len(dataset)))
    
    model = ST_GAGCN(num_nodes=66, in_features=3, hidden_dim=CONFIG['hidden_dim'], heads=CONFIG['heads'], pred_len=CONFIG['pred_len']).to(CONFIG['device'])
    
    try:
        model.load_state_dict(torch.load('saved_st_gagcn_model.pth'))
        print(">> 成功加载预训练模型权重 'saved_st_gagcn_model.pth'")
    except FileNotFoundError:
        print("错误：找不到模型权重文件！请先运行 train.py 进行训练。")
        exit()
        
    real_traffic, env_upper, env_lower, pure_preds, triggers = run_simulation(model, dataset, test_indices)
    
    TARGET_SATELLITES = [0, 42, 43, 44]  
    
    print(f"\n正在为卫星 {TARGET_SATELLITES} 生成全新管状图表...")
    for sat_idx in TARGET_SATELLITES:
        plot_satellite_dashboard(sat_idx, real_traffic, env_upper, env_lower, pure_preds, triggers, dataset.flow_min, dataset.flow_max)
        print(f"  -> Satellite #{sat_idx} 的图表已生成完毕！")