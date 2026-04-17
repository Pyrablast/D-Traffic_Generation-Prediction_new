import pandas as pd
import numpy as np
import torch
import math
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import uniform_filter1d

class LEOSatelliteDataset(Dataset):     #大驼峰命名法，用于类的命名 #括号表示类的继承关系 #缩进替代大括号
    def __init__(self, traffic_file, location_file, history_len=10, pred_len=5):    #__init__构造函数，创建对象时进行初始化
        self.history_len = history_len      #将参数值保存为实例属性以便访问
        self.pred_len = pred_len
        self.num_nodes = 66
        
        # ===========================
        # 1. 读取并处理流量数据
        # ===========================
        # 读取 CSV，如果没有表头 header=None
        df_traffic = pd.read_csv(traffic_file, header=None)
        raw_traffic = df_traffic.values
        
        # 100ms 级别的总时间步
        high_freq_timesteps = raw_traffic.shape[0] // self.num_nodes
        traffic_tensor_high_freq = raw_traffic.reshape(high_freq_timesteps, self.num_nodes, self.num_nodes)
        
        # 每 10 个 100ms 为 1 秒
        self.total_timesteps = high_freq_timesteps // 10 
        
        # 出流量 (Out-flow) 
        out_flow_high = np.sum(traffic_tensor_high_freq, axis=2, keepdims=True) # [High_Time, Node, 1]
        # 入流量 (In-flow) 
        in_flow_high = np.sum(traffic_tensor_high_freq, axis=1, keepdims=True).transpose(0, 2, 1)
        
        # 执行 Max-pooling 
        out_flow = np.max(out_flow_high[:self.total_timesteps * 10].reshape(self.total_timesteps, 10, self.num_nodes, 1), axis=1)
        in_flow = np.max(in_flow_high[:self.total_timesteps * 10].reshape(self.total_timesteps, 10, self.num_nodes, 1), axis=1)

        # --- 归一化 ---
        self.flow_min = np.min(np.concatenate([out_flow, in_flow]))
        self.flow_max = np.max(np.concatenate([out_flow, in_flow]))
        out_flow_norm = (out_flow - self.flow_min) / (self.flow_max - self.flow_min + 1e-5)
        in_flow_norm = (in_flow - self.flow_min) / (self.flow_max - self.flow_min + 1e-5)
        
        # --- 邻接矩阵 A ---
        adj = np.zeros_like(traffic_tensor_high_freq)
        for t in range(self.total_timesteps):
            for i in range(self.num_nodes):
                # 只有当该节点对外发送了有效流量时，才去寻找 top-4 邻居
                if np.sum(traffic_tensor_high_freq[t, i, :]) > 1e-5:
                    top4_neighbors = np.argsort(traffic_tensor_high_freq[t, i, :])[-4:]
                    adj[t, i, top4_neighbors] = 1.0
                
                # 无论如何，保留自环 (让卫星时刻记住自己的状态)
                adj[t, i, i] = 1.0 
                
        self.adj_matrices = adj.astype(np.float32)
        
        # ===========================
        # 2. 读取并处理位置数据
        # ===========================
        df_loc = pd.read_csv(location_file)
        
        if '时间' in df_loc.columns and '当前节点' in df_loc.columns:
            df_loc = df_loc.sort_values(by=['时间', '当前节点'])
        
        # 此时 df_loc 长度是 100ms 级别的高频长度
        high_freq_time = len(df_loc) // self.num_nodes
        
        # Reshape 为 [时间帧, 卫星节点]
        lats_matrix = df_loc['纬度'].values.reshape(high_freq_time, self.num_nodes)
        lons_matrix = df_loc['经度'].values.reshape(high_freq_time, self.num_nodes)
        
        # 按照每 10 帧 (1秒) 抽取一次，与流量时间步严格对齐
        lats_matrix = lats_matrix[::10][:self.total_timesteps]
        lons_matrix = lons_matrix[::10][:self.total_timesteps]
        
        # 展平回一维数组计算 Grid ID
        lats = lats_matrix.flatten()
        lons = lons_matrix.flatten()
        
        # --- 计算 Grid ID ---
        grid_ids = []
        for lat, lon in zip(lats, lons):    
            grid_ids.append(self._calculate_grid_id(lat, lon))  
            
        # 重塑为 [Time, Node, 1]
        grid_ids = np.array(grid_ids).reshape(self.total_timesteps, self.num_nodes, 1)
        grid_ids_norm = grid_ids / 96.0
        
        # ===========================
        # 3. 特征融合
        # ===========================
        # 现在维度应该是:
        # out_flow_norm: (100, 66, 1)
        # in_flow_norm:  (100, 66, 1)  
        # grid_ids_norm: (100, 66, 1)
        self.node_features = np.concatenate([out_flow_norm, in_flow_norm, grid_ids_norm], axis=2) 
        
        # 用 5 秒的滑动平均滤掉纯随机尖刺，留下宏观趋势
        smoothed_out = uniform_filter1d(out_flow_norm, size=5, axis=0)
        smoothed_in = uniform_filter1d(in_flow_norm, size=5, axis=0)
        self.target_features = np.concatenate([smoothed_out, smoothed_in], axis=2)
        
        # 转为 Tensor
        self.node_features = torch.FloatTensor(self.node_features)
        self.target_features = torch.FloatTensor(self.target_features) # 新增平滑目标
        self.adj_matrices = torch.FloatTensor(self.adj_matrices)
        
        # 准备索引
        self.indices = []   #生成了一个从0到(总时间步-窗口总长)的连续整数列表用于索引
        for i in range(self.total_timesteps - self.history_len - self.pred_len + 1):
            self.indices.append(i)

    def _calculate_grid_id(self, lat, lon): #网格分区
        row = math.floor((90 - lat) / 22.5)
        row = min(max(row, 0), 7)
        col = math.floor((180 + lon) / 30)
        col = min(max(col, 0), 11)
        return row * 12 + col

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        # 输入 X 依然是包含尖刺的原始真实特征 (让模型看到恶劣的现状)
        X = self.node_features[t : t + self.history_len, :, :]  
        A = self.adj_matrices[t + self.history_len - 1]         
        
        # 预测目标 Y 变成了“平滑后的趋势” (只取前两维 out/in flow)
        Y = self.target_features[t + self.history_len : t + self.history_len + self.pred_len, :, :2]  
        return X, A, Y

# 测试部分
if __name__ == "__main__":  #Python 的习惯用法，当这个脚本被直接运行时，才执行下面这些测试代码
    try:                    #异常监测，尝试执行try中代码，出现错误后标记其特定类别比如Exception，
        # 请确保文件名正确
        dataset = LEOSatelliteDataset("traffic_matrix(Iridium).csv", "经纬度(Iridium)new.csv")
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        data = next(iter(dataloader))
        print("Success! Feature shape:", data[0].shape) # [Batch, 12, 66, 3]
    except Exception as e:  #若出现Exception异常，跳转执行这里的代码，将异常对象赋值给e
        import traceback
        traceback.print_exc()