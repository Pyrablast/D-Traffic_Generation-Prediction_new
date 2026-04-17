import pandas as pd
import numpy as np
import torch
import math
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import uniform_filter1d

class LEOSatelliteDataset(Dataset):
    def __init__(self, traffic_file, location_file, history_len=10, pred_len=5):
        self.history_len = history_len
        self.pred_len = pred_len
        self.num_nodes = 66
        
        # ===========================
        # 1. 读取并处理流量数据 (适配原版，1步=1帧)
        # ===========================
        # 原版流量矩阵直接生成 [Time * Nodes, Nodes] 的结构
        df_traffic = pd.read_csv(traffic_file, header=None)
        raw_traffic = df_traffic.values
        
        # 直接计算总时间步
        self.total_timesteps = raw_traffic.shape[0] // self.num_nodes
        traffic_tensor = raw_traffic.reshape(self.total_timesteps, self.num_nodes, self.num_nodes)
        
        # 计算出流量和入流量
        out_flow = np.sum(traffic_tensor, axis=2, keepdims=True) # [Time, Node, 1]
        in_flow = np.sum(traffic_tensor, axis=1, keepdims=True).transpose(0, 2, 1) # [Time, Node, 1]
        
        # --- 归一化 ---
        self.flow_min = np.min(np.concatenate([out_flow, in_flow]))
        self.flow_max = np.max(np.concatenate([out_flow, in_flow]))
        # 防止除零错误
        denominator = self.flow_max - self.flow_min if self.flow_max > self.flow_min else 1.0
        out_flow_norm = (out_flow - self.flow_min) / denominator
        in_flow_norm = (in_flow - self.flow_min) / denominator
        
        # --- 构建邻接矩阵 A ---
        adj = np.zeros_like(traffic_tensor)
        for t in range(self.total_timesteps):
            for i in range(self.num_nodes):
                # 依然采用业务流量Top-4驱动的动态邻接矩阵 (符合你模型的创新点)
                if np.sum(traffic_tensor[t, i, :]) > 1e-5:
                    top4_neighbors = np.argsort(traffic_tensor[t, i, :])[-4:]
                    adj[t, i, top4_neighbors] = 1.0
                # 无论如何，保留自环
                adj[t, i, i] = 1.0 
                
        self.adj_matrices = adj.astype(np.float32)
        
        # ===========================
        # 2. 读取并处理位置数据 (适配原版经纬度.csv)
        # ===========================
        df_loc = pd.read_csv(location_file)
        
        # 【重要】提取卫星节点ID映射，为后续NS-3联调做准备
        # 原版数据含有 "当前节点" 列（如 Iridium_1）
        unique_nodes = df_loc['当前节点'].unique()
        self.node_name_to_id = {name: idx for idx, name in enumerate(unique_nodes)}
        
        # 确保按时间和节点顺序排列，保证和流量矩阵严格对应
        # 注意：这里假设时间列已经是数值/整数类型
        df_loc = df_loc.sort_values(by=['时间', '当前节点'])
        
        # 直接 Reshape 为 [时间帧, 卫星节点]
        # 如果 df_loc 的行数少于 total_timesteps * num_nodes，截断流量矩阵对齐
        valid_timesteps = min(self.total_timesteps, len(df_loc) // self.num_nodes)
        self.total_timesteps = valid_timesteps
        
        lats_matrix = df_loc['纬度'].values[:self.total_timesteps * self.num_nodes].reshape(self.total_timesteps, self.num_nodes)
        lons_matrix = df_loc['经度'].values[:self.total_timesteps * self.num_nodes].reshape(self.total_timesteps, self.num_nodes)
        
        # --- 计算 Grid ID ---
        grid_ids = []
        for lat, lon in zip(lats_matrix.flatten(), lons_matrix.flatten()):    
            grid_ids.append(self._calculate_grid_id(lat, lon))  
            
        grid_ids = np.array(grid_ids).reshape(self.total_timesteps, self.num_nodes, 1)
        grid_ids_norm = grid_ids / 96.0
        
        # ===========================
        # 3. 特征融合
        # ===========================
        # 截断流量特征以与 valid_timesteps 对齐
        out_flow_norm = out_flow_norm[:self.total_timesteps]
        in_flow_norm = in_flow_norm[:self.total_timesteps]
        self.adj_matrices = self.adj_matrices[:self.total_timesteps]
        
        self.node_features = np.concatenate([out_flow_norm, in_flow_norm, grid_ids_norm], axis=2) 
        
        # 因为现在时间步通常较长(如1分钟)，平滑窗口(size)建议缩小为3
        smoothed_out = uniform_filter1d(out_flow_norm, size=3, axis=0)
        smoothed_in = uniform_filter1d(in_flow_norm, size=3, axis=0)
        self.target_features = np.concatenate([smoothed_out, smoothed_in], axis=2)
        
        self.node_features = torch.FloatTensor(self.node_features)
        self.target_features = torch.FloatTensor(self.target_features) 
        self.adj_matrices = torch.FloatTensor(self.adj_matrices)
        
        # 生成滑动窗口索引
        self.indices = []   
        for i in range(self.total_timesteps - self.history_len - self.pred_len + 1):
            self.indices.append(i)
            
        print(f"Dataset 初始化完成！总有效时间步: {self.total_timesteps}, 可用样本数: {len(self.indices)}")
        print(f"请牢记节点映射关系供NS-3使用: 0号节点为 {unique_nodes[0]}")

    def _calculate_grid_id(self, lat, lon): 
        row = math.floor((90 - lat) / 22.5)
        row = min(max(row, 0), 7)
        col = math.floor((180 + lon) / 30)
        col = min(max(col, 0), 11)
        return row * 12 + col

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        X = self.node_features[t : t + self.history_len, :, :]  
        A = self.adj_matrices[t + self.history_len - 1]         
        Y = self.target_features[t + self.history_len : t + self.history_len + self.pred_len, :, :2]  
        return X, A, Y

if __name__ == "__main__":
    try:
        # 注意替换为你原版STK生成的文件名
        dataset = LEOSatelliteDataset("traffic_matrix(Iridium).csv", "经纬度(Iridium).csv")
        if len(dataset) > 0:
            dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
            data = next(iter(dataloader))
            print("Success! Feature shape:", data[0].shape) 
        else:
            print("警告：可用样本数为 0！请增加 STK 的仿真总时长或减少 history_len / pred_len。")
    except Exception as e:
        import traceback
        traceback.print_exc()