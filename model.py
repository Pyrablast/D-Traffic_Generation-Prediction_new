import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GatedTCN(nn.Module):  #继承nn.Module
    """ 
    时间特征提取: Gated TCN
    """
    #in_channels（输入通道数，输入序列每个时间步的特征维度）; out_channels（输出通道数，经过 GatedTCN 后每个时间步的新特征维度）（卷积核数）
    #kernel_size（卷积核大小）; dilation（膨胀率，各个卷积核之间的间隔步长）
    #一维卷积的一维含义为，卷积进行的方向是一维的，比如xy平面中沿着一条直线进行卷积
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):   #Conv1d一维卷积，卷积意义为保留空间关系的前提下，提取局部特征
        super(GatedTCN, self).__init__()    #调用父类nn.Module的构造函数
        self.padding = (kernel_size - 1) * dilation
        self.conv_1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                                dilation=dilation, padding=self.padding)
        self.conv_2 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                                dilation=dilation, padding=self.padding)

    def forward(self, x):       #前向传播函数，从输入层到输出层的方向
        # x: [Batch*Nodes, Channels, Time]
        # 裁剪掉 Padding，保持因果性
        p = self.conv_1(x)[:, :, :-self.padding]    #对输入 x 应用第一个一维卷积层，输出形状为[Batch*Nodes, out_channels, Time + padding]。[]为切片操作
        q = self.conv_2(x)[:, :, :-self.padding]    #对输入 x 应用第二个卷积层，虽然两层设置相同，但是输出不一样
        return torch.tanh(p) * torch.sigmoid(q)     #tanh输出候选内容（结果），sigmoid输出门控权重（结果可信度）

class NativeDenseGATLayer(nn.Module):   #继承nn.Module
    """
    原生 PyTorch 实现的 GAT 层 (无需 torch_geometric)
    适用于稠密邻接矩阵 (Batch, N, N)
    """
    def __init__(self, in_features, out_features, alpha=0.2):
        super(NativeDenseGATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        # 线性变换 W
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # 注意力参数 a
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        # h: [Batch, N, in_features]
        # adj: [Batch, N, N]
        
        Batch, N, _ = h.size()
        
        # 1. 线性变换: Wh
        # h @ W -> [Batch, N, out_features]
        Wh = torch.matmul(h, self.W) 
        
        # 2. 准备注意力机制的输入
        # 我们需要拼接 Wh_i 和 Wh_j
        # repeat logic to create [Batch, N, N, 2*out]
        a_input = self._prepare_attentional_mechanism_input(Wh)
        
        # 3. 计算注意力分数
        # [Batch, N, N, 2*out] @ [2*out, 1] -> [Batch, N, N, 1]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        
        # 4. Masking (只关注有连接的邻居)
        # 如果 adj 是 0，我们要把它设为负无穷，这样 Softmax 后权重为 0
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        
        # 5. Softmax 归一化
        attention = F.softmax(attention, dim=2) # [Batch, N, N]
        
        # 6. 加权求和
        # [Batch, N, N] @ [Batch, N, out] -> [Batch, N, out]
        h_prime = torch.matmul(attention, Wh)
        
        return F.elu(h_prime)

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh: [Batch, N, Out]
        Batch, N, Out = Wh.size()
        
        # Wh_repeated_in_chunks: [Batch, N, N, Out] (source nodes)
        # Wh_repeated_alternating: [Batch, N, N, Out] (target nodes)
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1).view(Batch, N, N, Out)
        Wh_repeated_alternating = Wh.repeat(1, N, 1).view(Batch, N, N, Out)
        
        # cat -> [Batch, N, N, 2*Out]
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=3)
        return all_combinations_matrix

class ST_GAGCN(nn.Module):
    def __init__(self, num_nodes, in_features, hidden_dim, heads=4, pred_len=60):
        super(ST_GAGCN, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.pred_len = pred_len
        
        # --- 替换为原生 GAT ---
        # 多头注意力通过创建多个 Layer 实现
        self.gat_heads = nn.ModuleList([
            NativeDenseGATLayer(in_features, hidden_dim) for _ in range(heads)
        ])
        # 线性层将多头输出融合 (Heads * Hidden -> Hidden)
        self.spatial_linear = nn.Linear(heads * hidden_dim, hidden_dim)
        # 时间层
        self.temporal_layer = GatedTCN(in_features, hidden_dim, kernel_size=3, dilation=2)
        # 融合层
        self.fusion_mha = nn.MultiheadAttention(embed_dim=2*hidden_dim, num_heads=heads, batch_first=True)
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(2 * hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.pred_len * 2),
        )
    def forward(self, x, adj):
        """
        x: [Batch, Time, Nodes, Features]
        adj: [Batch, Nodes, Nodes]
        """
        Batch, Time, Nodes, Feats = x.shape
        
        # ====================
        # Path A: Spatial
        # ====================
        x_last = x[:, -1, :, :] # [Batch, Nodes, Feats]
        
        # 多头并行计算
        spatial_outputs = [gat(x_last, adj) for gat in self.gat_heads] # List of [Batch, Nodes, Hidden]
        spatial_cat = torch.cat(spatial_outputs, dim=2) # [Batch, Nodes, Heads*Hidden]
        spatial_out = self.spatial_linear(spatial_cat) # [Batch, Nodes, Hidden]
        
        # ====================
        # Path B: Temporal
        # ====================
        x_permuted = x.permute(0, 2, 3, 1) # [Batch, Nodes, Feats, Time]
        x_reshaped = x_permuted.reshape(Batch * Nodes, Feats, Time)
        
        temporal_out = self.temporal_layer(x_reshaped) 
        temporal_out = temporal_out[:, :, -1] # 取最后一个时间步
        temporal_out = temporal_out.reshape(Batch, Nodes, self.hidden_dim)
        
        # ====================
        # Path C: Fusion
        # ====================
        fusion_input = torch.cat([spatial_out, temporal_out], dim=2)
        attn_out, _ = self.fusion_mha(fusion_input, fusion_input, fusion_input)
        fusion_out = fusion_input + attn_out
        
        # ====================
        # Output
        # ====================
        prediction = self.output_layer(fusion_out) # 形状: [Batch, Nodes, pred_len * 2]
        
        # 将展平的输出重新 Reshape 成 [Batch, pred_len, Nodes, Features]
        prediction = prediction.view(Batch, Nodes, self.pred_len, 2)
        # 调整维度顺序为 [Batch, Time, Nodes, Features]
        prediction = prediction.permute(0, 2, 1, 3) 
        
        return prediction # [Batch, pred_len, Nodes, 2]

# 测试部分
if __name__ == "__main__":
    dummy_x = torch.randn(2, 12, 66, 3) # Batch=2, 过去12步, 66个节点, 3个特征
    dummy_adj = torch.ones(2, 66, 66)   # 全连接图
    
    # 实例化模型，并指定预测未来 60 步
    model = ST_GAGCN(num_nodes=66, in_features=3, hidden_dim=64, heads=4, pred_len=60)
    out = model(dummy_x, dummy_adj)
    
    # 我们期望看到: [2, 60, 66, 2]
    print("支持多步预测的 ST-GAGCN 模型输出尺寸:", out.shape)