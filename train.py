import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt

from dataset import LEOSatelliteDataset
from model import ST_GAGCN

# ==========================================
# 1. 宏观系统配置
# ==========================================
CONFIG = {
    'traffic_file': 'traffic_matrix(Iridium).csv',
    'location_file': '经纬度(Iridium).csv',
    'history_len': 10,   
    'pred_len': 5,      
    'batch_size': 16,    
    'epochs': 200,       
    'learning_rate': 0.001,
    'hidden_dim': 64,
    'heads': 4,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
}

def train_cloud_model(dataset, train_indices):
    print(f"--- 阶段一：云端模型离线预训练 (Device: {CONFIG['device']}) ---")
    train_dataset = Subset(dataset, train_indices)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    
    model = ST_GAGCN(
        num_nodes=66, in_features=3, hidden_dim=CONFIG['hidden_dim'], 
        heads=CONFIG['heads'], pred_len=CONFIG['pred_len']
    ).to(CONFIG['device'])
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    loss_history = [] 
    
    model.train()
    for epoch in range(CONFIG['epochs']):
        epoch_loss = 0
        for batch_X, batch_A, batch_Y in train_loader:
            batch_X, batch_A, batch_Y = batch_X.to(CONFIG['device']), batch_A.to(CONFIG['device']), batch_Y.to(CONFIG['device'])
            
            optimizer.zero_grad()
            outputs = model(batch_X, batch_A)
            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss/len(train_loader)
        loss_history.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{CONFIG['epochs']}], Loss: {avg_loss:.6f}")
            
    # 画收敛曲线
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, CONFIG['epochs']+1), loss_history, 'b-', linewidth=2)
    plt.title('Training Convergence Curve (MSE Loss)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('convergence_curve.png', dpi=300)
    print(">> 收敛曲线已保存为 convergence_curve.png")
            
    return model

if __name__ == "__main__":
    print("正在加载数据...")
    full_dataset = LEOSatelliteDataset(
        CONFIG['traffic_file'], CONFIG['location_file'], 
        history_len=CONFIG['history_len'], pred_len=CONFIG['pred_len']
    )
    
    train_size = int(0.8 * len(full_dataset))
    train_indices = list(range(0, train_size))
    
    # 训练模型
    trained_model = train_cloud_model(full_dataset, train_indices)
    
    # 保存模型权重
    torch.save(trained_model.state_dict(), 'saved_st_gagcn_model.pth')
    print(">> 模型权重已成功保存为 'saved_st_gagcn_model.pth'！")