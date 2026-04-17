# -*- coding: utf-8 -*-
"""
这个脚本的主要功能是根据地面区域的流量需求和卫星的实时位置，选取最近卫星建立星地链路
利用重力模型计算卫星之间的流量矩阵。
"""
import random
import sys
import pandas as pd
from math import radians, sin, cos, sqrt, atan2

# --- 配置常量 ---
# 地面区域流量权重矩阵，表示不同地理区域的流量需求强度。
# 这是一个12x24的矩阵，对应12个纬度区域和24个经度区域。
OLD_WEIGHT_LIST = [[0, 0, 0, 0, 2, 2, 2, 2, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                   [5, 17, 19, 2, 2, 2, 2, 2, 1, 1, 1, 1, 4, 9, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                   [1, 17, 0, 19, 19, 19, 19, 19, 2, 0, 0, 46, 178, 76, 16, 6, 6, 6, 5, 6, 72, 5, 5, 1],
                   [1, 0, 0, 17, 32, 17, 17, 17, 0, 0, 1, 51, 52, 75, 71, 27, 41, 70, 67, 134, 151, 38, 0, 0],
                   [0, 17, 0, 0, 15, 33, 38, 24, 0, 0, 1, 13, 3, 15, 22, 37, 115, 116, 118, 156, 149, 0, 0, 0],
                   [0, 1, 0, 1, 0, 1, 26, 24, 15, 0, 0, 13, 99, 5, 27, 1, 1, 94, 40, 39, 34, 1, 1, 1],
                   [1, 0, 1, 0, 0, 0, 10, 31, 14, 14, 0, 1, 2, 5, 31, 1, 0, 0, 10, 10, 13, 13, 1, 1],
                   [1, 0, 1, 0, 0, 0, 0, 18, 21, 14, 0, 1, 1, 15, 14, 1, 0, 0, 0, 2, 2, 2, 2, 1],
                   [1, 0, 0, 0, 0, 0, 0, 10, 22, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 2, 2, 2, 2, 4],
                   [0, 0, 0, 0, 0, 0, 0, 10, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

TOTAL_OLD_WEIGHT_SUM = sum(sum(row) for row in OLD_WEIGHT_LIST)
# 流量权重缩放因子，根据星座特性调整，目前采用new方式设计
# Old: 13.2 = 66颗卫星 * 2 (双向) * 0.1 (假设的平均流量), 0.44 是一个调整系数，使得表格均值符合预期
# New : 66 * 2 * 0.1 * 20 ( 假设的平均流量，链路数<约等于卫星数*2> * 负载强度<0.1> * 链路带宽<20> Mbps)
WEIGHT_SCALE_FACTOR = 264 # New: 66 * 2 * 0.1 * 20=264      Old: 13.2 * 0.44
WEIGHT_LIST = [[(element / TOTAL_OLD_WEIGHT_SUM) * WEIGHT_SCALE_FACTOR for element in row] for row in OLD_WEIGHT_LIST]

# 地理区域划分
NUM_LAT_REGIONS = 12    # 纬度方向的区域数量。
NUM_LON_REGIONS = 24    # 经度方向的区域数量。
LAT_RANGE = (90, -90)   # 纬度范围 (北到南)
LON_RANGE = (-180, 180) # 经度范围 (西到东)

# 仿真时间
TOTAL_SIM_TIME_STEPS = 1441 # 仿真时间步数 (例如，101表示从时间步1到100)

# 输入/输出文件路径
SATELLITE_TRAJECTORY_CSV = '.\\经纬度(Iridium).csv' # 包含卫星轨迹数据的CSV文件路径。
TRAFFIC_MATRIX_CSV = 'traffic_matrix(Iridium).csv'  # 输出流量矩阵的CSV文件路径。

# 地球半径 (公里)
EARTH_RADIUS_KM = 6371  # 用于Haversine公式计算距离。

# --- 辅助函数 ---

def create_regions(weight_matrix, num_lat, num_lon, lat_range, lon_range):
    """
    创建地理区域字典，包含中心经纬度和流量权重。

    Args:
        weight_matrix (list[list[float]]): 流量权重矩阵。
        num_lat (int): 纬度区域数量。
        num_lon (int): 经度区域数量。
        lat_range (tuple[int, int]): 纬度范围 (北到南)。
        lon_range (tuple[int, int]): 经度范围 (西到东)。

    Returns:
        dict: 包含每个地理区域信息的字典，键为 (lat_idx, lon_idx)，值为包含
              'lat_center', 'lon_center', 'weight' 的字典。
    """
    # 计算每个纬度区域和经度区域的跨度。
    lat_span = (lat_range[0] - lat_range[1]) / num_lat
    lon_span = (lon_range[0] - lon_range[1]) / num_lon
    
    regions_dict = {}
    # 遍历所有纬度区域和经度区域。
    for lat_idx in range(num_lat):
        for lon_idx in range(num_lon):
            # 计算当前区域的中心纬度。
            lat_center = lat_range[0] - lat_idx * lat_span - lat_span / 2
            # 计算当前区域的中心经度
            lon_center = lon_range[0] + lon_idx * lon_span + lon_span / 2 # 经度从-180开始递增
            
            # 获取当前区域的流量权重。
            weight = weight_matrix[lat_idx][lon_idx]
            # 将区域信息存储到字典中。
            regions_dict[(lat_idx, lon_idx)] = {
                'lat_center': lat_center,
                'lon_center': lon_center,
                'weight': weight
            }
    return regions_dict

def haversine(lat1, lon1, lat2, lon2):
    """
    使用Haversine公式计算两个地理坐标点之间的球面距离（公里）。

    Args:
        lat1 (float): 第一个点的纬度。
        lon1 (float): 第一个点的经度。
        lat2 (float): 第二个点的纬度。
        lon2 (float): 第二个点的经度。

    Returns:
        float: 两个点之间的球面距离（公里）。
    """
    # 将经纬度从度转换为弧度。
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # 计算经度和纬度差。
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    # Haversine公式的核心计算。
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    
    # 避免浮点误差导致a略大于1，将其限制在1以内。
    if a >= 1:
        a = 1.0 - sys.float_info.epsilon
    # 计算中心角。
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    # 返回球面距离。
    return c * EARTH_RADIUS_KM

def gravity_model(demand_i, demand_j, distance_ij, normalization_factor):
    """
    根据重力模型计算两个节点之间的流量需求。

    Args:
        demand_i (float): 节点i的流量需求（权重）。
        demand_j (float): 节点j的流量需求（权重）。
        distance_ij (float): 节点i和节点j之间的距离。
        normalization_factor (float): 标准化因子，用于调整流量大小。

    Returns:
        float: 节点i和节点j之间的流量需求。
    """
    # 避免除以零或非常小的距离
    if distance_ij == 0: # 同一节点，流量为0
        return 0.0
    if normalization_factor == 0: # 避免除以零
        return 0.0
    
    demand_ij = (demand_i * demand_j) / (distance_ij * normalization_factor)
    return demand_ij

def assign_ground_demand_to_satellites(current_time_data, regions_dict):
    """
    将地面区域的流量需求分配给最近的卫星。
    返回每个卫星的流量权重字典。

    Args:
        current_time_data (pd.DataFrame): 当前时间步的卫星位置数据，包含 '纬度' 和 '经度' 列。
        regions_dict (dict): 包含地理区域信息的字典。

    Returns:
        dict: 键为卫星ID（DataFrame内部索引），值为该卫星累积的流量权重。
    """
    satellite_weights_at_time = {}
    
    # 遍历每个地面区域
    for region_key, region_value in regions_dict.items():
        # 计算当前时间点所有卫星到该区域中心的距离
        distances = current_time_data.apply(lambda row: haversine(
            row['纬度'],
            row['经度'],
            region_value['lat_center'],
            region_value['lon_center']
        ), axis=1)
        
        # 找出最近的卫星
        nearest_satellite_idx_in_df = distances.idxmin() # 获取DataFrame中的索引
        
        # 获取最近卫星的原始数据行，并提取其在当前time_data中的“节点编号”
        # 注意：这里的“节点编号”是time_data的内部索引，不是Cons_Construction.py中的卫星名称
        # 如果需要使用卫星名称，需要修改Cons_Construction.py输出或此处逻辑
        satellite_id_for_weight = current_time_data.index.get_loc(nearest_satellite_idx_in_df)
        
        # 累加区域权重到最近的卫星
        satellite_weights_at_time[satellite_id_for_weight] = \
            satellite_weights_at_time.get(satellite_id_for_weight, 0) + region_value['weight']
            
    return satellite_weights_at_time

def generate_traffic_matrix_for_time(time_step, satellite_positions, satellite_weights_at_time, total_nodes):
    """
    为单个时间步生成卫星间的流量矩阵。

    Args:
        time_step (int): 当前的时间步。
        satellite_positions (dict): 键为卫星ID，值为 (纬度, 经度) 的字典。
        satellite_weights_at_time (dict): 键为卫星ID，值为其流量权重的字典。
        total_nodes (int): 卫星的总数量。

    Returns:
        pd.DataFrame: 表示卫星间流量的DataFrame（流量矩阵）。
    """
    # 初始化一个全零的流量矩阵。
    traffic_matrix = [[0.0 for _ in range(total_nodes)] for _ in range(total_nodes)]
    
    # 预计算所有卫星对之间的距离
    distance_matrix = {}
    for i in range(total_nodes):
        for j in range(i + 1, total_nodes): # 只计算上三角，避免重复
            dist = haversine(satellite_positions[i][0], satellite_positions[i][1],
                             satellite_positions[j][0], satellite_positions[j][1])
            distance_matrix[(i, j)] = dist
            distance_matrix[(j, i)] = dist # 对称存储
            
    # 使用重力模型填充流量矩阵
    for i in range(total_nodes):
        demand_i = satellite_weights_at_time.get(i, 0)
        
        # 计算标准化因子 (total_d)
        # 这是重力模型中的一个关键部分，用于衡量卫星i与其他所有卫星的相对“吸引力”。
        normalization_factor = 0.0
        for k in range(total_nodes):
            if k != i:
                demand_k = satellite_weights_at_time.get(k, 0)
                dist_ik = distance_matrix.get((i, k), 0) # 从预计算的距离矩阵中获取
                if dist_ik > 0: # 避免除以零
                    normalization_factor += (demand_k / dist_ik)

        # 遍历所有其他卫星j，计算卫星i到卫星j的流量。
        for j in range(total_nodes):
            if i != j:
                demand_j = satellite_weights_at_time.get(j, 0)
                dist_ij = distance_matrix.get((i, j), 0) # 从预计算的距离矩阵中获取
                # 调用重力模型计算流量，并填充到流量矩阵中。
                traffic_matrix[i][j] = gravity_model(demand_i, demand_j, dist_ij, normalization_factor)
                
    return pd.DataFrame(traffic_matrix)

# --- 主程序逻辑 ---
def main():
    # 1. 初始化地面区域
    # 根据预设的权重列表、区域数量和范围创建地理区域字典。
    regions = create_regions(WEIGHT_LIST, NUM_LAT_REGIONS, NUM_LON_REGIONS, LAT_RANGE, LON_RANGE)

    # 2. 导入卫星轨迹数据
    try:
        # 尝试从CSV文件读取卫星轨迹数据。
        all_satellite_data = pd.read_csv(SATELLITE_TRAJECTORY_CSV, encoding='utf-8')
    except FileNotFoundError:
        # 如果文件未找到，打印错误信息并退出程序。
        print(f"错误: 未找到卫星轨迹文件 '{SATELLITE_TRAJECTORY_CSV}'。请确保文件存在且路径正确。")
        sys.exit(1)
    
    # 动态确定卫星总数
    # 假设在第一个时间步，所有卫星都已出现
    first_time_step_data = all_satellite_data[all_satellite_data['时间'] == all_satellite_data['时间'].min()]
    total_nodes = len(first_time_step_data)
    print(f"检测到卫星总数: {total_nodes}")

    # 确保CSV文件在开始写入前是空的，或者只包含一次header
    # 如果是追加模式，第一次运行前需要手动删除文件或清空
    # 这里为了简化，假设文件不存在或可以覆盖
    # 如果需要保留历史数据，请调整为更复杂的逻辑
    try:
        # 以写入模式打开CSV文件，清空其内容（如果存在），不写入header和索引。
        pd.DataFrame().to_csv(TRAFFIC_MATRIX_CSV, mode='w', header=False, index=False)
    except Exception as e:
        # 如果清空或创建文件失败，打印警告信息。
        print(f"警告: 无法清空或创建流量矩阵文件 '{TRAFFIC_MATRIX_CSV}'。可能文件被占用或权限不足。错误: {e}")

    # 3. 遍历每个时间步，计算卫星流量权重并生成流量矩阵
    for time_step in range(1, TOTAL_SIM_TIME_STEPS):
        print(f"\n处理时间步: {time_step}...")
        # 筛选出当前时间步的卫星数据
        current_time_data = all_satellite_data[all_satellite_data['时间'] == time_step]
        
        if not current_time_data.empty:
            # 获取当前时间步的卫星位置 (以DataFrame索引作为ID)
            satellite_positions = {}
            for idx, row in current_time_data.iterrows():
                # 使用DataFrame的内部索引作为卫星的临时ID
                satellite_positions[current_time_data.index.get_loc(idx)] = (row['纬度'], row['经度'])

            # 将地面需求分配给卫星
            satellite_weights_at_time = assign_ground_demand_to_satellites(current_time_data, regions)
            
            # 生成流量矩阵
            traffic_df = generate_traffic_matrix_for_time(time_step, satellite_positions, satellite_weights_at_time, total_nodes)
            
            # 追加到CSV文件
            traffic_df.to_csv(TRAFFIC_MATRIX_CSV, mode='a', header=False, index=False)
            print(f"  - 流量矩阵已为时间步 {time_step} 保存。")
        else:
            print(f"  - 时间步 {time_step} 没有卫星数据，跳过。")

    print(f'\n所有时间步的流量矩阵已保存至 {TRAFFIC_MATRIX_CSV}')

if __name__ == '__main__':
    main()