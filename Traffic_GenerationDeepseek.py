# -*- coding: utf-8 -*-
"""
优化版流量生成脚本
功能：
  1. 根据地面区域的流量权重和卫星实时位置，将地面需求分配给最近的卫星。
  2. 利用重力模型计算卫星之间的流量矩阵。
  3. 采用 NumPy 向量化运算，显著提升运行效率。
"""

import sys
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2

# --- 配置常量（与原脚本保持一致）---
OLD_WEIGHT_LIST = [
    [0,0,0,0,2,2,2,2,1,1,1,0,1,1,0,1,1,1,1,1,0,0,0,0],
    [5,17,19,2,2,2,2,2,1,1,1,1,4,9,5,5,5,5,5,5,5,5,5,5],
    [1,17,0,19,19,19,19,19,2,0,0,46,178,76,16,6,6,6,5,6,72,5,5,1],
    [1,0,0,17,32,17,17,17,0,0,1,51,52,75,71,27,41,70,67,134,151,38,0,0],
    [0,17,0,0,15,33,38,24,0,0,1,13,3,15,22,37,115,116,118,156,149,0,0,0],
    [0,1,0,1,0,1,26,24,15,0,0,13,99,5,27,1,1,94,40,39,34,1,1,1],
    [1,0,1,0,0,0,10,31,14,14,0,1,2,5,31,1,0,0,10,10,13,13,1,1],
    [1,0,1,0,0,0,0,18,21,14,0,1,1,15,14,1,0,0,0,2,2,2,2,1],
    [1,0,0,0,0,0,0,10,22,0,0,0,0,9,0,0,0,0,0,2,2,2,2,4],
    [0,0,0,0,0,0,0,10,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
]

TOTAL_OLD_WEIGHT_SUM = sum(sum(row) for row in OLD_WEIGHT_LIST)
WEIGHT_SCALE_FACTOR = 25000
WEIGHT_LIST = [[(element / TOTAL_OLD_WEIGHT_SUM) * WEIGHT_SCALE_FACTOR for element in row] for row in OLD_WEIGHT_LIST]
HOTSPOT_MULTIPLIER = 5.0

NUM_LAT_REGIONS = 12
NUM_LON_REGIONS = 24
LAT_RANGE = (90, -90)
LON_RANGE = (-180, 180)

SATELLITE_TRAJECTORY_CSV = '.\\经纬度(Iridium)new.csv'
TRAFFIC_MATRIX_CSV = 'traffic_matrix(Iridium).csv'

EARTH_RADIUS_KM = 6371

# 突发流量模拟参数（保留未使用）
BURST_PROBABILITY = 0.02
PARETO_SHAPE = 1.3
BURST_SCALE_MIN = 2.0
BURST_SCALE_MAX = 20.0

# --- 全局变量（预计算地面区域数组，避免重复转换）---
REGION_LATS = None
REGION_LONS = None
REGION_WEIGHTS = None

# --- 辅助函数（与原脚本一致）---
def create_regions(weight_matrix, num_lat, num_lon, lat_range, lon_range):
    """创建地理区域字典，包含中心经纬度和流量权重"""
    lat_span = (lat_range[0] - lat_range[1]) / num_lat
    lon_span = (lon_range[0] - lon_range[1]) / num_lon
    regions_dict = {}
    for lat_idx in range(num_lat):
        for lon_idx in range(num_lon):
            lat_center = lat_range[0] - lat_idx * lat_span - lat_span / 2
            lon_center = lon_range[0] + lon_idx * lon_span + lon_span / 2
            weight = weight_matrix[lat_idx][lon_idx]
            regions_dict[(lat_idx, lon_idx)] = {
                'lat_center': lat_center,
                'lon_center': lon_center,
                'weight': weight
            }
    return regions_dict

def haversine(lat1, lon1, lat2, lon2):
    """Haversine公式计算球面距离（公里）"""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    if a >= 1:
        a = 1.0 - sys.float_info.epsilon
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return c * EARTH_RADIUS_KM

# --- 向量化核心函数 ---
def assign_ground_demand_vectorized(sat_lats, sat_lons, reg_lats, reg_lons, reg_weights):
    """
    向量化分配地面需求到卫星
    参数：
        sat_lats, sat_lons: 卫星经纬度数组，形状 (S,)
        reg_lats, reg_lons, reg_weights: 地面区域中心经纬度和权重，形状 (R,)
    返回：
        sat_weights: 每个卫星累积的权重，形状 (S,)
    """
    # 转为弧度
    sat_lats_rad = np.radians(sat_lats)
    sat_lons_rad = np.radians(sat_lons)
    reg_lats_rad = np.radians(reg_lats)
    reg_lons_rad = np.radians(reg_lons)

    # 广播计算所有卫星到所有区域的差值
    dlat = sat_lats_rad[:, np.newaxis] - reg_lats_rad[np.newaxis, :]   # (S, R)
    dlon = sat_lons_rad[:, np.newaxis] - reg_lons_rad[np.newaxis, :]   # (S, R)

    # 向量化 Haversine
    a = np.sin(dlat / 2)**2 + np.cos(sat_lats_rad[:, np.newaxis]) * np.cos(reg_lats_rad[np.newaxis, :]) * np.sin(dlon / 2)**2
    a = np.clip(a, 0, 1.0)
    c = 2 * np.arcsin(np.sqrt(a))
    dist_matrix = c * EARTH_RADIUS_KM   # (S, R)

    # 对每个区域，找到最近卫星的索引
    nearest_sat_idx = np.argmin(dist_matrix, axis=0)   # (R,)
    # 累加权重
    sat_weights = np.bincount(nearest_sat_idx, weights=reg_weights, minlength=len(sat_lats))
    return sat_weights

def generate_traffic_matrix_vectorized(lats, lons, weights, time_step):
    """
    向量化生成卫星间流量矩阵
    参数：
        lats, lons: 卫星经纬度数组，形状 (S,)
        weights: 卫星流量权重数组，形状 (S,)
        time_step: 当前时间步（用于热点放大）
    返回：
        traffic: 流量矩阵，形状 (S, S)
    """
    S = len(weights)
    # 转为弧度
    lats_rad = np.radians(lats)
    lons_rad = np.radians(lons)

    # 计算距离矩阵 (S, S)
    dlat = lats_rad[:, np.newaxis] - lats_rad[np.newaxis, :]
    dlon = lons_rad[:, np.newaxis] - lons_rad[np.newaxis, :]
    a = np.sin(dlat / 2)**2 + np.cos(lats_rad[:, np.newaxis]) * np.cos(lats_rad[np.newaxis, :]) * np.sin(dlon / 2)**2
    a = np.clip(a, 0, 1.0)
    c = 2 * np.arcsin(np.sqrt(a))
    dist_matrix = c * EARTH_RADIUS_KM

    # 将对角线设为无穷大，避免自距离干扰
    np.fill_diagonal(dist_matrix, np.inf)

    # 标准化因子 norm_factor[i] = sum_{k!=i} weights[k] / dist_ik
    inv_dist = 1.0 / dist_matrix
    norm_factor = inv_dist @ weights   # (S,)

    # 流量矩阵 T[i,j] = (w_i * w_j) / (dist_ij * norm_factor_i)   (i != j)
    outer_weights = np.outer(weights, weights)
    denominator = dist_matrix * norm_factor[:, np.newaxis]
    traffic = outer_weights / denominator   # 分母为 inf 时结果为 0

    # 热点放大：当两个卫星的权重都超过阈值时，流量额外放大
    high_demand_mask = weights > (WEIGHT_SCALE_FACTOR * 0.05)
    both_high = np.outer(high_demand_mask, high_demand_mask) & ~np.eye(S, dtype=bool)
    if np.any(both_high):
        surge_factor = 1.0 + HOTSPOT_MULTIPLIER * np.sin(time_step / 10.0)**2
        traffic[both_high] *= surge_factor

    return traffic

# --- 修改后的接口函数（保持与原脚本同名，内部使用向量化）---
def assign_ground_demand_to_satellites(current_time_data, regions_dict):
    """
    将地面区域流量需求分配给最近的卫星（向量化实现）
    返回字典 {卫星ID: 累积权重}
    """
    global REGION_LATS, REGION_LONS, REGION_WEIGHTS
    # 提取当前时刻所有卫星的经纬度
    sat_lats = current_time_data['纬度'].values
    sat_lons = current_time_data['经度'].values
    # 调用向量化核心
    sat_weights_arr = assign_ground_demand_vectorized(sat_lats, sat_lons,
                                                      REGION_LATS, REGION_LONS, REGION_WEIGHTS)
    # 转换为字典（卫星ID为数组索引）
    return {i: sat_weights_arr[i] for i in range(len(sat_weights_arr))}

def generate_traffic_matrix_for_time(time_step, satellite_positions, satellite_weights_at_time, total_nodes):
    """
    为单个时间步生成卫星间流量矩阵（向量化实现）
    返回 DataFrame，格式与原脚本一致
    """
    # 从字典构建数组
    lats = np.array([satellite_positions[i][0] for i in range(total_nodes)])
    lons = np.array([satellite_positions[i][1] for i in range(total_nodes)])
    weights = np.array([satellite_weights_at_time.get(i, 0.0) for i in range(total_nodes)])

    traffic_arr = generate_traffic_matrix_vectorized(lats, lons, weights, time_step)
    return pd.DataFrame(traffic_arr)

# --- 主程序 ---
def main():
    global REGION_LATS, REGION_LONS, REGION_WEIGHTS

    # 1. 初始化地面区域并预计算数组
    regions = create_regions(WEIGHT_LIST, NUM_LAT_REGIONS, NUM_LON_REGIONS, LAT_RANGE, LON_RANGE)
    # 按插入顺序提取区域信息（与权重矩阵顺序一致）
    region_items = list(regions.values())
    REGION_LATS = np.array([r['lat_center'] for r in region_items])
    REGION_LONS = np.array([r['lon_center'] for r in region_items])
    REGION_WEIGHTS = np.array([r['weight'] for r in region_items])

    # 2. 读取卫星轨迹数据
    print(f"正在读取卫星轨迹文件: {SATELLITE_TRAJECTORY_CSV} ...")
    try:
        all_satellite_data = pd.read_csv(SATELLITE_TRAJECTORY_CSV, encoding='utf-8')
    except FileNotFoundError:
        print(f"错误: 未找到文件 '{SATELLITE_TRAJECTORY_CSV}'。")
        sys.exit(1)

    # 动态计算时间步数和卫星总数
    max_time_step = int(all_satellite_data['时间'].max())
    total_sim_steps = max_time_step + 1
    print(f"检测到最大时间步: {max_time_step} (总计 {total_sim_steps} 帧)")

    first_time_step_data = all_satellite_data[all_satellite_data['时间'] == 0]
    total_nodes = len(first_time_step_data)
    print(f"检测到卫星总数: {total_nodes}")

    # 3. 遍历每个时间步，收集流量矩阵
    all_traffic_dfs = []
    for time_step in range(total_sim_steps):
        if time_step % 10 == 0:
            print(f"处理进度: {time_step}/{max_time_step}...")

        current_time_data = all_satellite_data[all_satellite_data['时间'] == time_step]
        if current_time_data.empty:
            continue

        # 构建卫星位置字典（用于兼容原接口）
        satellite_positions = {}
        for idx, row in current_time_data.iterrows():
            satellite_positions[current_time_data.index.get_loc(idx)] = (row['纬度'], row['经度'])

        # 分配地面需求
        satellite_weights_at_time = assign_ground_demand_to_satellites(current_time_data, regions)

        # 生成流量矩阵
        traffic_df = generate_traffic_matrix_for_time(time_step, satellite_positions,
                                                      satellite_weights_at_time, total_nodes)
        all_traffic_dfs.append(traffic_df)

    # 4. 一次性写入 CSV
    if all_traffic_dfs:
        final_df = pd.concat(all_traffic_dfs, ignore_index=True)
        final_df.to_csv(TRAFFIC_MATRIX_CSV, index=False, header=False)
        print(f'\n所有时间步 ({total_sim_steps} 帧) 的流量矩阵已保存至 {TRAFFIC_MATRIX_CSV}')
    else:
        print('没有生成任何流量矩阵。')

if __name__ == '__main__':
    main()