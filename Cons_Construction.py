# -*- coding: utf-8 -*-
"""
This script automates STK to generate satellite constellation trajectory data.

本脚本通过与STK (Systems Tool Kit)的COM接口进行交互，实现了自动化生成
多种卫星星座（如Iridium, OneWeb, Kuiper, Starlink）的轨道数据。
它会创建一个仿真场景，构建指定的星座，然后提取每颗卫星在设定时间内的
经纬度轨迹(每隔 1h 统计一次经纬度)，并最终将这些数据导出为CSV文件。
"""
import pandas as pd
from comtypes.client import CreateObject
from comtypes.gen import STKObjects

# --- Configuration Constants ---
SCENARIO_NAME = 'sce_Iridium_Constellation' # 在STK中创建的场景名称
CONSTELLATION_NAME = 'Iridium'              # 要生成的星座名称，也用作卫星模板名  Iridium, OneWeb, Kuiper, Starlink  
OUTPUT_CSV_FILE = '经纬度(Iridium).csv'     # 输出的CSV文件名

# Scenario Time
SCENARIO_START_TIME = '12 Oct 2025 00:00:00.00' # 仿真开始时间
SCENARIO_STOP_TIME = '14 Oct 2025 00:00:00.00'   # 仿真结束时间

# Constellation Parameters
NUM_ORBITAL_PLANES = 6  # 轨道平面数量。  # Iridium:6, OneWeb:18, Kuiper:34, Starlink:72
SATS_PER_PLANE = 11     # 每个轨道平面上的卫星数量。 # Iridium:11, OneWeb:40, Kuiper:34, Starlink:22
PHASE_SHIFT = 0         # 相邻轨道平面的相位差。     # Iridium:0, OneWeb:0, Kuiper:1, Starlink:1
CONS_TYPE = 'Star'      # 星座类型。                # Iridium:Star, OneWeb:Star, Kuiper:Delta, Starlink:Delta

# Orbit Parameters
ORBIT_EPOCH = 3         # 轨道坐标系，3代表J2000惯性坐标系
SEMI_MAJOR_AXIS_KM = 7151  # 轨道半长轴（公里），等于地球半径(约6371km) + 轨道高度。 # 高度参考: Iridium:780, OneWeb:1200, Kuiper:630, Starlink:550
INCLINATION_DEG = 86.4  # 轨道倾角（度）。          # Iridium:86.4, OneWeb:87.9, Kuiper:51.9, Starlink:53

# Data Export Parameters
TIME_STEP_SEC = 60    # 经纬度数据采样的时间步长（秒）。例如，3600代表每小时采样一次。


def setup_stk_scenario(app):
    """Creates and configures a new STK scenario."""
    # 获取STK对象模型的根入口
    root = app.Personality2
    # 检查当前是否有已打开的场景，如果有，则先关闭，以避免冲突
    if root.CurrentScenario is not None:
        root.CloseScenario()
    

    # 创建一个指定名称的新场景
    root.NewScenario(SCENARIO_NAME)
    # 获取对当前场景对象的引用
    scenario_object = root.CurrentScenario
    
    # 为了设置场景时间等高级属性，需要查询到更具体的IAgScenario接口
    scenario_interface = scenario_object.QueryInterface(STKObjects.IAgScenario)
    scenario_interface.StartTime = SCENARIO_START_TIME
    scenario_interface.StopTime = SCENARIO_STOP_TIME
    
    # 将仿真时间重置到开始时刻
    root.Rewind()
    # 返回场景的根对象，后续创建卫星等子对象需要用到它
    return scenario_object


def create_satellite_constellation(scenario, root):
    """Creates the satellite constellation."""
    # 使用场景对象(IAgStkObject)的Children属性来创建一个新的卫星对象作为模板
    constellation_template = scenario.Children.New(STKObjects.eSatellite, CONSTELLATION_NAME)
    
    # 查询卫星对象的IAgSatellite接口，以访问其轨道传播器等属性
    satellite_interface = constellation_template.QueryInterface(STKObjects.IAgSatellite)
    # 获取传播器对象
    propagator = satellite_interface.Propagator
    # 将传播器转换为二体模型(TwoBody)的专用接口
    two_body_propagator = propagator.QueryInterface(STKObjects.IAgVePropagatorTwoBody)
    
    # 使用经典的轨道六根数来定义卫星的初始轨道状态
    two_body_propagator.InitialState.Representation.AssignClassical(
        ORBIT_EPOCH, SEMI_MAJOR_AXIS_KM, 0, INCLINATION_DEG, 0, 0, 0
    )
    # 命令STK根据初始状态计算并传播轨道
    two_body_propagator.Propagate()

    # 构建并执行STK的Walker工具命令，以基于模板卫星自动生成整个星座
    command = (
        f'Walker */Satellite/{CONSTELLATION_NAME} Type {CONS_TYPE} '
        f'NumPlanes {NUM_ORBITAL_PLANES} NumSatsPerPlane {SATS_PER_PLANE} '
        f'InterPlanePhaseIncrement {PHASE_SHIFT} ColorByPlane Yes'
    )
    root.ExecuteCommand(command)
    # 模板卫星的任务已完成，将其从场景中卸载以保持整洁
    constellation_template.Unload()


def export_satellite_trajectories(scenario, root):
    """Exports the latitude and longitude of each satellite to a CSV file."""
    # 获取当前场景中所有类型为“卫星”的对象
    satellites = root.CurrentScenario.Children.GetElements(STKObjects.eSatellite)
    # 设置STK的默认时间单位为“纪元秒”，方便后续进行统一的时间计算
    root.UnitPreferences.SetCurrentUnit("DateFormat", "EpSec")

    # 初始化一个列表，用于存储所有卫星的轨迹数据点
    trajectory_data = []
    # 定义需要从STK提取的数据项
    elements_to_extract = ['Time', 'Lat', 'Lon']

    print(f"Extracting data for {satellites.Count} satellites...")
    
    # 在此函数内部查询场景的IAgScenario接口，以获取正确的仿真时间范围
    scenario_interface = scenario.QueryInterface(STKObjects.IAgScenario)

    # 遍历每一颗卫星
    for i in range(satellites.Count):
        satellite = satellites.Item(i)
        sat_name = satellite.InstanceName
        
        # 获取卫星的LLA（经纬高）数据提供者，这需要一系列精确的接口查询
        data_provider = satellite.DataProviders.Item('LLA State')
        data_provider_group = data_provider.QueryInterface(STKObjects.IAgDataProviderGroup)
        fixed_group_item = data_provider_group.Group.Item('Fixed')
        lla_provider_timevar = fixed_group_item.QueryInterface(STKObjects.IAgDataPrvTimeVar)

        # 执行数据提取命令，指定时间范围、步长和需要的数据元素
        results = lla_provider_timevar.ExecElements(
            scenario_interface.StartTime, scenario_interface.StopTime, TIME_STEP_SEC, elements_to_extract
        )

        # 从结果中分别获取时间、纬度和经度的数据集
        time_values = results.DataSets.GetDataSetByName('Time').GetValues()
        lat_values = results.DataSets.GetDataSetByName('Lat').GetValues()
        lon_values = results.DataSets.GetDataSetByName('Lon').GetValues()

        # 将每个时间点的数据整理后添加到总列表中
        for time, lat, lon in zip(time_values, lat_values, lon_values):
            # time / TIME_STEP_SEC 将纪元秒转换为以步长为单位的时间索引 (0, 1, 2, ...)
            trajectory_data.append([sat_name, lat, lon, time / TIME_STEP_SEC])
        
        print(f"  - Extracted data for {sat_name}")

    # 使用pandas将收集到的数据转换为DataFrame
    df = pd.DataFrame(trajectory_data, columns=['当前节点', '纬度', '经度', '时间'])
    # 将DataFrame保存为CSV文件，不包含行索引，并使用'utf-8-sig'编码以正确处理中文
    df.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8-sig')
    print(f"\nTrajectory data successfully saved to {OUTPUT_CSV_FILE}")


def main():
    """Main function to run the STK automation."""
    app = None  # 初始化app变量，确保在finally块中可用
    try:
        # 连接到STK应用程序
        app = CreateObject("STK11.Application")
        # 使STK界面可见
        app.Visible = True
        
        # 获取STK根对象
        root = app.Personality2
        # 设置场景
        scenario_object = setup_stk_scenario(app)
        
        # 创建星座
        create_satellite_constellation(scenario_object, root)
        # 导出轨迹数据
        export_satellite_trajectories(scenario_object, root)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("\nScript finished. Press Enter to close STK and exit.")
        input() # 等待用户按键，以便在退出前查看STK界面或控制台输出
        if app is not None:
            app.Quit()


if __name__ == '__main__':
    main()