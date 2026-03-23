from dis import disco
from socket import AI_CANONNAME

import numpy as np
from numpy.typing import NDArray
from matplotlib import pyplot as plt
import pandas as pd
import os
from typing import Any
import copy
    
class Element:
    def __init__(
        self,
        vertex1: NDArray | list[float],
        vertex2: NDArray | list[float],
        vertex3: NDArray | list[float],
        cell_data: dict[str, float],
        point_data: dict) -> None:
        self.vertex1 = np.array(vertex1)
        self.vertex2 = np.array(vertex2)
        self.vertex3 = np.array(vertex3)
        self.vertices: list[NDArray] = [self.vertex1, self.vertex2, self.vertex3]

        self.cell_data: dict[str, float] = cell_data
        self.point_data: dict[str, list] = point_data

        vec1 = self.vertex2 - self.vertex1
        vec2 = self.vertex3 - self.vertex1
        cross = np.cross(vec1, vec2)
        self.area = float(0.5 * np.linalg.norm(cross))
    
    def add_cell_data(self, key: str, value: float) -> None:
        self.cell_data[key] = value

    def add_point_data(self, key: str, value: list) -> None:
        self.point_data[key] = value

    def get_cell_data(self, key: str) -> Any:
        return self.cell_data.get(key)
    
    def get_point_data(self, key: str) -> Any:
        return np.array(self.point_data.get(key))

    def __repr__(self) -> str:
        area = self.area
        return (
            f"Element(area={area:.6f}, "
            f"cell_data={list(self.cell_data.keys())}), "
            f"point_data={list(self.point_data.keys())})"
        )
    
class Block:
    def __init__(self) -> None:
        self.elements : list[Element] = []
        self.tail_elements : list[Element] = []
        self.points : dict[tuple[float, float, float], list] = {}

    def add_element(self, element: Element) -> None:
        """添加单个面元"""
        self.elements.append(element)

    def get_all_elements(self) -> list[Element]:
        """获取所有面元"""
        return self.elements
    
    def get_data_by_key(self, key: str) -> list[float | None]:
        return [e.get_cell_data(key) for e in self.elements]

    def get_element_by_value(self, key: str, value: float) -> list[Element]:
        return [e for e in self.elements if e.get_cell_data(key) == value]
    
    def read_wake(self, wake_vtk_file: str):
        wake_block = read_block(wake_vtk_file)
        wake_vertex_dict = {}
        for ew in wake_block.elements:
            v_wake1, v_wake2, v_wake3 = ew.vertices
            mu1, mu2, mu3 = ew.get_point_data("mu")  # 三个顶点的mu

            # 逐个顶点加入字典（自动去重）
            p1 = tuple(v_wake1.round(9))  # 转元组才能做key，浮点精度9位足够
            p2 = tuple(v_wake2.round(9))
            p3 = tuple(v_wake3.round(9))

            wake_vertex_dict[p1] = mu1
            wake_vertex_dict[p2] = mu2
            wake_vertex_dict[p3] = mu3

        # 🔥 最终输出：无重复节点列表 + 对应mu列表
        wake_ver = np.array([np.array(p) for p in wake_vertex_dict.keys()])  # 节点列表
        wake_mu = list(wake_vertex_dict.values())                  # 对应mu列表
        print(wake_ver.shape)

        count = 0
        for e in self.elements:
            e.add_cell_data("is_tail", 0)
            discon = 0
            v1, v2, v3 = e.vertices
            v1 = np.array(v1).reshape(1, 3); v2 = np.array(v2).reshape(1, 3); v3 = np.array(v3).reshape(1, 3)
            new_mu = e.get_point_data("mu").copy()
            for i in range(3):
                v = locals()[f"v{i+1}"]
                ver_diff = wake_ver - v
                ver_diff = np.linalg.norm(ver_diff, axis=1)
                for j,d in enumerate(ver_diff):
                    if d < 1e-5:
                        discon += 1
                        count += 1
                        continue
            if discon > 0:
                self.tail_elements.append(e)

            e.add_cell_data("discon", discon)
            e.add_point_data("mu2", new_mu)

        print(count)
    
    def write_dat(self) -> None:
        var_list = ["X", "Y", "Z"] + list(self.elements[0].point_data.keys())\
                     + list(self.elements[0].cell_data.keys())
        N_data_end = 3 + len(list(self.elements[0].point_data.keys()))
        points_count = 1
        for e in self.elements:
            polygen = [0, 0, 0]
            for v_idx, ver in enumerate(e.vertices):
                p_tmp = (int(round(ver[0]*1e6)), int(round(ver[1]*1e6)), int(round(ver[2]*1e6)))
                if p_tmp in self.points:
                    polygen[v_idx] = self.points[p_tmp][0]
                else:
                    self.points[p_tmp] = [points_count]
                    point_data = e.point_data
                    for item in point_data.values():
                        self.points[p_tmp].append(item[v_idx])
                    polygen[v_idx] = self.points[p_tmp][0]
                    points_count += 1
            e.add_point_data("polygen", polygen)
        with open("test.dat", "w") as f:
            title1 = "TITLE = \"Panel Data\"\n"+"VARIABLES = "
            var_head = " ".join([f"\"{v}\"" for v in var_list]) + "\n"
            title2 = f"ZONE T=\"Triangle Mesh\", N={points_count-1}, E={len(self.elements)}, DATAPACKING=BLOCK, ZONETYPE=FETRIANGLE\n"
            title3 = f"VARLOCATION=([1-{N_data_end}]=NODAL, [{N_data_end+1}-{len(var_list)}]=CELLCENTERED)\n"
            f.writelines([title1, var_head, title2, title3])
            points_lines = []
            cells_lines = []
            polygen_lines = []
            ##先写节点数据
            keys_list = list(self.points.keys())
            points = np.array([[float(v[0])*1e-6, float(v[1])*1e-6, float(v[2])*1e-6] for v in keys_list], dtype=float).T.flatten()
            other_data = np.array([b[1:] for b in self.points.values()], dtype=float).T.flatten()
            points = np.concatenate([points, other_data]).flatten()
            for i in range(0, len(points), 5):
                chunk = points[i:i+5]
                points_lines.append(" ".join(f"{v:.6f}" for v in chunk) + "\n")
            f.write(''.join(points_lines))
            ##再写单元数据
            ##最后写索引
            cells_data = []

            for e in self.elements:
                polygen_lines.append(" ".join(f"{p}" for p in e.get_point_data("polygen")) + "\n")
                cells_data.append([c for c in e.cell_data.values()])
            cells_data = np.array(cells_data).T.flatten()
            for data in cells_data:
                cells_lines.append(f"{data}" + "\n")
            f.write(''.join(cells_lines))
            f.write(''.join(polygen_lines))

        print("写出数据, 路径为test.dat")

    def cal_v_vert(self) -> None:
        v_free = np.array([1., 0., 0.], dtype=float)
        Ma = 1.6; B = np.sqrt(Ma**2 - 1)
        C_hat_g = v_free/np.linalg.norm(v_free)
        C_mat_g = (1 - Ma**2)*np.eye(3) + Ma**2*np.outer(C_hat_g, C_hat_g)
        print(C_mat_g)
        for e in self.elements:
            #读取三个节点坐标与mu强度
            if e.get_cell_data("discon") > 0:
                continue
            ver = e.vertices
            normal_name = ["normals_x", "normals_y", "normals_z"]
            n1, n2, n3 = [e.get_cell_data(n) for n in normal_name]
            normal = np.array([n1, n2, n3], dtype=float)
            mu = np.array(e.get_point_data("mu2"), dtype=float)
            #计算面元局部坐标
            v0 = np.cross(normal, v_free); v0 = v0/np.linalg.norm(v0)
            u0 = np.cross(v0, normal); u0 = u0/np.linalg.norm(u0)
            B_mat = np.zeros([3, 3]); B_mat[0,0] = 1 - Ma**2; B_mat[1, 1] = 1; B_mat[2, 2] = 1
            vg = B_mat @ normal
            x = normal @ vg
            assert(x > 0), "!!! Panel is superinclined, which is not allowed."
            y = 1.0/np.sqrt(abs(x))
            rs = -1 * x/abs(x)
            A_g_ls = np.zeros([3, 3])
            A_g_ls[0, :] = y*np.dot(C_mat_g, u0)
            A_g_ls[1, :] = rs/B*np.dot(C_mat_g, v0)
            A_g_ls[2, :] = B*y*normal
            # A_ls_g = np.linalg.inv(A_g_ls)
            centroid = (ver[0] + ver[1] + ver[2])/3
            P_rel = (ver - centroid).T
            P_ls = A_g_ls[:2] @ P_rel
            S_mu = np.ones([3, 3]); S_mu[:, 1:] = P_ls.T
            T_mu = np.linalg.inv(S_mu)
            mu_param = T_mu @ mu
            dv = np.array([mu_param[1], mu_param[2], 0], dtype=float)
            dv = A_g_ls.T @ dv

            e.add_cell_data("dv_cx", dv[0])
            e.add_cell_data("dv_cy", dv[1])
            e.add_cell_data("dv_cz", dv[2])

        print("正在为尾缘面板赋值最近邻 dv...")
        # 先收集所有 非尾缘 面板的质心 + dv
        non_tail_centers = []
        non_tail_dvs = []
        for e in self.elements:
            if e.get_cell_data("discon") == 0:
                c = (e.vertex1 + e.vertex2 + e.vertex3) / 3
                dvx = e.get_cell_data("dv_cx")
                dvy = e.get_cell_data("dv_cy")
                dvz = e.get_cell_data("dv_cz")
                non_tail_centers.append(c)
                non_tail_dvs.append((dvx, dvy, dvz))

        non_tail_centers = np.array(non_tail_centers)
        non_tail_dvs = np.array(non_tail_dvs)

        # 遍历尾缘面板，找最近非尾缘面板
        for e in self.elements:
            if e.get_cell_data("discon") == 0:
                continue

            # 当前尾缘面板质心
            c = (e.vertex1 + e.vertex2 + e.vertex3) / 3
            # 最近邻搜索
            dists = np.linalg.norm(non_tail_centers - c, axis=1)
            idx = np.argmin(dists)
            # 赋值
            dvx, dvy, dvz = non_tail_dvs[idx]
            e.add_cell_data("dv_cx", dvx)
            e.add_cell_data("dv_cy", dvy)
            e.add_cell_data("dv_cz", dvz)

        print("✅ 所有尾缘面板 dv 已赋值为最近邻非尾缘值！")



def read_block(vtk_file: str) -> Block:
    ##读取vtk数据
    with open(vtk_file, mode="r") as f:
        data = f.readlines()

    cell_start = None
    point_start = None
    point_data = None
    cell_data = None

    for idx, da in enumerate(data):
        da = da.split()
        if da[0] == "POINTS":
            point_num = int(da[1])
            points_line = data[idx+1:idx+1+point_num]
        elif da[0] == "POLYGONS":
            cell_num = int(da[1])
            cells_line = data[idx+1:idx+1+cell_num]
        elif da[0] == "CELL_DATA":
            cell_start = idx
        elif da[0] == "POINT_DATA":
            point_start = idx
            if cell_start:
                cell_data = data[cell_start+1:point_start]
            point_data = data[point_start+1:]
            break

    if (point_start == None and cell_start != None):
        cell_data = data[cell_start+1:]
    
    points = np.zeros([point_num, 3])
    polygons = np.zeros([cell_num, 4])
    for idx, da in enumerate(points_line):
        da = np.float32(da.split())
        points[idx] = np.array(da)
    for idx, da in enumerate(cells_line):
        da = np.int32(da.split())
        polygons[idx] = np.array(da)
    # polygons += 1

    point_data_dict = {}
    if point_data != None:
        for idx, da in enumerate(point_data):
            da = da.strip().split()
            if da[0] == "SCALARS":
                point_data_dict[str(da[1])] = np.float64(point_data[idx+2:idx+2+point_num])

    cell_data_dict = {}
    if cell_data != None:
        for idx, da in enumerate(cell_data):
            da = da.strip().split()
            if da[0] == "SCALARS":
                cell_data_dict[str(da[1])] = np.float64(cell_data[idx+2:idx+2+cell_num])
            if da[0] == "VECTORS" or da[0] == "NORMALS":
                vector = np.array([da.strip().split() for da in cell_data[idx+1:idx+1+cell_num]])
                cell_data_dict[da[1]+'_x'] = [float(num) for num in vector[:, 0].tolist()]
                cell_data_dict[da[1]+'_y'] = [float(num) for num in vector[:, 1].tolist()]
                cell_data_dict[da[1]+'_z'] = [float(num) for num in vector[:, 2].tolist()]

    # print(point_data_dict.keys())
    # print(cell_data_dict.keys())
    aircraft = Block()
    normal_keys = ["normals_x", "normals_y", "normals_z"]
    for i in range(cell_num):
        poly_idx = polygons[i, 1:]
        v1 = points[int(poly_idx[0])]; v2 = points[int(poly_idx[1])]; v3 = points[int(poly_idx[2])]
        data_dict = {}
        point_dict = {}
        for key in cell_data_dict:
            # 按索引i取出当前面元的该字段值，存入字典
            data_dict[key] = cell_data_dict[key][i]
        for key in point_data_dict:
            da1 = point_data_dict[key][int(poly_idx[0])]
            da2 = point_data_dict[key][int(poly_idx[1])]
            da3 = point_data_dict[key][int(poly_idx[2])]
            # data_dict[key] = (da1+da2+da3)/3
            point_dict[key] = [da1, da2, da3]
        e = Element(v1, v2, v3, data_dict, point_dict)
        aircraft.add_element(e)

    return aircraft

def read_wake(wake_vtk_file: str, aircraft_origin: Block) -> Block:
    aircraft = aircraft_origin.copy()
    wake_block = read_block(wake_vtk_file)
    count = 0
    for ew in wake_block.elements:
        v_wake1, v_wake2, v_wake3 = ew.vertices
        wake_mu = ew.get_point_data("mu")
        for i in range(len(aircraft.elements)):
            e = aircraft.elements[i]   # 取出内部真实对象
            v1, v2, v3 = e.vertices

            # 每个面元只复制一次 mu
            new_mu = e.get_point_data("mu").copy()
            updated = False

            # 三个顶点逐点判断更新
            for j in range(3):
                ev = [v1, v2, v3][j]
                wv = [v_wake1, v_wake2, v_wake3][j]
                if np.linalg.norm(ev - wv) < 1e-6:
                    new_mu[j] = wake_mu[j]
                    updated = True
            e.add_point_data
            # 直接修改内部真实元素
            if updated:
                aircraft.elements[i].add_point_data("mu2", new_mu)
                count += 1
                      
    print(count)
    
    return aircraft
    
def cal_cp(aircraft: Block) -> float:
    Lift = 0.0
    Sref = 300.6
    for e in aircraft.elements:
        # if e.vertex1[0] > 62.7 and e.vertex1[1] < 5.23:
        if False:
            Lift += 0
        else:
            Cp = e.get_cell_data("C_p_2nd")
            Si = e.area
            nz = e.get_cell_data("normals_z")
            dL = -Cp * Si * nz / Sref
            Lift += dL

    return Lift

def gene_observe(aircraft_length: float, Mach: float, Rovel: float = 3, n_sample: int = 200) -> NDArray:
    y = np.ones([n_sample]) * 0.0
    z = np.ones([n_sample]) * -aircraft_length * Rovel
    x = np.linspace(0, 1, n_sample) * aircraft_length
    x += 1/np.tan(np.arcsin(1/Mach)) * (aircraft_length * Rovel)
    observe_points = np.stack([x, y, z], axis=0).T

    return observe_points

def read_csv():
    filename = f"7105_notail_dp.csv"
    if os.path.isfile(filename):
        print(f"文件 {filename} 存在！")
    else:
        raise FileNotFoundError("当前目录下未找到CSV文件！")
        
    csv_path = filename
    
    col_mapping = {
        'x': 0,    # 第1列
        'vx': 8,   # 第9列
        'vy': 9,   # 第10列
        'vz': 10,  # 第11列
        'Vx': 20,  # 第21列
        'Vy': 21,  # 第22列
        'Vz': 22   # 第23列
    }
    
    # -------------------------- 2. 读取CSV数据 --------------------------
    try:
        df = pd.read_csv(csv_path, encoding='utf-8', sep=None, engine='python')
        print(f"成功读取CSV文件：{csv_path}，共{len(df)}行，{len(df.columns)}列")
        
        max_col_idx = max(col_mapping.values())
        if len(df.columns) <= max_col_idx:
            raise ValueError(f"CSV文件列数不足！需要至少{max_col_idx+1}列，当前只有{len(df.columns)}列")
        
        # 提取数据并保存（用于最后绘图）
        x = df.iloc[:, col_mapping['x']].values
        vx = df.iloc[:, col_mapping['vx']].values
        vy = df.iloc[:, col_mapping['vy']].values
        vz = df.iloc[:, col_mapping['vz']].values
        Vx = df.iloc[:, col_mapping['Vx']].values
        Vy = df.iloc[:, col_mapping['Vy']].values
        Vz = df.iloc[:, col_mapping['Vz']].values
        
    except Exception as e:
        raise RuntimeError(f"读取/解析CSV文件失败：{str(e)}")
    
    dVx = Vx - vx
    dVy = Vy - vy
    dVz = Vz - vz

    # plt.plot(x, dVx)

def check_mu_difference(aircraft_before: Block, aircraft_after: Block) -> None:
    """
    检查两个Block对象中所有面板的mu是否一致
    输出：不同的面板数量 + 具体差异
    """
    if len(aircraft_before.elements) != len(aircraft_after.elements):
        print("❌ 面板数量不一致！")
        return

    diff_count = 0
    total_count = len(aircraft_before.elements)

    print("\n" + "="*60)
    print("🔍 开始校验 mu 更新前后差异")
    print("="*60)

    for idx, (e_before, e_after) in enumerate(zip(aircraft_before.elements, aircraft_after.elements)):
        mu_before = e_before.get_point_data("mu")
        mu_after = e_after.get_point_data("mu")

        # 数值精度判断
        if not np.allclose(mu_before, mu_after, atol=1e-10):
            diff_count += 1
            # 可选：打印前5个不同的面板详情
            if diff_count <= 5:
                print(f"面板 {idx}:")
                print(f"  旧 mu = {mu_before}")
                print(f"  新 mu = {mu_after}")
                print("-"*50)

    print(f"\n✅ 校验完成：")
    print(f"   总面板数：{total_count}")
    print(f"   mu 不同的面板数：{diff_count}")
    print(f"   mu 相同的面板数：{total_count - diff_count}")

    if diff_count == 0:
        print("\n❌ 严重：mu 完全没有被修改！")
    else:
        print(f"\n✅ 成功：{diff_count} 个面板的 mu 已更新！")

def main() -> None:
    aircraft = read_block("JWB_CQ.vtk")
    aircraft.read_wake("JWB_CQ_wake.vtk")
    print(len(aircraft.tail_elements))
    aircraft.cal_v_vert()
    aircraft.write_dat()


if __name__ == "__main__":
    main()
    plt.show()