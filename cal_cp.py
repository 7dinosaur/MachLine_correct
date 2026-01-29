import numpy as np
from numpy.typing import NDArray
from matplotlib import pyplot as plt
import pandas as pd
import os
from typing import Any
    
class Element:
    def __init__(
        self,
        vertex1: NDArray | list[float],
        vertex2: NDArray | list[float],
        vertex3: NDArray | list[float],
        normal_x: float,
        normal_y: float,
        normal_z: float,
        cell_data: dict[str, float],
        point_data: dict) -> None:
        self.vertex1 = np.array(vertex1)
        self.vertex2 = np.array(vertex2)
        self.vertex3 = np.array(vertex3)
        self.vertices: list[NDArray] = [self.vertex1, self.vertex2, self.vertex3]
        self.normal_x: float = normal_x
        self.normal_y: float = normal_y
        self.normal_z: float = normal_z

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
        return self.point_data.get(key)

    def __repr__(self) -> str:
        area = self.area
        return (
            f"Element(area={area:.6f}, "
            f"normal=({self.normal_x:.4f}, {self.normal_y:.4f}, {self.normal_z:.4f}), "
            f"cell_data={list(self.cell_data.keys())}), "
            f"point_data={list(self.point_data.keys())})"
        )
    
class Block:
    def __init__(self) -> None:
        self.elements : list[Element] = []
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
    
    def write_dat(self) -> None:
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
            title1 = "TITLE = \"Panel Data\"\n"+"VARIABLES = \"X\", \"Y\", \"Z\"\n"
            title2 = f"ZONE T=\"Triangle Mesh\", N={points_count-1}, E={len(self.elements)}, DATAPACKING=BLOCK, ZONETYPE=FETRIANGLE\n"
            title3 = f"VARLOCATION=([1-3]=NODAL)\n"
            f.writelines([title1, title2, title3])
            points_lines = []
            cells_lines = []
            polygen_lines = []
            ##先写节点数据
            keys_list = list(self.points.keys())
            points = np.array([[float(v[0])*1e-6, float(v[1])*1e-6, float(v[2])*1e-6] for v in keys_list], dtype=float).T.flatten()
            for i in range(0, len(points), 5):
                chunk = points[i:i+5]
                points_lines.append(" ".join(f"{v:.6f}" for v in chunk) + "\n")
            f.write(''.join(points_lines))
            ##再写单元数据
            ##最后写索引
            for e in self.elements:
                polygen_lines.append(" ".join(f"{p}" for p in e.get_point_data("polygen")) + "\n")
            f.write(''.join(polygen_lines))

def read_block(vtk_file: str) -> Block:
    ##读取vtk数据
    with open(vtk_file, mode="r") as f:
        data = f.readlines()

    mark_line = []
    var_list = ["X","Y","Z"]

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
        n1 = cell_data_dict[normal_keys[0]][i]; n2 = cell_data_dict[normal_keys[1]][i]; n3 = cell_data_dict[normal_keys[2]][i]
        data_dict = {}
        point_dict = {}
        for key in cell_data_dict:
            # 跳过法向量键，只处理其他格心数据
            if key not in normal_keys:
                # 按索引i取出当前面元的该字段值，存入字典
                data_dict[key] = cell_data_dict[key][i]
        for key in point_data_dict:
            da1 = point_data_dict[key][int(poly_idx[0])]
            da2 = point_data_dict[key][int(poly_idx[1])]
            da3 = point_data_dict[key][int(poly_idx[2])]
            # data_dict[key] = (da1+da2+da3)/3
            point_dict[key] = [da1, da2, da3]
        e = Element(v1, v2, v3, n1, n2, n3, data_dict, point_dict)
        aircraft.add_element(e)

    return aircraft
    
def cal_cp(aircraft: Block) -> float:
    Lift = 0.0
    Sref = 300.6
    for e in aircraft.elements:
        # if e.vertex1[0] > 62.7 and e.vertex1[1] < 5.23:
        if False:
            Lift += 0
        else:
            Cp = e.get_cell_data("C_p_ise")
            Si = e.area
            nz = e.normal_z
            dL = -Cp * Si * nz / Sref
            Lift += dL

    return Lift

def cal_V(aircraft: Block, observe_points: NDArray) -> NDArray:
    V = []
    for ob_point in observe_points:
        xp = ob_point[0]; yp = ob_point[1]; zp = ob_point[2]
        Vi = np.zeros([3])
        for e in aircraft.elements:
            cx = e.get_cell_data("centroid_x"); cy = e.get_cell_data("centroid_y"); cz = e.get_cell_data("centroid_z")
            n = np.array([e.normal_x, e.normal_y, e.normal_z])
            r = np.array([xp - cx, yp - cy, zp - cz])
            if r[1]**2 + r[2]**2 <= 3*r[0]**2:
                mu = e.get_cell_data("mu")*1000
                # mu = e.normal_x * (2 + e.get_cell_data("C_p_2nd"))
                # mu = e.area * e.get_cell_data("mu")
                mo = np.sqrt(r[0]**2 + r[1]**2 + r[2]**2) + 1e-8
                er = r/mo
                K = n[0]*er[0] + n[1]*er[1] + n[2]*er[2]
                Dv = mu/(4*np.pi*(mo**3))*(3*K*er - n)
                Vi += Dv
        V.append(Vi)

    V = np.array(V)

    return V

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

def main() -> None:
    aircraft = read_block("7105_notail.vtk")
    aircraft.write_dat()
    print(aircraft.elements[0].get_point_data("polygen"))
    # read_csv()
    # lift = cal_cp(aircraft)
    # observe_points = gene_observe(72.0, 2.0)
    # V = cal_V(aircraft, observe_points)
    # plt.plot(observe_points[:, 0], V[:, 0])
    # print(V)

if __name__ == "__main__":
    main()
    plt.show()