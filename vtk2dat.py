import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import glob

def vtk2dat(vtk_file : str, dat_file : str):
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
    polygons += 1

    point_data_1 = []
    if point_data != None:
        for idx, da in enumerate(point_data):
            da = da.strip().split()
            if da[0] == "SCALARS":
                point_data_1.append(point_data[idx+2:idx+2+point_num])
                var_list.append(da[1])
        
    nodal_num = len(var_list)
    cell_data_1 = []
    if cell_data != None:
        for idx, da in enumerate(cell_data):
            da = da.strip().split()
            if da[0] == "SCALARS":
                cell_data_1.append(cell_data[idx+2:idx+2+cell_num])
                var_list.append(da[1])
            if da[0] == "VECTORS" or da[0] == "NORMALS":
                vector = np.array([da.strip().split() for da in cell_data[idx+1:idx+1+cell_num]])
                cell_data_1.append([str(num) + '\n' for num in vector[:, 0].tolist()])
                var_list.append(da[1]+'x')
                cell_data_1.append([str(num) + '\n' for num in vector[:, 1].tolist()])
                var_list.append(da[1]+'y')
                cell_data_1.append([str(num) + '\n' for num in vector[:, 2].tolist()])
                var_list.append(da[1]+'z')
                # vector = np.float32(vector)
                # vector_abs = np.sqrt(vector[:, 0]**2 + vector[:, 1]**2 + vector[:, 2]**2)
                # cell_data_1.append([str(num) + '\n' for num in vector_abs.tolist()])
                # var_list.append(da[1]+'abs')

    ##写入dat数据
    title1 = ["TITLE = \"Panel Data\"\n",
            "VARIABLES = "]
    title2 = f"ZONE T=\"Triangle Mesh\", N={point_num}, E={cell_num}, DATAPACKING=BLOCK, ZONETYPE=FETRIANGLE\n"
    if len(var_list) > nodal_num:
        title3 = f"VARLOCATION=([1-{nodal_num}]=NODAL, [{nodal_num+1}-{len(var_list)}]=CELLCENTERED)\n"
    else:
        title3 = f"VARLOCATION=([1-{nodal_num}]=NODAL)\n"
    with open(dat_file, mode="w") as f:
        f.writelines(title1)
        for var in var_list:
            f.write(f" \"{var}\"")
        f.write("\n")
        f.write(title2)
        f.write(title3)
        count = 0
        for da in points.T.flatten():
            f.write(str(da)+" ")
            count += 1
            if count == 5:
                f.write("\n")
                count = 0
        f.write("\n")
        if point_data_1:
            for da in point_data_1:
                f.writelines(da)
                f.write("\n")
        for da in cell_data_1:
            f.writelines(da)
            f.write("\n")
        for da in polygons:
            line = ' '.join(str(int(x)) for x in da[1:])
            f.write(line)
            f.write("\n")


def main():

    vtk_files = glob.glob("*.vtk")
    
    if not vtk_files:
        print("在当前目录下未找到任何.vtk文件。")
        return
    
    print(f"找到 {len(vtk_files)} 个.vtk文件，开始转换...")
    success_count = 0

    for vtk_file in vtk_files:
        # 生成输出的dat文件名（主文件名相同，后缀改为.dat）
        dat_file = os.path.splitext(vtk_file)[0] + ".dat"
        
        print(f"正在转换: {vtk_file} -> {dat_file}", end=" ... ")
        
        try:
            vtk2dat(vtk_file, dat_file)
            print("[成功]")
            success_count += 1
        except FileNotFoundError:
            print(f"[失败] 错误：找不到文件 '{vtk_file}'。")
        except PermissionError:
            print(f"[失败] 错误：没有权限读写文件。")
        except Exception as e:
            print(f"[失败] 错误：{e}")
    
    print(f"\n转换完成！成功处理 {success_count}/{len(vtk_files)} 个文件。")

if __name__ == "__main__":
    main()