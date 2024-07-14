import scipy.io as sio
import numpy as np
import RBF
from PLOT import *
#导入matlab数据
load_f1 = 'data\WALL.mat'
load_data = sio.loadmat(load_f1)
WALL= load_data['WALL']


load_f2 = 'data\Grid.mat'
load_data = sio.loadmat(load_f2)
Grid= load_data['Grid']


load_f3 = 'data\Coord.mat'
load_data = sio.loadmat(load_f3)
Coord= load_data['Coord']


load_f4 = 'data\wallNodes.mat'
load_data = sio.loadmat(load_f4)
wallNodes= load_data['wallNodes']
wallNodes = wallNodes[0]

xCoord = np.array([Coord[:, 0]]).T
yCoord = np.array([Coord[:, 1]]).T
nNodes = Coord.shape[0]
nWallNodes = wallNodes.shape[0]

#参数
lamda = 1      #波长
c = 0.1        #波速
v = -0.2       #游动速度
T = 2.0        #周期
t = 0          #起始时间
dt = 0.5       #时间间隔
r0 = 10.0      #紧支半径
basis = 11      #基函数类型

while t < 10:
    dy = np.zeros((nWallNodes,1))
    xCoord_new = np.array([xCoord[:, 0]]).T #避免改变原数组
    yCoord_new = np.array([yCoord[:, 0]]).T
    t = t + dt

    # 对物面点进行变形
    for i in range(nWallNodes):
        wall_index = wallNodes[i] - 1
        xCoord_new[wall_index] = xCoord[wall_index] + v * t
    nose_x = min(xCoord_new[wallNodes-1])

    for i in range(nWallNodes):
        wall_index = wallNodes[i] - 1
        x = xCoord_new[wall_index] - nose_x
        y = yCoord[wall_index]

        #此处表示已知所有物面点的位移
        A = min(1, t / T) * (0.02 - 0.0825 * x + 0.1625 * x**2)
        B = A * np.sin(2 * np.pi / lamda * (x - c * t))
        dy[i,0] = B[0]
        yCoord_new[wall_index] = yCoord[wall_index] + B[0]

    # 计算权重系数矩阵W
    fai = np.zeros((nWallNodes, nWallNodes))

    for i in range(nWallNodes):
        wall_index = wallNodes[i] - 1
        x1 = xCoord[wall_index]
        y1 = yCoord[wall_index]
        for j in range(nWallNodes):
            wall_index2 = wallNodes[j] - 1
            x2 = xCoord[wall_index2]
            y2 = yCoord[wall_index2]
            
            #距离加上1e-40防止除以零
            dis = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) + 1e-40
            fai[i, j] = RBF.RBF_func(dis[0], r0, basis)
    W = np.dot(np.linalg.inv(fai),dy)

    # 利用W计算内场点的位移
    # 这里将远场边界点通过插值更改了坐标，不清楚是否需要保持不变或者保持整个框架不变
    fai = np.zeros((1, nWallNodes))

    for i in range(nNodes):
        xNode = xCoord[i]
        yNode = yCoord[i]
        if i in wallNodes - 1:
            continue
        for j in range(nWallNodes):
            wall_index = wallNodes[j] - 1
            xw = xCoord[wall_index]
            yw = yCoord[wall_index]

            dis = np.sqrt((xNode - xw)**2 + (yNode - yw)**2) + 1e-40
            fai[0, j] = RBF.RBF_func(dis[0], r0, basis)

        dy = np.dot(fai[0, :] , W)  
        yCoord_new[i] = yCoord_new[i] + dy[0]  

        dx = v * t  
        xCoord_new[i] = xCoord_new[i] + dx 

    # 绘制网格
    plot_grid(Grid, xCoord_new, yCoord_new, nose_x)

    if input("按任意键继续生成t=t0+dt时刻网格，或输入q退出循环:\n") == 'q':
        break

