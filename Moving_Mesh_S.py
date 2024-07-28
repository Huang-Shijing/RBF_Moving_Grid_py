import numpy as np
import RBF
from PLOT import plot_grid_S
from read_grid import Coord , wallNodes , Grid

xCoord = np.array([Coord[:, 0]]).T
yCoord = np.array([Coord[:, 1]]).T
nNodes = Coord.shape[0]
nWallNodes = len(wallNodes)

r0 = 10.0      #紧支半径
basis = 11      #基函数类型

xCoord_new = np.array([xCoord[:, 0]]).T #避免改变原数组
yCoord_new = np.array([yCoord[:, 0]]).T

# 对物面点进行变形
dy = np.zeros((nWallNodes, 1))
for i in range(nWallNodes):
    wall_index = wallNodes[i] - 1
    dy[i,0] = 0.2 * np.sin(-2*np.pi * xCoord_new[wall_index] )
    yCoord_new[wall_index] += dy[i,0]

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

# 绘制网格
plot_grid_S(Grid, xCoord_new, yCoord_new)
