import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d
import numpy as np
import math
import os
import cv2
from scipy.interpolate import griddata
from scipy import fftpack
import time


def find_marker(gray):
    mask = cv2.inRange(gray, 0, 70)
    # kernel = np.ones((5,5), np.uint8)
    # dilation = cv2.dilate(mask, kernel, iterations=2)
    return mask

def dilate(img, ksize=3, iter=1):
    # 生成带平滑过渡的核
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

    # 迭代膨胀
    dilated = cv2.dilate(img, kernel, iterations=iter)

    # 最终结果后处理
    dilated = cv2.GaussianBlur(dilated.astype(np.float32), (3, 3), 0)
    return (dilated > 0.4).astype(np.uint8) * 255

def erode(img, ksize=3):
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.erode(img, kernel, iterations=1)

def matching_rows(A,B):
    ### https://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays
    matches=[i for i in range(B.shape[0]) if np.any(np.all(A==B[i],axis=1))]
    if len(matches)==0:
        return B[matches]
    return np.unique(B[matches],axis=0)

def interpolate_gradients(gx, gy, img, cm, markermask):
    ''' interpolate gradients at marker location '''
    # if np.where(cm)[0].shape[0] != 0:
    cmcm = np.zeros(img.shape[:2])
    ind1 = np.vstack(np.where(cm)).T
    ind2 = np.vstack(np.where(markermask)).T
    ind2not = np.vstack(np.where(~markermask)).T
    ind3 = matching_rows(ind1, ind2)
    cmcm[(ind3[:, 0], ind3[:, 1])] = 1.
    ind4 = ind1[np.all(np.any((ind1 - ind3[:, None]), axis=2), axis=0)]
    x = np.linspace(0, 240, 240)
    y = np.linspace(0,320, 320)
    X, Y = np.meshgrid(x, y)

    '''interpolate at the intersection of cm and markermask '''
    # gx_interpol = griddata(ind4, gx[(ind4[:, 0], ind4[:, 1])], ind3, method='nearest')
    # gx[(ind3[:, 0], ind3[:, 1])] = gx_interpol
    # gy_interpol = griddata(ind4, gy[(ind4[:, 0], ind4[:, 1])], ind3, method='nearest')
    # gy[(ind3[:, 0], ind3[:, 1])] = gy_interpol

    ''' interpolate at the entire markermask '''
    gx_interpol = griddata(ind2, gx[(ind2[:, 0], ind2[:, 1])], gx[(ind2not[:, 0], ind2not[:, 1])], method='nearest')
    gx[(ind2not[:, 0], ind2not[:, 1])] = gx_interpol
    gy_interpol = griddata(ind2, gy[(ind2[:, 0], ind2[:, 1])], gy[(ind2not[:, 0], ind2not[:, 1])], method='nearest')
    gy[(ind2not[:, 0], ind2not[:, 1])] = gy_interpol
    #print (gy_interpol.shape, gx_interpol.shape, gx.shape, gy.shape)

    ''' interpolate using samples in the vicinity of marker '''


    ''' method #3 '''
    # ind1 = np.vstack(np.where(markermask)).T
    # gx_interpol = scipy.ndimage.map_coordinates(gx, [ind1[:, 0], ind1[:, 1]], order=1, mode='constant')
    # gx[(ind1[:, 0], ind1[:, 1])] = gx_interpol
    # gy_interpol = scipy.ndimage.map_coordinates(gy, [ind1[:, 0], ind1[:, 1]], order=1, mode='constant')
    # gx[(ind1[:, 0], ind1[:, 1])] = gy_interpol

    ''' method #4 '''
    # x = np.arange(0, img.shape[0])
    # y = np.arange(0, img.shape[1])
    # fgx = scipy.interpolate.RectBivariateSpline(x, y, gx, kx=2, ky=2, s=0)
    # gx_interpol = fgx.ev(ind2[:,0],ind2[:,1])
    # gx[(ind2[:, 0], ind2[:, 1])] = gx_interpol
    # fgy = scipy.interpolate.RectBivariateSpline(x, y, gy, kx=2, ky=2, s=0)
    # gy_interpol = fgy.ev(ind2[:, 0], ind2[:, 1])
    # gy[(ind2[:, 0], ind2[:, 1])] = gy_interpol

    return gx_interpol, gy_interpol

## 在标记区域插值梯度
def interpolate_grad(img, mask):
    # mask = (soft_mask > 0.5).astype(np.uint8) * 255
    # pixel around markers
    mask_around = (dilate(mask, ksize=3, iter=2) > 0) & ~(mask != 0)
    # mask_around = mask == 0
    mask_around = mask_around.astype(np.uint8)

    x, y = np.arange(img.shape[0]), np.arange(img.shape[1])
    yy, xx = np.meshgrid(y, x)

    # mask_zero = mask == 0
    mask_zero = mask_around == 1
    # cv2.imshow("mask_zero", mask_zero*1.)

    # if np.where(mask_zero)[0].shape[0] != 0:
    #     print ('interpolating')
    mask_x = xx[mask_around == 1]
    mask_y = yy[mask_around == 1]
    points = np.vstack([mask_x, mask_y]).T
    values = img[mask_x, mask_y]
    markers_points = np.vstack([xx[mask != 0], yy[mask != 0]]).T
    method = "nearest"
    # method = "linear"
    # method = "cubic"
    x_interp = griddata(points, values, markers_points, method=method) # 使用griddata进行最近邻/线性插值 处理标记区域与非标记区域的边界
    x_interp[x_interp != x_interp] = 0.0
    ret = img.copy()
    ret[mask != 0] = x_interp
    # else:
    #     ret = img
    return ret

## 用插值替换掩盖区域
def demark(gx, gy, markermask):
    # mask = find_marker(img)
    gx_interp = interpolate_grad(gx.copy(), markermask)
    gy_interp = interpolate_grad(gy.copy(), markermask)
    return gx_interp, gy_interp

#@njit(parallel=True)
def get_features(img,pixels,features,imgw,imgh):
    features[:,3], features[:,4]  = pixels[:,0] / imgh, pixels[:,1] / imgw
    for k in range(len(pixels)):
        i,j = pixels[k]
        rgb = img[i, j] / 255.
        features[k,:3] = rgb

#
# 2D integration via Poisson solver
# 泊松重建
def poisson_dct_neumaan(gx,gy):

    gxx = 1 * (gx[:, (list(range(1, gx.shape[1])) + [gx.shape[1] - 1])] - gx[:, ([0] + list(range(gx.shape[1] - 1)))])
    gyy = 1 * (gy[(list(range(1,gx.shape[0]))+[gx.shape[0]-1]), :] - gy[([0]+list(range(gx.shape[0]-1))), :])

    f = gxx + gyy

    ### Right hand side of the boundary condition
    b = np.zeros(gx.shape)
    b[0,1:-2] = -gy[0,1:-2]
    b[-1,1:-2] = gy[-1,1:-2]
    b[1:-2,0] = -gx[1:-2,0]
    b[1:-2,-1] = gx[1:-2,-1]
    b[0,0] = (1/np.sqrt(2))*(-gy[0,0] - gx[0,0])
    b[0,-1] = (1/np.sqrt(2))*(-gy[0,-1] + gx[0,-1])
    b[-1,-1] = (1/np.sqrt(2))*(gy[-1,-1] + gx[-1,-1])
    b[-1,0] = (1/np.sqrt(2))*(gy[-1,0]-gx[-1,0])

    ## Modification near the boundaries to enforce the non-homogeneous Neumann BC (Eq. 53 in [1])
    f[0,1:-2] = f[0,1:-2] - b[0,1:-2]
    f[-1,1:-2] = f[-1,1:-2] - b[-1,1:-2]
    f[1:-2,0] = f[1:-2,0] - b[1:-2,0]
    f[1:-2,-1] = f[1:-2,-1] - b[1:-2,-1]

    ## Modification near the corners (Eq. 54 in [1])
    f[0,-1] = f[0,-1] - np.sqrt(2) * b[0,-1]
    f[-1,-1] = f[-1,-1] - np.sqrt(2) * b[-1,-1]
    f[-1,0] = f[-1,0] - np.sqrt(2) * b[-1,0]
    f[0,0] = f[0,0] - np.sqrt(2) * b[0,0]

    ## 离散余弦变换（DCT）求解
    ## Cosine transform of f   DCT变换
    tt = fftpack.dct(f, norm='ortho')
    fcos = fftpack.dct(tt.T, norm='ortho').T

    # Cosine transform of z (Eq. 55 in [1])  频域求解
    (x, y) = np.meshgrid(range(1, f.shape[1] + 1), range(1, f.shape[0] + 1), copy=True)  # 生成频率坐标
    denom = 4 * ( (np.sin(0.5*math.pi*x/(f.shape[1])))**2 + (np.sin(0.5*math.pi*y/(f.shape[0])))**2)   # 离散拉普拉斯算子的频域表示

    # 4 * ((sin(0.5 * pi * x / size(p, 2))). ^ 2 + (sin(0.5 * pi * y / size(p, 1))). ^ 2)

    f = -fcos / denom   # 频域中的解：Z = -F / Denom

    # Inverse Discrete cosine Transform逆DCT重建深度
    tt = fftpack.idct(f, norm='ortho')  # 行逆DCT
    img_tt = fftpack.idct(tt.T, norm='ortho').T  # 列逆DCT

    img_tt = img_tt.mean() + img_tt
    # img_tt = img_tt - img_tt.min()

    return img_tt


''' nn architecture for mini '''
class RGB2NormNet(nn.Module):
    def __init__(self):
        super(RGB2NormNet, self).__init__() #关键父类初始化
        input_size = 5
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,64)
        self.fc4 = nn.Linear(64,2)
        self.drop_layer = nn.Dropout(p=0.05)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop_layer(x)
        x = F.relu(self.fc2(x))
        x = self.drop_layer(x)
        x = F.relu(self.fc3(x))
        x = self.drop_layer(x)
        x = self.fc4(x)
        return x

class Reconstruction3D:
    def __init__(self, dev):
        self.cpuorgpu = "cpu"
        self.dm_zero_counter = 0
        self.dm_zero = np.zeros((dev.imgw, dev.imgh))
        self.dilated_mm = None  # 新增成员变量存储膨胀掩膜
        self.initial_marker_matrix = None  # 新增初始坐标存储
        self.current_delta = None       # 当前帧变化量 (7,9,2)
        self.current_marker_matrix = None  # 新增当前帧标记矩阵属性
        # self.marker_matrix = None
        from sklearn.cluster import KMeans
        self.KMeans = KMeans  # 缓存类引用
        pass

    def load_nn(self, net_path, cpuorgpu):

        self.cpuorgpu = cpuorgpu
        device = torch.device(cpuorgpu)

        if not os.path.isfile(net_path):
            print('Error opening ', net_path, ' does not exist')
            return


        net = RGB2NormNet().float().to(device)

        if cpuorgpu=="cuda":
            ### load weights on gpu
            # net.load_state_dict(torch.load(net_path))
            checkpoint = torch.load(net_path, map_location=lambda storage, loc: storage.cuda(0) ,weights_only=True)
            net.load_state_dict(checkpoint['state_dict'])
        else:
            ### load weights on cpu which were actually trained on gpu
            checkpoint = torch.load(net_path, map_location=lambda storage, loc: storage ,weights_only=True)
            net.load_state_dict(checkpoint['state_dict'])

        self.net = net

        return self.net

    def get_depthmap(self, frame, mask_markers, cm=None):
        MARKER_INTERPOLATE_FLAG = mask_markers

        ''' find contact region '''
        # cm, cmindx = find_contact_mask(f1, f0)
        ###################################################################
        ### check these sizes
        ##################################################################
        if (cm is None):
            cm, cmindx = np.ones(frame.shape[:2]), np.where(np.ones(frame.shape[:2]))
        imgh = frame.shape[:2][0]
        imgw = frame.shape[:2][1]
        # cv2.imshow('Original Frame', frame)#################################测试用
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # cv2.imshow('Gray Image', gray)
        if True:   ##############################################################修改
            ''' find marker mask '''
            markermask = find_marker(gray)
            ##markermask = fill_holes(markermask)  # 新增孔洞填充（没啥用）
            ## cv2.imshow('1. Markers Raw', markermask) #########################修改

            cm = ~markermask
            '''intersection of cm and markermask '''
            # cmmm = np.zeros(img.shape[:2])
            # ind1 = np.vstack(np.where(cm)).T
            # ind2 = np.vstack(np.where(markermask)).T
            # ind2not = np.vstack(np.where(~markermask)).T
            # ind3 = matching_rows(ind1, ind2)
            # cmmm[(ind3[:, 0], ind3[:, 1])] = 1.
            cmandmm = (np.logical_and(cm, markermask)).astype('uint8')
            cmandnotmm = (np.logical_and(cm, ~markermask)).astype('uint8')

        ''' Get depth image with NN '''
        nx = np.zeros(frame.shape[:2])
        ny = np.zeros(frame.shape[:2])
        dm = np.zeros(frame.shape[:2])

        ''' ENTIRE CONTACT MASK THRU NN '''
        # if np.where(cm)[0].shape[0] != 0:
        rgb = frame[np.where(cm)] / 255     ########使用无标记区域训练
        # rgb = diffimg[np.where(cm)]
        pxpos = np.vstack(np.where(cm)).T
        # pxpos[:, [1, 0]] = pxpos[:, [0, 1]] # swapping
        pxpos[:, 0], pxpos[:, 1] = pxpos[:, 0] / imgh, pxpos[:, 1] / imgw
        # the neural net was trained using height=320, width=240
        # pxpos[:, 0] = pxpos[:, 0] / ((320 / imgh) * imgh)
        # pxpos[:, 1] = pxpos[:, 1] / ((240 / imgw) * imgw)

        features = np.column_stack((rgb, pxpos))
        features = torch.from_numpy(features).float().to(self.cpuorgpu)
        with torch.no_grad():
            self.net.eval()
            out = self.net(features)                 ######训练模型结果

        nx[np.where(cm)] = out[:, 0].cpu().detach().numpy()
        ny[np.where(cm)] = out[:, 1].cpu().detach().numpy()
        # print(nx.min(), nx.max(), ny.min(), ny.max())
        # nx = 2 * ((nx - nx.min()) / (nx.max() - nx.min())) -1
        # ny = 2 * ((ny - ny.min()) / (ny.max() - ny.min())) -1
        # print(nx.min(), nx.max(), ny.min(), ny.max())

        '''OPTION#1 normalize gradient between [a,b]'''
        # a = -5
        # b = 5
        # gx = (b-a) * ((gx - gx.min()) / (gx.max() - gx.min())) + a
        # gy = (b-a) * ((gy - gy.min()) / (gy.max() - gy.min())) + a
        '''OPTION#2 calculate gx, gy from nx, ny. '''
        ### normalize normals to get gradients for poisson
        ################################################################修改根据表面硬度动态调整法线计算的梯度阈值，忽略微观波动
        grad_threshold = 0.03  # 经验值（硬质表面建议0.01-0.03）
        nx[(np.abs(nx) < grad_threshold)] = 0
        ny[(np.abs(ny) < grad_threshold)] = 0
        ################################################################修改完成

        nz = np.sqrt(1 - nx ** 2 - ny ** 2)
        if np.isnan(nz).any():
            print ('nan found')
        nz[np.where(np.isnan(nz))] = np.nanmean(nz)
        gx = -nx / nz
        gy = -ny / nz

        if MARKER_INTERPOLATE_FLAG:
            # gx, gy = interpolate_gradients(gx, gy, img, cm, cmmm)
            # 修改dilate参数（原ksize=3, iter=2）

            self.dilated_mm = dilate(markermask, ksize=3, iter=1)  # 膨胀核
            self.dilated_mm = cv2.morphologyEx(self.dilated_mm, cv2.MORPH_CLOSE, None)  # 添加闭运算
            # 在生成dilated_mm后添加以下代码（假设图像高度为h，宽度为w）
            CROP_SIZEh = 20  # 需要裁剪的右下角区域大小
            CROP_SIZEw = 17  # 需要裁剪的右下角区域大小
            h, w = self.dilated_mm.shape
            self.dilated_mm[h - CROP_SIZEh:, w - CROP_SIZEw:] = 0  # 将右下角5x5像素置零
            # ################################################################################################修改后有好转
            # dilated_mm = cv2.GaussianBlur(dilated_mm.astype(np.float32), (3, 3), 0)
            # dilated_mm = (dilated_mm > 0.6).astype(np.uint8) * 255

            # 显示最终膨胀掩膜
            cv2.imshow('Smoothed Dilated Mask', self.dilated_mm)
            dilated_figure=cv2.cvtColor(self.dilated_mm.astype(np.uint8), cv2.COLOR_GRAY2BGR)

                # # 更新历史状态
                # self.prev_marker_matrix = self.current_marker_matrix.copy()


                # 示例：打印第一个点的坐标
                # print(marker_matrix)
            # 添加位移标尺
            # cv2.imshow('Dilated Mask with Centroids', dilated_figure)  # 新增独立窗口
            gx_interp, gy_interp = demark(gx, gy, self.dilated_mm)

        else:
            gx_interp, gy_interp = gx, gy # 深度梯度

        # nz = np.sqrt(1 - nx ** 2 - ny ** 2)
        boundary = np.zeros((imgh, imgw))

        # 计算连通区域及其质心
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(self.dilated_mm.astype(np.uint8))
        min_area = 20  # 最小有效标记面积（像素），根据实际情况调整
        centers = []
        valid_centroids = []

        for i in range(1, num_labels):  # 跳过背景（标签0）
            x, y = centroids[i]
            valid_centroids.append([x, y])
            center = (int(round(x)), int(round(y)))
            centers.append(center)
            # 绘制红色圆形标记（半径2像素）
            cv2.circle(dilated_figure, center, 2, (0, 0, 255), -1)
        valid_centroids = np.array(valid_centroids)  # 转换为numpy数组

        if len(valid_centroids) != 63:
            print(f"检测到{len(valid_centroids)}个质心，应为63个。跳过排序。")
        else:
            # 使用K-means按y坐标分为7行
            y_coords = valid_centroids[:, 1].reshape(-1, 1)
            kmeans = self.KMeans(n_clusters=7, random_state=0, n_init=10).fit(y_coords)
            cluster_labels = kmeans.labels_
            # 按行收集并排序x坐标
            sorted_rows = []
            for i in range(7):
                row_points = valid_centroids[cluster_labels == i]
                row_points = row_points[row_points[:, 0].argsort()]  # 按x排序
                sorted_rows.append(row_points)
            # 按行的平均y坐标排序
            cluster_centers = kmeans.cluster_centers_.flatten()
            row_order = np.argsort(cluster_centers)
            sorted_matrix = [sorted_rows[i] for i in row_order]
            # 转换为7x9的矩阵
            self.current_marker_matrix = np.array(sorted_matrix).reshape(7, 9, 2)
            # 计算坐标变化量
            if self.initial_marker_matrix is None:
                self.initial_marker_matrix = self.current_marker_matrix.copy()
            self.current_delta = self.current_marker_matrix - self.initial_marker_matrix
            # 计算相对于基准的位移
            self.current_delta = self.current_marker_matrix - self.initial_marker_matrix
            ##############################################################################################新增便偏移可视化
            # 可视化设置
            img_h, img_w = dilated_figure.shape[:2]
            GRID_COLOR = (100, 100, 100)
            LINE_THICKNESS = 2
            SCALE_FACTOR = 5  # 位移放大系数

            # 绘制网格背景
            for i in range(1, 7):
                cv2.line(dilated_figure, (0, int(i * img_h / 7)), (img_w, int(i * img_h / 7)), GRID_COLOR, 1)
            for j in range(1, 9):
                cv2.line(dilated_figure, (int(j * img_w / 9), 0), (int(j * img_w / 9), img_h), GRID_COLOR, 1)

            # 绘制所有标记点
            for i in range(7):
                for j in range(9):
                    # 获取基准坐标（交换x,y）
                    base_x = int(self.initial_marker_matrix[i, j][0])
                    base_y = int(self.initial_marker_matrix[i, j][1])
                    base_pos = (max(0, min(base_x, img_w - 1)),
                                max(0, min(base_y, img_h - 1)))

                    # 计算位移量（注意坐标轴对应）
                    dx = int(self.current_delta[i, j][0] * SCALE_FACTOR)  # delta的x对应原始y轴
                    dy = int(self.current_delta[i, j][1] * SCALE_FACTOR)  # delta的y对应原始x轴

                    # 计算终点坐标
                    end_x = base_pos[0] + dx
                    end_y = base_pos[1] + dy
                    end_pos = (max(0, min(end_x, img_w - 1)),
                               max(0, min(end_y, img_h - 1)))

                    # 绘制箭头
                    cv2.arrowedLine(dilated_figure, base_pos, end_pos,
                                    (255, 255, 0), LINE_THICKNESS, tipLength=0.2)

                    # 绘制基准点
                    cv2.circle(dilated_figure, base_pos, 3, (0, 255, 255), -1)

                    # 显示位移量
                    disp = np.sqrt(dx ** 2 + dy ** 2) / SCALE_FACTOR  # 实际位移量
                    if disp > 2:  # 仅显示显著位移
                        cv2.putText(dilated_figure, f"{disp:.1f}px",
                                    (end_pos[0] + 5, end_pos[1]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


            # 显示标尺信息
            cv2.putText(dilated_figure,
                        f"Displacement Scale: x{SCALE_FACTOR} (Threshold:2px)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow('Marker Tracking', dilated_figure)
            ########################################################################################################
            # print(self.current_marker_matrix)
            # 合并坐标与变化量 (7,9,4)
            # self.marker_matrix = np.concatenate([
            #     self.current_marker_matrix,
            #     self.current_delta
            # ], axis=-1)


        dm = poisson_dct_neumaan(gx_interp, gy_interp)
        dm = np.reshape(dm, (imgh, imgw))
        #print(dm.shape)
        # cv2.imshow('dm',dm)

        ''' remove initial zero depth '''
        if self.dm_zero_counter < 50:
            self.dm_zero += dm
            print ('zeroing depth. do not touch the gel!')
            if self.dm_zero_counter == 49:
                self.dm_zero /= self.dm_zero_counter
        if self.dm_zero_counter == 50:
            print ('Ok to touch me now!')
        self.dm_zero_counter += 1
        dm = dm - self.dm_zero
        # print(dm.min(), dm.max())

        ''' ENTIRE MASK. GPU OPTIMIZED VARIABLES. '''
        # if np.where(cm)[0].shape[0] != 0:
        ### Run things through NN. FAST!!??
        # pxpos = np.vstack(np.where(cm)).T
        # features = np.zeros((len(pxpos), 5))
        # get_features(img, pxpos, features, imgw, imgh)
        # features = torch.from_numpy(features).float().to(device)
        # with torch.no_grad():
        #     net.eval()
        #     out = net(features)
        # # Create gradient images and do reconstuction
        # gradx = torch.from_numpy(np.zeros_like(cm, dtype=np.float32)).to(device)
        # grady = torch.from_numpy(np.zeros_like(cm, dtype=np.float32)).to(device)
        # grady[pxpos[:, 0], pxpos[:, 1]] = out[:, 0]
        # gradx[pxpos[:, 0], pxpos[:, 1]] = out[:, 1]
        # # dm = poisson_reconstruct_gpu(grady, gradx, denom).cpu().numpy()
        # dm = cv2.resize(poisson_reconstruct(grady, gradx, denom).cpu().numpy(), (640, 480))
        # dm = cv2.resize(dm, (imgw, imgh))
        # # dm = np.clip(dm / img.max(), 0, 1)
        # # dm = 255 * dm
        # # dm = dm.astype(np.uint8)

        ''' normalize gradients for plotting purpose '''
        #print(gx.min(), gx.max(), gy.min(), gy.max())
        gx = (gx - gx.min()) / (gx.max() - gx.min())
        gy = (gy - gy.min()) / (gy.max() - gy.min())
        gx_interp = (gx_interp - gx_interp.min()) / (gx_interp.max() - gx_interp.min())
        gy_interp = (gy_interp - gy_interp.min()) / (gy_interp.max() - gy_interp.min())

        return dm


class Visualize3D:
    def __init__(self, n, m, save_path, mmpp):
        self.n, self.m = n, m
        self.init_open3D()
        self.cnt = 212
        self.save_path = save_path

        self.frame_count = 0   #  初始化帧计数器
        self.start_time = time.time()  #  初始化起始时间
        pass

    def init_open3D(self):
        x = np.arange(self.n) # * self.mmpp           #################################未使用，待查看
        y = np.arange(self.m) # * self.mmpp
        self.X, self.Y = np.meshgrid(x,y) # X/Y为二维网格坐标矩阵
        Z = np.sin(self.X)

        self.points = np.zeros([self.n * self.m, 3]) # 每个点存储(x,y,z)
        self.points[:, 0] = np.ndarray.flatten(self.X) #/ self.m X坐标赋值给第0列
        self.points[:, 1] = np.ndarray.flatten(self.Y) #/ self.n Y坐标赋值给第1列

        self.depth2points(Z) # 更新Z坐标
        # 创建Open3D点云对象
        self.pcd = open3d.geometry.PointCloud()
        self.pcd.points = open3d.utility.Vector3dVector(self.points)
        # self.pcd.colors = Vector3dVector(np.zeros([self.n, self.m, 3]))

        # 初始化可视化窗口
        self.vis = open3d.visualization.Visualizer()
        self.vis.create_window(width=640, height=480) # 固定窗口尺寸
        self.vis.add_geometry(self.pcd) # 添加点云到视图

    def depth2points(self, Z):
        self.points[:, 2] = np.ndarray.flatten(Z)

    def update(self, Z ,markermask):##def update(self, Z) #主函数调用
        self.depth2points(Z)
        dx, dy = np.gradient(Z) # 只取用横向梯度？
        dx, dy = dx * 0.5, dy * 0.5

        np_colors = dx + 0.5
        np_colors[np_colors < 0] = 0
        np_colors[np_colors > 1] = 1

        # # 标记点覆盖为黑色 ##################################################### 修改开始
        # if markermask is not None:
        #     np_colors[markermask == 255] = 0  # 黑色对应的颜色值为0
        # ##################################################################### 修改完毕

        np_colors = np.ndarray.flatten(np_colors)
        colors = np.zeros([self.points.shape[0], 3])
        for _ in range(3): colors[:,_]  = np_colors

        self.pcd.points = open3d.utility.Vector3dVector(self.points)
        self.pcd.colors = open3d.utility.Vector3dVector(colors)

        self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()

        self.frame_count += 1
        if self.frame_count % 30 == 0:  # 每30帧计算一次
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed
            print(f"FPS: {fps:.2f}")
            self.frame_count = 0
            self.start_time = time.time()
        # =============================
        #### SAVE POINT CLOUD TO A FILE
        if self.save_path != '':
            open3d.io.write_point_cloud(self.save_path + "/pc_{}.pcd".format(self.cnt), self.pcd)
        self.cnt += 1

    def save_pointcloud(self):
        open3d.io.write_point_cloud(self.save_path + "pc_{}.pcd".format(self.cnt), self.pcd)





