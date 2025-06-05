import sys
import numpy as np
import cv2
import os
import gelsight.gsdevice as gsdevice
import gelsight.gs3drecon as gs3drecon


def main(argv):
    # Set flags
    SAVE_VIDEO_FLAG = False  # 视频保存
    FIND_ROI = False  # 手动选择感兴趣区域
    GPU = True  # GPU加速
    MASK_MARKERS_FLAG = True # 标记点遮罩

    # Path to 3d model
    path = '.'

    # Set the camera resolution
    mmpp = 0.0634  # mini gel 18x24mm at 240x320
    # 毫米/像素比例（18x24mm凝胶区域对应320x240分辨率）
    # the device ID can change after unplugging and changing the usb ports.
    # on linux run, v4l2-ctl --list-devices, in the terminal to get the device ID for camera
    dev = gsdevice.Camera("GelSight Mini") # 设备对象
    net_file_path = 'nnmini.pt' # 神经网络模型文件

    dev.connect() # 连接硬件设备
    # 加载3D重建神经网络
    ''' Load neural network '''
    model_file_path = path
    net_path = os.path.join(model_file_path, net_file_path)
    print('net path = ', net_path)

    if GPU:
        gpuorcpu = "cuda"
    else:
        gpuorcpu = "cpu"

    nn = gs3drecon.Reconstruction3D(dev)
    net = nn.load_nn(net_path, gpuorcpu)

    f0 = dev.get_raw_image()
    roi = (0, 0, f0.shape[1], f0.shape[0])

    if SAVE_VIDEO_FLAG:
        #### Below VideoWriter object will create a frame of above defined The output is stored in 'filename.avi' file.
        file_path = './3dnnlive.mov'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(file_path, fourcc, 60, (f0.shape[1], f0.shape[0]), isColor=True)
        print(f'Saving video to {file_path}')
    # 手动选择ROI
    if FIND_ROI:
        roi = cv2.selectROI(f0)
        roi_cropped = f0[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
        cv2.imshow('ROI', roi_cropped)
        print('Press q in ROI image to continue')
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print('roi = ', roi)
    print('press q on image to exit')

    ''' use this to plot just the 3d '''
    vis3d = gs3drecon.Visualize3D(dev.imgh, dev.imgw, '', mmpp) # 3D可视化对象

    try:
        while dev.while_condition:

            # get the roi image 获取+显示图像
            f1 = dev.get_image()
            bigframe = cv2.resize(f1, (f1.shape[1] * 2, f1.shape[0] * 2))
            # cv2.imshow('Image', bigframe)

            # compute the depth map 深度图计算
            dm = nn.get_depthmap(f1, MASK_MARKERS_FLAG)
            # 3D可视化更新
            ''' Display the results '''
            # 显示膨胀掩膜（放大后的版本仅用于可视化）
            # if nn.dilated_mm is not None:
            #     dilated_display = cv2.resize(nn.dilated_mm.astype(np.uint8) * 255,
            #                                  (nn.dilated_mm.shape[1] * 2, nn.dilated_mm.shape[0] * 2))
            #     # cv2.imshow('Dilated Markers', dilated_display)

            # 3D更新使用原始尺寸的掩膜
            # vis3d.update(dm, None)  # 不传入掩膜
            vis3d.update(dm, nn.dilated_mm)  # 传入掩膜

            # vis3d.update(dm)  ###vis3d.update(dm)
            # 退出机制
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if SAVE_VIDEO_FLAG:
                out.write(f1)

    except KeyboardInterrupt:
        print('Interrupted!')
        dev.stop_video()


if __name__ == "__main__":
    main(sys.argv[1:])
