import cv2
import glob
import os
from datetime import datetime

#定义一个函数：取出视频每一帧，存储在指定文件夹中
#变量path ： 要存入的文件夹的路径

def video_to_frames(path):
    """
    输入：path(视频文件的路径)
    """
    # VideoCapture视频读取类
    videoCapture = cv2.VideoCapture()
    videoCapture.open(path)
    # 帧率
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    # 总帧数
    frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    print("fps=", int(fps), "frames=", int(frames))

    for i in range(int(frames)):
        ret, frame = videoCapture.read()
        cv2.imwrite("C:\\dataspy\\images3\\frames%d.jpg" % (i), frame)
    return


if __name__ == '__main__':
    t1 = datetime.now()
    video_to_frames("C:\\dataspy\\lqy.mp4")
    t2 = datetime.now()
    #print("Time cost = ", (t2 - t1))
    #print("SUCCEED !!!")

