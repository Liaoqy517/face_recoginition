#本功能毫无疑问可以帮助女生在家里就可以完成口红的试色。大体的步骤依然是三步：
#首先，找到人俩关键点（仍用上面那个特征点数据库，这次是嘴唇部分）。
#第二步，得到两张图片的rgb值做差得到平移量（人脸图片嘴唇部分、口红颜色），为的是更加逼真（亮暗的变化可以保持） 。
#最后，多输入几种颜色，并填充颜色，完成口红试色。


from collections import deque
import dlib
import numpy as np
import cv2

# 获得人脸矩形的坐标信息
def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return x, y, w, h

# 将包含68个特征的的shape转换为numpy array格式
def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

# 将待检测的image进行resize
def resize(image, width=1200):
    r = width * 1.0 / image.shape[1]
    dim = (width, int(image.shape[0] * r))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized


def feature(image_file, lipstick_color):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    image = cv2.imread(image_file)
    image = resize(image, width=600)
    # print(image.shape)
    # 转为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 灰度图里定位人脸
    rects = detector(gray, 1)
    # shapes存储找到的人脸框，人脸框仅包含四个角数值如frontal_face_detector.png所示。
    shapes = []
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        shape = shape[48:]
        shapes.append(shape)

    # 图片转为hsv形式，色调（H），饱和度（S），亮度（V）
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    for shape in shapes:
        sx, sy = image.shape[0], image.shape[1]
        # 外嘴唇凸包
        hull = cv2.convexHull(shape)
        # 内嘴唇凸包
        hull2 = cv2.convexHull(shape[12:])
        ## 圈出凸包区域
        # cv2.drawContours(image, [hull], -1, (255, 100, 168), -1)
        # cv2.drawContours(image, [hull2], -1, (168, 100, 168), -1)
        for xx in range(sx):
            for yy in range(sy):
                dist = cv2.pointPolygonTest(hull, (xx, yy), False)
                dist_inside = cv2.pointPolygonTest(hull2, (xx, yy), False)
                # 在外嘴唇凸包以内、在内嘴唇凸包以外部分为嘴唇
                if (dist >= 0 and dist_inside < 0):
                    image[yy, xx][0] = lipstick_color[0]
                    image[yy, xx][1] = lipstick_color[1]
                    image[yy, xx][2] += 10

    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    # 高斯滤波是bfs尝试失败来做边缘柔和处理的
    # image = cv2.GaussianBlur(image, (7, 7), 0)
    return image

def update(h,s):
    input_image_path = "shiyuan.jpg"  # 输入图像.jpg
    lipstick_color = [h,s, 0]  # 嘴唇颜色
    image_output = feature(input_image_path, lipstick_color)  # 处理图像
    cv2.imshow("output", image_output)  # 显示
    # cv2.imwrite("process2+" + input_image_path, image_output)  # 保存
    # cv2.waitKey(0)

def nothing(x):
    pass

if __name__=="__main__":
    cv2.namedWindow('output')
    cv2.createTrackbar('H','output',20,50,nothing)
    cv2.createTrackbar('S', 'output', 110, 255, nothing)
    cv2.createTrackbar('on', 'output', 1, 1, nothing)
    while(1):
        h = cv2.getTrackbarPos('H','output')+155
        if(h>180):
            h-=180
        s = cv2.getTrackbarPos('S','output')
        tag = cv2.getTrackbarPos('on','output')
        if s == -1:
            break
        if tag != 0:
            # print(h,s)
            update(h,s)
        k = cv2.waitKey(1) & 0xFF4
        if k == 27:
            break

    cv2.destroyAllWindows()
