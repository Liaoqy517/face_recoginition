#自新冠病毒流行以后，出门戴口罩成为了一个重点话题。为了响应这个号召，一些网民也为自己头像上的人物戴上了“口罩”。由此，本人想实现一个识别人脸头像图片并加上口罩的功能。整体的构思主要分为三步：
#首先，到网上找到一张口罩的图片。最好是找一张不带背景的（否则影响美观），实在不行，就用抠图软件去除背景。
#其次，检测人脸的一些关键点（网上查阅了一下，dlib自带人脸识别68个特征点检测数据库 shape_predictor_68_face_landmarks.dat），找到几个口罩关键点（鼻梁、下巴、耳朵附近）标注出来
#最后，则将实际的人脸图片中的关键点找出，得到口罩的宽度和高度，调整口罩图片的大小，之后再叠加

import cv2
import numpy as np 
import dlib
 
#识别人脸关键点函数
#变量img：要读取的图片
def key_points(img):
	points_key = []
	PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat" #网上搜索下载
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(PREDICTOR_PATH)
	rects = detector(img, 1)
 
	for i in range(len(rects)):
		landmarks = np.matrix([[p.x, p.y] for p in predictor(img, rects[i]).parts()])
		print(landmarks)
		for idx, point in enumerate(landmarks): 
			print(idx)
			pos = (point[0, 0], point[0, 1])
			print(pos)
			if idx in [2, 8, 14, 28]:
				points_key.append(pos)
				# cv2.circle(img, pos, 2, (255, 0, 0))
 
	return(points_key)
 
#戴口罩的函数
#变量：mask_img 口罩图片；face_img 人脸图片
def wear_mask(mask_img, face_img):
	h_mask, w_mask = mask_img.shape[:2]   # 高，宽
	gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
	face_keys = key_points(gray)
	left = face_keys[0][0]
	jaw = face_keys[1][1]
	right = face_keys[2][0]
	nose = face_keys[3][1]
	w_mouth = right - left
	h_mouth = jaw - nose
	mask_img = cv2.resize(mask_img, (w_mouth, h_mouth))
	
  #分成三个通道
	mask_channels = cv2.split(mask_img)
	face_channels = cv2.split(face_img)
	b, g, r, a = cv2.split(mask_img)
	ans_img = face_img.copy()
	print(nose, nose+h_mouth, left, left+w_mouth)
	for c in range(0, 3):
		face_channels[c] = np.array(face_channels[c], dtype=np.uint8)
		k = np.uint8((255.0-a)/255.0)
		face_channels[c][nose:nose+h_mouth, left:left+w_mouth] = face_channels[c][nose:nose+h_mouth, left:left+w_mouth]*k
		mask_channels[c] *= np.array(a/255, dtype=np.uint8)
		face_channels[c][nose:nose+h_mouth, left:left+w_mouth] += np.array(mask_channels[c], dtype=np.uint8)
	ans = cv2.merge(face_channels)
 #最后合并在一起 
	return ans
 
face_img = cv2.imread("lyh.jpg")
mask_img = cv2.imread("mask2.png", -1)
ans_img = wear_mask(mask_img, face_img)
 
cv2.imwrite("ans3.jpg", ans_img)
cv2.imshow("ans", ans_img)
cv2.waitKey(0)
