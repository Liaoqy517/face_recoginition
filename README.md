深度学习算法安全性研究

本项目是2021年1-3月份期间所做的深度学习算法安全性研究（主题是人脸识别），你可以使用Python和命令行工具提取、识别、操作人脸。

本项目的人脸识别是基于业内领先的C++开源库 dlib中的深度学习模型，用Labeled Faces in the Wild人脸数据集进行测试，有高达99.38%的准确率。但对小孩和亚洲人脸的识别准确率尚待提升。

Labeled Faces in the Wild是美国麻省大学安姆斯特分校（University of Massachusetts Amherst)制作的人脸数据集，该数据集包含了从网络收集的13,000多张面部图像。

本项目部分代码所使用的简易的face_recognition命令行工具，你可以用它处理整个文件夹里的图片。

本项目实现了简单的人脸检测和识别功能；除此之外，实现了简易的戴口罩、口红试色的小程序。


安装
环境配置
Python 3.3+ 
Windows
anaconda3
CUDA
CMake
Visual Studio2019

使用方法
命令行界面
当你安装好了本项目，你可以使用两种命令行工具：

face_recognition - 在单张图片或一个图片文件夹中认出是谁的脸。
face_detection - 在单张图片或一个图片文件夹中定位人脸位置。
face_recognition 命令行工具
face_recognition命令行工具可以在单张图片或一个图片文件夹中认出是谁的脸。

首先，你得有一个你已经知道名字的人脸图片文件夹，一个人一张图，图片的文件名即为对应的人的名字：

然后，你需要有一个视频，视频中是待检测的人脸：

然后，你在命令行中切换到这两个文件夹所在路径，然后使用splash.py，传入这两个图片文件夹，然后就会框出未知人脸及其名字。

输出结果的每一行对应着图片中的一张脸，图片名字和对应人脸识别结果用逗号分开。

如果结果输出了unknown_person，那么代表这张脸没有对应上已知人脸图片文件夹中的任何一个人。


调整人脸识别的容错率
如果一张脸识别出不止一个结果，或者人脸框时有时无，那么这意味着他和其他人长的太像了（本项目对于小孩和亚洲人的人脸识别准确率有待提升），或是容错率过高，有那么一瞬间无法识别人脸。你可以把容错率调低一些，使识别结果更加严格。

通过传入参数 --tolerance 来实现这个功能，默认的容错率是0.6，容错率越低，识别越严格准确。

$ face_recognition --tolerance 0.54 ./pictures_of_people_i_know/ ./unknown_pictures/

/unknown_pictures/unknown.jpg,Barack Obama
/face_recognition_test/unknown_pictures/unknown.jpg,unknown_person
如果你想看人脸匹配的具体数值，可以传入参数 --show-distance true：

$ face_recognition --show-distance true ./pictures_of_people_i_know/ ./unknown_pictures/

/unknown_pictures/unknown.jpg,Barack Obama,0.378542298956785
/face_recognition_test/unknown_pictures/unknown.jpg,unknown_person,None


在图片中定位人脸的位置
import face_recognition

image = face_recognition.load_image_file("my_picture.jpg")
face_locations = face_recognition.face_locations(image)

你也可以使用深度学习模型达到更加精准的人脸定位。

注意：这种方法需要GPU加速（通过英伟达显卡的CUDA库驱动），你在编译安装dlib的时候也需要开启CUDA支持。

import face_recognition

image = face_recognition.load_image_file("my_picture.jpg")
face_locations = face_recognition.face_locations(image, model="cnn")

如果你有很多图片需要识别，同时又有GPU，那么你可以参考这个例子：案例：使用卷积神经网络深度学习模型批量识别图片中的人脸.

在这里，同样提供了提取视频每一帧的代码，详情请看each.py

其他部分功能如下：

识别单张图片中人脸的关键点
import face_recognition

image = face_recognition.load_image_file("my_picture.jpg")
face_landmarks_list = face_recognition.face_landmarks(image)


识别图片中的人是谁
import face_recognition

picture_of_me = face_recognition.load_image_file("me.jpg")
my_face_encoding = face_recognition.face_encodings(picture_of_me)[0]

unknown_picture = face_recognition.load_image_file("unknown.jpg")
unknown_face_encoding = face_recognition.face_encodings(unknown_picture)[0]

results = face_recognition.compare_faces([my_face_encoding], unknown_face_encoding)

if results[0] == True:
    print("It's a picture of me!")
else:
    print("It's not a picture of me!")
    
实际应用：
1.戴口罩
自新冠病毒流行以后，出门戴口罩成为了一个重点话题。为了响应这个号召，一些网民也为自己头像上的人物戴上了“口罩”。由此，本人想实现一个识别人脸头像图片并加上口罩的功能。整体的构思主要分为三步：

首先，到网上找到一张口罩的图片。最好是找一张不带背景的（否则影响美观），实在不行，就用抠图软件去除背景。

其次，检测人脸的一些关键点（希望能用face_recognition实现），找到几个口罩关键点（鼻梁、下巴、耳朵附近）标注出来.

最后，则将实际的人脸图片中的关键点找出，得到口罩的宽度和高度，调整口罩图片的大小，之后再叠加。

2.口红试色
本功能毫无疑问可以帮助女生在家里就可以完成口红的试色。大体的步骤依然是三步：

首先，找到一张女生的图片。

第二步，这次检测人脸嘴唇部分的关键点。对于上下嘴唇，分别按顺时针方向的顺序组成一个多边形，为的是后面上色。

最后，多输入几种颜色，并在多边形中填充颜色，完成口红试色。

额外思考：第一点，是角度的问题（侧脸、抬头、低头），对于不同情况要做不同的讨论（关键点的选取、多边形的大小）；

以上功能的实现，请见mask.py和lip.py

警告说明
本项目的人脸识别模型是基于成年人的，在孩子身上效果可能一般。如果图片中有孩子的话，建议把临界值设为0.6.
同样，亚洲人脸准确率也不高，建议把临界值设为0.6



鸣谢
非常感谢 Davis King (@nulhom)创建了dlib库，提供了响应的人脸关键点检测和人脸编码相关的模型，你可以查看 blog post这个网页获取更多有关ResNet的信息。
感谢每一个相关Python模块（包括numpy,scipy,scikit-image,pillow等）的贡献者。
感谢 Cookiecutter 和audreyr/cookiecutter-pypackage 项目模板，使得Python的打包方式更容易接受。
