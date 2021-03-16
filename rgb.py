#这是一个简单的程序，在口红试色功能中可能需要
#其目的是得到相应图片的rgb值

from PIL import Image

im = Image.open('shiyuan.jpg')
width = im.size[0]
height = im.size[1]
im = im.convert('RGB')
array = []
for x in range(width):
    for y in range(height):
        r, g, b = im.getpixel((x,y))
        rgb = (r, g, b)
        array.append(rgb)
print(array)
