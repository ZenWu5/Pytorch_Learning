"""
1. 如何使用 transforms 来处理 tensor 数据类型
2. 为什么要使用 Tensor 数据类型
"""

from torchvision import transforms
from PIL import Image
import numpy as np

# 使用 PIL.Image 打开图片
img_path = 'Dataset/train/ants/0013035.jpg'
img = Image.open(img_path)
print(type(img))  # <class 'PIL.JpegImagePlugin.JpegImageFile'>

# 使用 transforms.ToTensor() 将 PIL.Image 转换为 Tensor
ImgToTensor = transforms.ToTensor() # 实例化
tensor_img = ImgToTensor(img)
print(type(tensor_img))  # <class 'torch.Tensor'>

# 查看 Tensor 的形状和数据类型
print(tensor_img.shape)  # torch.Size([3, 375, 500]) # [C, H, W]
print(tensor_img.dtype)  # torch.float32

# 使用 numpy.ndarray 来表示图片数据
np_img = np.zeros((100, 100, 3)) # 由于是适配opencv所以输入为HWC，ToTensor会自动转换为CHW（即转置操作）
tensor_np_img = ImgToTensor(np_img)
print(type(tensor_np_img))  # <class 'torch.Tensor'>
print(tensor_np_img.shape)  # torch.Size([3, 100, 100]) 
