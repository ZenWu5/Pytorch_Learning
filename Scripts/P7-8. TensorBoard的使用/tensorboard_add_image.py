from torch.utils.tensorboard import SummaryWriter # 需要先安装tensorboard, pip install tensorboards
from torchvision.transforms import ToTensor
import numpy as np
from PIL import Image
import cv2

# 创建SummaryWriter对象，指定日志保存路径
writer = SummaryWriter("Scripts/P7-8. TensorBoard的使用/logs/add_image")

# 添加图片
ant_image_path = "Dataset/train/ants/0013035.jpg"
bee_image_path = "Dataset/train/bees/16838648_415acd9e3f.jpg"

# 使用OpenCV加载图片，注意OpenCV加载的图片格式为HWC且通道顺序为BGR(Blue,Green,Red)，需要转换为RGB格式
ant_image = cv2.imread(ant_image_path)
ant_image = cv2.cvtColor(ant_image, cv2.COLOR_BGR2RGB) 

# 使用PIL加载图片，PIL加载的图片格式为HWC且通道顺序为RGB
bee_image = Image.open(bee_image_path)
bee_image = ToTensor()(bee_image)  # 转换为CHW格式的Tensor

# 也可以直接传入numpy数组，此处示例为CHW格式
np_img = np.zeros((3, 100, 100))
np_img[0] = np.arange(0, 10000).reshape(100, 100) / 10000
np_img[1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000

# 使用writer.add_image()方法添加图片，注意指定dataformats参数默认为'CHW'
# 如果传入的图片格式为HWC，则需要指定dataformats='HWC'
writer.add_image("ant", ant_image, 0, dataformats='HWC')
writer.add_image("bee", bee_image, 1)
writer.add_image("numpy_CHW", np_img, 2)

"""
SummaryWriter.add_image() 方法参数说明:
    tag (str): 数据标识符
    img_tensor (torch.Tensor, numpy.ndarray, 或 string/blobname): 图像数据
    global_step (int): 记录的全局步数值
    walltime (float): 可选，覆盖默认的 walltime（time.time()），
      表示事件发生的时间（自 epoch 起的秒数）
    dataformats (str): 图像数据格式的规范，例如 CHW, HWC, HW, WH 等，此处的默认值为 'CHW'
    此处的字母C, H, W 分别表示通道数、高度、宽度,
    例如，假设 img_tensor 的形状为 (3, 100, 100)，则 dataformats 应设置为 'CHW'；
    如果 img_tensor 的形状为 (100, 100, 3)，则 dataformats 应设置为 'HWC'；
"""
writer.close()

# 在终端运行命令: tensorboard --logdir="Scripts/P7-8. TensorBoard的使用/logs/add_image"
# 要更换端口使用 --port参数，示例： tensorboard --logdir="Scripts/P7-8. TensorBoard的使用/logs/add_image" --port=6666
# 然后打开浏览器访问 http://localhost:6666/  查看结果，默认端口为6006