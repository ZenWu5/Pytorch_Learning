from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("Scripts/P9-12. Transforms的使用/logs")
img_path = "Dataset/train/ants/0013035.jpg"
img = Image.open(img_path)  # Open image with PIL

# ToTensor
totensor = transforms.ToTensor()
img_tensor = totensor(img)
writer.add_image("ToTensor", img_tensor)

# Normalize
trans_mean=[0.485, 0.456, 0.406]
trans_std=[0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=trans_mean, std=trans_std)
img_normalized = normalize(img_tensor)
writer.add_image("Normalize", img_normalized)

# Resize
resize = transforms.Resize((300, 300)) # 没有裁切，直接比例缩放到指定大小
img_resized = resize(img)
img_resized_tensor = totensor(img_resized)
writer.add_image("Resize", img_resized_tensor)

# Compose
compose = transforms.Compose([transforms.Resize((300, 300)), transforms.ToTensor(), transforms.Normalize(mean=trans_mean, std=trans_std)])
img_composed = compose(img)
writer.add_image("Compose", img_composed)

# Invert Normalize
inv_normalize = transforms.Normalize(
    mean=[-m/s for m, s in zip(trans_mean, trans_std)],
    std=[1/s for s in trans_std]
)
img_inv_normalized = inv_normalize(img_composed)
writer.add_image("Invert_Normalize", img_inv_normalized)

# RandomCrop
random_crop = transforms.RandomCrop((200, 200))
img_random_cropped = random_crop(img_tensor)
writer.add_image("RandomCrop", img_random_cropped)

# CenterCrop
center_crop = transforms.CenterCrop((200, 200))
img_center_cropped = center_crop(img_tensor)
writer.add_image("CenterCrop", img_center_cropped)

# Close the writer
writer.close()

# 在终端运行命令: tensorboard --logdir="Scripts/P9-12. Transforms的使用/logs"
