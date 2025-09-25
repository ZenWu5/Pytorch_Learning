from torch.utils.data import Dataset
import cv2
import os

# 自定义的数据集类，继承自torch.utils.data.Dataset
class ImageDataset(Dataset): 
    def __init__(self, image_root_path: str, dir_label: str):
        self.image_root_path = image_root_path # 图像根目录
        self.dir_label = dir_label # 图像子目录（标签）
        self.images_paths = os.listdir(os.path.join(self.image_root_path, self.dir_label)) # 获取图像文件列表

    def __len__(self): # 返回数据集的大小
        return len(self.images_paths)

    def __getitem__(self, idx): # 根据索引获取图像和标签
        image = cv2.imread(os.path.join(self.image_root_path, self.dir_label, self.images_paths[idx]))
        label = self.dir_label
        
        return image, label

# 使用示例
if __name__ == "__main__":
    image_root_path = "Dataset/hymenoptera_data/train" # 相对路径
    label = "ants"
    
    dataset = ImageDataset(image_root_path, label)
    # for img, lbl in dataset: # 遍历数据集
        # print(f"Image shape: {img.shape}, Label: {lbl}")
    print(f"Data type: {type(dataset)}, Length: {dataset.__len__()}") # 输出数据集类型和长度，这里可以看到使用opencv读取的图像是numpy数组