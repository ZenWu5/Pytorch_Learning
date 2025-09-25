from torch.utils.data import Dataset
import cv2
import os

class ImageDataset(Dataset):
    def __init__(self, image_root_path: str, dir_label: str, transform=None):
        self.image_root_path = image_root_path
        self.dir_label = dir_label
        self.images_paths = os.listdir(os.path.join(self.image_root_path, self.dir_label))
        self.transform = transform

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.image_root_path, self.dir_label, self.images_paths[idx]))
        label = self.dir_label
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

if __name__ == "__main__":
    # Example usage
    image_root_path = "hymenoptera_data/hymenoptera_data/train"
    label = "ants"
    
    dataset = ImageDataset(image_root_path, label)
    # for img, lbl in dataset:
        # print(f"Image shape: {img.shape}, Label: {lbl}")
    print(f"Data type: {type(dataset)}, Length: {dataset.__len__()}")