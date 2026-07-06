import os
from PIL import Image
import torch
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((84, 84)),
    transforms.ToTensor()
])

class CUBDataset:
    def __init__(self, root):
        self.data = []
        self.labels = []

        image_dir = os.path.join(root, "images")
        self.class_to_idx = {}

        idx = 0

        for cls in sorted(os.listdir(image_dir)):
            cls_path = os.path.join(image_dir, cls)
            if not os.path.isdir(cls_path):
                continue

            self.class_to_idx[cls] = idx

            for img in os.listdir(cls_path):
                try:
                    img_path = os.path.join(cls_path, img)
                    image = Image.open(img_path).convert("RGB")
                    self.data.append(transform(image))
                    self.labels.append(idx)
                except:
                    continue

            idx += 1

        self.data = torch.stack(self.data)
        self.labels = torch.tensor(self.labels)

    def get_data(self):
        return self.data, self.labels