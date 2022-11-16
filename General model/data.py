import os
from skimage import io
from torch.utils.data import Dataset



class dataload():
    def __init__(self,root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform


    def __len__(self):
        return len(self)

    def __getitem__(self, index):
        img_path  = os.path.join
        image = io.read(img_path)

        if self.transform:
            image = self.transform(image)
