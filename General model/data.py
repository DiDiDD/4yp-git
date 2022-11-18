import os

import imageio
import numpy
import cv2
from tqdm import tqdm
from glob import glob
from albumentations import HorizontalFlip, VerticalFlip, Rotate

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path):
    train_x = sorted(glob(os.path.join(path,"Images", '*.jpg')))
    train_y = sorted(glob(os.path.join(path,"G+N", '*.bmp')))
    test_x = sorted(glob(os.path.join(path,"Images", '*.jpg')))
    test_y = sorted(glob(os.path.join(path,"G+N", '*.bmp')))

    return (train_x, train_y), (test_x,test_y)

def augment_data(images, masks,save_path, augment=True):
    size = (1634,1634)

    for index, (x,y) in tqdm(enumerate(zip(images,masks)),total=len(images)):

        '''Extracting the name'''
        name = x.split('/')[-1].split('.')[0]
        print(name)

        '''reading image and masks'''
        x = cv2.imread(x, cv2.IMREAD_COLORS)
        y = imageio.mimread(y)[0]
        print(x.shape, y.shape)

        break

if __name__ == "__main__":
    data_path = "/Users/diwan/Desktop/Data1"
    (train_x, train_y), (test_x, test_y) = load_data(data_path)
    create_dir('new_data/train/image')
    create_dir('new_data/train/mask')
    create_dir('new_data/test/image')
    create_dir('new_data/test/mask')

