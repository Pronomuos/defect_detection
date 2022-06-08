import random
from PIL import Image

from torch.utils.data import Dataset


class SiameseDataset(Dataset):
    def __init__(self, imageFolderDataset, transform=None):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.class_ids = [self.imageFolderDataset.class_to_idx[class_name] for class_name in
                          self.imageFolderDataset.classes]

    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)

        while True:
            img1_tuple = random.choice(self.imageFolderDataset.imgs)
            if img0_tuple[1] == img1_tuple[1]:
                break

        while True:
            img2_tuple = random.choice(self.imageFolderDataset.imgs)
            if img0_tuple[1] != img2_tuple[1]:
                break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img2 = Image.open(img2_tuple[0])

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img0, img1, img2

    def get_random_imgs(self):
        imgs = []
        for class_id in self.class_ids:
            while True:
                img = random.choice(self.imageFolderDataset.imgs)
                if img[1] == class_id:
                    imgs.append(img)
                    break

        for i in range(len(imgs)):
            imgs[i] = [Image.open(imgs[i][0]), imgs[i][1]]

        if self.transform is not None:
            for i in range(len(imgs)):
                imgs[i][0] = self.transform(imgs[i][0])

        return imgs

    def get_image(self, index):
        img = self.imageFolderDataset.imgs[index]

        img = [Image.open(img[0]), img[1]]

        if self.transform is not None:
            img[0] = self.transform(img[0])

        return img

    def __len__(self):
        return len(self.imageFolderDataset.imgs)