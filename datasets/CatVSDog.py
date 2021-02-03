from torchvision import transforms as T
from torch.utils import data
import os
from PIL import Image


class DogCat(data.Dataset):
    def __init__(self, root, trainform=None, train=True, test=False):
        super(DogCat, self).__init__()
        # data/train/cat.100.jpg
        # data/test/100.jpg
        self.imgs = [os.path.join(root, img) for img in os.listdir(root)]
        len_num = len(self.imgs)
        if train:
            imgs = sorted(self.imgs, key=lambda x: int(x.split('.')[-2]))
            self.imgs = imgs[:int(0.7 * len_num)]
        elif test:
            imgs = sorted(self.imgs, key=lambda x: int(x.split('/')[-1]))
            self.imgs = imgs
        else: # val
            imgs = sorted(self.imgs, key=lambda x: int(x.split('.')[-2]))
            self.imgs = imgs[int(0.7 * len_num):]

        if trainform is None:
            self.normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])

            self.transform = T.Compose([T.Resize(256),
                                   T.CenterCrop(256),
                                   T.ToTensor(),
                                   self.normalize])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        # data/train/cat.100.jpg
        img_path = self.imgs[item]
        # print(img_path)
        # data/train/cat
        label = 1 if 'dog' in img_path.split('/')[-1] else 0

        data = Image.open(img_path)
        data = self.transform(data)

        return data, label





if __name__ == '__main__':
    dataset = DogCat('../data/train')
    print(dataset.__len__())
    train_dataset = data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=4)
    print(len(train_dataset))
    # for i,(img, label) in enumerate(train_dataset):
    #     print(img.shape)
    #     print(label)