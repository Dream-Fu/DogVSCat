from torch.utils import data
import os

# 读取图片名,标签 并 拼接成完成图片路径
def get_image(txt_path, file_path):
    imgs = []
    label = []
    with open(txt_path, 'r') as lines:
        for line in lines:
            # line ../data/
            # emotion/2smile/4353smile.jpg 2
            img_path = os.path.join(file_path, line[:-3])
            imgs.append(img_path)
            label.append(line[-1])
    return imgs, label

if __name__ == '__main__':
    images, label = get_image('../data/all_shuffle_train.txt', '../data/train')
    print(len(label))
    print(images[:2])