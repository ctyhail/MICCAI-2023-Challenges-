# 加载一些基础的库
import os
import cv2
import albumentations as A
import torchvision.transforms as T
from torch.utils.data import Dataset


totensor = T.Compose(
    [
        T.ToTensor(),
        # T.Normalize([0.5] * 3, [0.5] * 3),
    ]
)
# totensor = T.ToTensor()
transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),#垂直翻转
        A.VerticalFlip(p=0.5),#水平翻转
        A.OneOf(
            [
                A.RandomGamma(p=1),#随机伽马变换
                A.RandomBrightnessContrast(p=1),#随机亮度
                A.Blur(p=1),#模糊
                A.OpticalDistortion(p=1),#光学畸变
            ],
            p=0.5,
        ),
        A.OneOf(
            [
                A.ElasticTransform(p=1),#弹性变换
                A.GridDistortion(p=1),#网格失真
                A.MotionBlur(p=1),#运动模糊
                A.HueSaturationValue(p=1),#色调，饱和度值随机变化
            ],
            p=0.5,
        ),
    ]
)

"""
class MyDataset(Dataset):
    def __init__(self, path):
        self.mode = "train" if "mask" in os.listdir(path) else "test"  # 表示训练模式
        self.path = path  # 图片路径
        dirlist = os.listdir(path + "image/")  # 图片的名称
        self.name = [n for n in dirlist if n[-3:] == "png"]  # 只读取图片

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):  # 获取数据的处理方式
        name = self.name[index]
        # 读取原始图片和标签
        if self.mode == "train":  # 训练模式
            ori_img = cv2.imread(self.path + "image/" + name)  # 原始图片
            lb_img = cv2.imread(self.path + "mask/" + name)  # 标签图片
            ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)  # 转为RGB三通道图
            lb_img = cv2.cvtColor(lb_img, cv2.COLOR_BGR2GRAY)  # 掩膜转为灰度图
            transformed = transform(image=ori_img, mask=lb_img)
            return totensor(transformed["image"]), totensor(transformed["mask"])

        if self.mode == "test":  # 测试模式
            ori_img = cv2.imread(self.path + "image/" + name)  # 原始图片
            ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)  # 转为RGB三通道图
            return totensor(ori_img)
"""



def get_data(image_path):
    img = cv2.imread(image_path)
    img =  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return totensor(img)

# a = get_data("train")[5]
# print(type(a[0]), type(a[1]), a[0].shape, a[1].shape)
