本项目用于 MICCAI 2023 Challenges ：STS-基于2D 全景图像的牙齿分割任务 比赛

[MICCAI 2023 Challenges ：STS-基于2D 全景图像的牙齿分割任务_学习赛_天池大赛-阿里云天池的团队 (aliyun.com)](https://tianchi.aliyun.com/competition/entrance/532086)

## 成绩

初赛-长期打榜：0.9530	排名：5/175

## 训练策略

- 训练方式：全监督
- 架构：deeplabv3+，unet++， unet
- backbone：resnet50, efficientnet_b5, se_resnext50_32x4d 
- optim：AdamW和Adam
- 损失函数：dice_loss，bce_loss，bce_dice_loss
- tricks：kfold_train，数据增强，TTA，学习率衰减，数据后处理

目前最终且最佳方案为resnet50，deeplabv3+，AdamW，bce_dice_loss