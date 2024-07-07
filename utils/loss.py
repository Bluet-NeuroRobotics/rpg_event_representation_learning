import torch

# 定义交叉熵损失函数和准确度的计算
def cross_entropy_loss_and_accuracy(prediction, target):
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    loss = cross_entropy_loss(prediction, target)
    accuracy = (prediction.argmax(1) == target).float().mean() # 预测值中数值最大的并且预测的值和目标值相等，转换成float数值，然后求平均
    return loss, accuracy