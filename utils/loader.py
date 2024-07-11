import torch
import numpy as np

from torch.utils.data.dataloader import default_collate


class Loader:
    """
    返回的是事件数组和标签数组
    """
    def __init__(self, dataset, flags, device):
        self.device = device
        split_indices = list(range(len(dataset))) # 创建一个拆分分割的索引
        sampler = torch.utils.data.sampler.SubsetRandomSampler(split_indices) #根据索引随机采样
        self.loader = torch.utils.data.DataLoader(dataset, batch_size=flags.batch_size, sampler=sampler,
                                             num_workers=flags.num_workers, pin_memory=flags.pin_memory,
                                             collate_fn=collate_events)

    def __iter__(self):
        for data in self.loader:
            data = [d.to(self.device) for d in data]
            yield data

    def __len__(self):
        return len(self.loader)

# 整理事件数据，数据格式[x,y,t,p]
def collate_events(data):
    labels = [] # 标签数组
    events = [] # 事件数组
    # print("data len", len(data)) data length是4
    for i, d in enumerate(data): # 遍历数据样本
        labels.append(d[1]) # data的第二维是labels
        ev = np.concatenate([d[0], i*np.ones((len(d[0]),1), dtype=np.float32)],1) # 处理np数组的第一个维度，
        # 每一个event都拼接一个和d[0]维度一样的index[0,1,2,3]的矩阵
        # 假设数据样本有1000个，拼接一个和事件数组shape一样的数组，按行方向拼接，也就是并列两个事件shape一样的矩阵。
        # d[0]是events原始数据，通常是类似这个(132425, 4)维度，第一维度每个事件矩阵数据都不一样。
        # i*np.ones((len(d[0]),1), dtype=np.float32)创建了一个和原始events数据第一维度一样的矩阵，第二维度是1,比如，(132425, 1)
        # 拼接完之后的矩阵维度是，第二维度增加1，比如，(132425, 5)        
        events.append(ev)
        # 最终事件数据的最后一维度就是[0,1,2,3]的一个，依次排列下来
        # print("ev ---》", ev)
    events = torch.from_numpy(np.concatenate(events,0)) # 最后再讲所有的事件的从列方向，把时间拼接起来，第一维度增加，第二维度还是5
    labels = default_collate(labels) # label的维度是4
    return events, labels