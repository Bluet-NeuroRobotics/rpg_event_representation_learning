import torch
import numpy as np

from torch.utils.data.dataloader import default_collate


class Loader:
    """
    返回的是事件数组和标签数组
    """
    def __init__(self, dataset, flags, device):
        self.device = device
        split_indices = list(range(len(dataset)))
        sampler = torch.utils.data.sampler.SubsetRandomSampler(split_indices) #索引随机采样
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
    for i, d in enumerate(data): # 遍历数据样本
        labels.append(d[1]) # data的第二维是labels
        ev = np.concatenate([d[0], i*np.ones((len(d[0]),1), dtype=np.float32)],1) # 处理np数组的第一个维度，
        # 假设数据样本有1000个，拼接一个和事件数组shape一样的数组，按行方向拼接，也就是并列两个事件shape一样的矩阵数组。
        events.append(ev)
    events = torch.from_numpy(np.concatenate(events,0)) # 最后再讲所有的事件的从列方向，把时间拼接起来
    labels = default_collate(labels)
    return events, labels