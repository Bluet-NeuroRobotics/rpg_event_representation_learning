import torch.nn as nn
from os.path import join, dirname, isfile
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.models.resnet import resnet34 # 导入torchvision内的resnet网络结构
import tqdm


class ValueLayer(nn.Module):
    def __init__(self, mlp_layers, activation=nn.ReLU(), num_channels=9):
        assert mlp_layers[-1] == 1, "Last layer of the mlp must have 1 input channel."
        assert mlp_layers[0] == 1, "First layer of the mlp must have 1 output channel"

        nn.Module.__init__(self)
        self.mlp = nn.ModuleList() # 设计神经网络的层数作为输入
        self.activation = activation # 激活函数

        # create mlp 创建MLP多层感知器，最后构建一个满足mlp_layers层数的MLP
        in_channels = 1 # 输入层是1维的
        for out_channels in mlp_layers[1:]: # 遍历设计的除输入层以外的网络层，依次往下传递输出的通道，比如 out_channels[100, 100, 1]，一共遍历三次
            self.mlp.append(nn.Linear(in_channels, out_channels)) # 给mlp添加一个全连接的module
            in_channels = out_channels # 交换输出通道到下一个全连接层作为输入通道

        # init with trilinear kernel 启动三线性插值内核
        path = join(dirname(__file__), "quantization_layer_init", "trilinear_init.pth")
        if isfile(path): #如果有这个模型路径，就加载三线性启动的模型参数
            state_dict = torch.load(path)
            self.load_state_dict(state_dict)
        else:
            self.init_kernel(num_channels) # 没有训练好的模型的话，启动内核

    def forward(self, x):
        # create sample of batchsize 1 and input channels 1
        x = x[None,...,None] # 创建一个batchsize为1和输入通道为1的样本

        # apply mlp convolution 将MLP的运算进行下去
        for i in range(len(self.mlp[:-1])):
            x = self.activation(self.mlp[i](x))

        x = self.mlp[-1](x) #最后一层网络不用激活函数
        x = x.squeeze() # 对维度为1的去除，维度不为1的无影响

        return x

    def init_kernel(self, num_channels):
        ts = torch.zeros((1, 2000)) # 创建一个times存储空间，为了存储两个极性的(t,x,y)
        optim = torch.optim.Adam(self.parameters(), lr=1e-2) # 定义梯度优化器

        torch.manual_seed(1)# 设置CPU生成随机数的种子，方便下次复现实验结果

        for _ in tqdm.tqdm(range(1000)):  # converges in a reasonable time 设置一个合理的拟合时间
            optim.zero_grad() # 重置所有的梯度为0，

            ts.uniform_(-1, 1) # 即在[-1, 1]的随机均匀分布里面取值并重新赋值, 给ts这个张量数组初始化

            # gt 真值
            gt_values = self.trilinear_kernel(ts, num_channels) # 返回三线性内核处理后的真值

            # pred 预测值
            values = self.forward(ts) # 推理输入的ts

            # optimize
            loss = (values - gt_values).pow(2).sum() # 计算误差

            loss.backward() # 误差反向传播
            optim.step() # 梯度下降迭代

    # 三线性插值的内核定义，固定的规则
    def trilinear_kernel(self, ts, num_channels):
        gt_values = torch.zeros_like(ts) # 开辟一个真值变量和ts维度一致

        #因为输入的ts是[-1,1]区间的，所以有大于0和小于0的分布，
        gt_values[ts > 0] = (1 - (num_channels-1) * ts)[ts > 0]
        gt_values[ts < 0] = ((num_channels-1) * ts + 1)[ts < 0]

        gt_values[ts < -1.0 / (num_channels-1)] = 0
        gt_values[ts > 1.0 / (num_channels-1)] = 0

        return gt_values

# 定义量化层,输出一个voxlize体素化的事件数据表征
class QuantizationLayer(nn.Module):
    def __init__(self, dim,
                 mlp_layers=[1, 100, 100, 1],
                 activation=nn.LeakyReLU(negative_slope=0.1)):
        nn.Module.__init__(self)
        # 创建一个value layer
        self.value_layer = ValueLayer(mlp_layers, # MLP层数，比如，[1, 30, 30, 1]
                                      activation=activation, # 激活函数，比如LeakyReLU
                                      num_channels=dim[0]) # 通道数是输入数据维度的第一维，比如是9
        self.dim = dim # 输入数据的维度,比如(9,180,240)

    def forward(self, events):
        # points is a list, since events can have any size
        B = int((1+events[-1,-1]).item()) # 获取最后一个事件元素的值,强转为int类型，等于4,[3+1]
        num_voxels = int(2 * np.prod(self.dim) * B) # 将相机一帧画面的(C, H, W) 和 一个类别文件中具有B个不同的事件属性数量相乘，然后乘上2, 所以维度[B*2,C,H,W]，等于4*2*9*180*240=3110400
        vox = events[0].new_full([num_voxels,], fill_value=0) # 构建一个和输入事件Tensor维度一样的数组空间存储值为0，并且可以保持和输入事件数据类型一致
        # print("vox", vox.shape) # 3110400
        C, H, W = self.dim #(9,180,240)

        # get values for each channel 提出每个事件的数据的子数据项，
        x, y, t, p, b = events.t() 
        # print("t shape",t.shape)每个事件集合的时间长度不一致

        # normalizing timestamps 将每个事件不同时间步数长度进行正则化到[0,1]
        for bi in range(B):
            t[events[:,-1] == bi] /= t[events[:,-1] == bi].max()

        p = (p+1)/2  # maps polarity to 0, 1 将每个事件的极性从[-1,1]映射到[0,1]之间
        # 下面的代码是计算索引
        idx_before_bins = x \
                          + W * y \
                          + 0 \
                          + W * H * C * p \
                          + W * H * C * 2 * b

        for i_bin in range(C):
            values = t * self.value_layer.forward(t-i_bin/(C-1)) #使用MLP将值进行kernel处理

            # draw in voxel grid
            idx = idx_before_bins + W * H * i_bin #加上C维度的bin的index
            vox.put_(idx.long(), values, accumulate=True) # 填充上面量化完的值到体素中

        vox = vox.view(-1, 2, C, H, W) # 展开VOX体素的维度
        # print("vox shape2", vox.shape) # [4, 2, 9, 180, 240]
        vox = torch.cat([vox[:, 0, ...], vox[:, 1, ...]], 1) #将VOX的第二维度，维度值为2，按行拼接
        # print("vox shape3", vox.shape) # torch.Size([4, 18, 180, 240])

        return vox


class Classifier(nn.Module):
    def __init__(self,
                 voxel_dimension=(9,180,240),  # dimension of voxel will be C x 2 x H x W
                 crop_dimension=(224, 224),  # dimension of crop before it goes into classifier
                 num_classes=101,
                 mlp_layers=[1, 30, 30, 1],
                 activation=nn.LeakyReLU(negative_slope=0.1),
                 pretrained=True):

        nn.Module.__init__(self)
        self.quantization_layer = QuantizationLayer(voxel_dimension, mlp_layers, activation)
        self.classifier = resnet34(pretrained=pretrained) #定义一个Resnet34

        self.crop_dimension = crop_dimension

        # replace fc layer and first convolutional layer 更换了两个层
        input_channels = 2*voxel_dimension[0]
        self.classifier.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False) # 输入的通道数更换成对应的2*9=18
        self.classifier.fc = nn.Linear(self.classifier.fc.in_features, num_classes) #输出的全连接层的类别数量换掉，101类

    def crop_and_resize_to_resolution(self, x, output_resolution=(224, 224)):
        B, C, H, W = x.shape
        if H > W:
            h = H // 2
            x = x[:, :, h - W // 2:h + W // 2, :]
        else:
            h = W // 2
            x = x[:, :, :, h - H // 2:h + H // 2]

        x = F.interpolate(x, size=output_resolution) #插值的方式改变分辨率到(224,224)

        return x

    def forward(self, x):
        vox = self.quantization_layer.forward(x)
        # print("x  SHAPE-->", x.shape)
        vox_cropped = self.crop_and_resize_to_resolution(vox, self.crop_dimension)
        pred = self.classifier.forward(vox_cropped)
        return pred, vox


