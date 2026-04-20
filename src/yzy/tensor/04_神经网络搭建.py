import torch.nn as nn
import torch
from torchsummary import summary

class MyNet(nn.Module):
    def __init__(self):
        # 继承父类
        super().__init__()
        # 定义网络
        self.linear1 = nn.Linear(3,3)
        self.linear2 = nn.Linear(3,2)
        self.output = nn.Linear(2,2)
        # 参数初始化
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)

        nn.init.kaiming_normal_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x):
        # 前向传播
        x = torch.sigmoid(self.linear1(x))

        x = torch.relu(self.linear2(x))

        x = torch.softmax(self.output(x), dim=-1)

        return x

def train():
    model = MyNet()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    model = model.to(device)

    data = torch.randn(5,3).to(device)
    print(f'data: {data}')
    print(f'data.shape: {data.shape}')
    print(f'data.requires_grad: {data.requires_grad}')

    output = model(data)
    print(f'output: {output}')
    print(f'output.shape: {output.shape}')
    print(f'output.requires_grad: {output.requires_grad}')

    #计算模型参数
    summary(model, input_size=(3,))
    print('_______')
    #查看模型参数
    for name, param in model.named_parameters():
        print(f'name: {name}, param: {param}')
        print('_______')


if __name__ == '__main__':
    train()
