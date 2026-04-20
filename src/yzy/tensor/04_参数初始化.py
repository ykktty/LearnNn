import torch.nn as nn

#随机初始化
def dm01():
    linear = nn.Linear(5,3)
    nn.init.uniform_(linear.weight)
    nn.init.uniform_(linear.bias)
    print(linear.weight.data)
    print(linear.bias.data)

#固定初始化
def dm02():
    linear = nn.Linear(5, 3)
    nn.init.constant_(linear.weight,3)
    nn.init.constant_(linear.bias,3)
    print(linear.weight.data)
    print(linear.bias.data)

#全零初始化
def dm03():
    linear = nn.Linear(5,3)
    nn.init.zeros_(linear.weight)
    nn.init.zeros_(linear.bias)
    print(linear.weight.data)
    print(linear.bias.data)


#全一初始化
def dm04():
    linear = nn.Linear(5,3)
    nn.init.ones_(linear.weight)
    nn.init.ones_(linear.bias)
    print(linear.weight.data)
    print(linear.bias.data)

#正态初始化
def dm05():
    linear = nn.Linear(5,3)
    nn.init.normal_(linear.weight)
    nn.init.normal_(linear.bias)
    print(linear.weight.data)
    print(linear.bias.data)


#kaiming初始化
def dm06():
    #正态分布
    linear1 = nn.Linear(5,3)
    nn.init.kaiming_normal_(linear1.weight)
    print(f'kaiming正态：{linear1.weight.data}')

    #均匀分布
    linear2 = nn.Linear(5,3)
    nn.init.kaiming_uniform_(linear2.weight)
    print(f'kaiming均匀：{linear2.weight.data}')

#xavier初始化
def dm07():
    linear = nn.Linear(5,3)
    nn.init.xavier_normal_(linear.weight)
    nn.init.xavier_normal_(linear.bias)
    print(linear.weight.data)
    print(linear.bias.data)


if __name__ == '__main__':
    dm06()