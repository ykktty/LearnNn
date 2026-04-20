"""
    torch.arange() linspace()
    random.initial_seed random.manual_seed()
    rand/randn()
    randint(low,high,size=())

    arange() linspace() manual_seed() randint()
"""

import torch

#线性张量
def dm01():
    t1 = torch.arange(0, 10,2)
    print(f't1:{t1},type:{type(t1)}')
    print('-' * 30)

    t2 = torch.linspace(0, 10, 2)
    print(f't2:{t2},type:{type(t2)}')
    print('-' * 30)
#随机张量
def dm02():
    #torch.random.initial_seed()
    torch.random.manual_seed(1)

    #随机张量
    # t1 = torch.rand(2, 3)
    # print(f't1:{t1},type:{type(t1)}')
    t1 = torch.rand(size=(2, 3))
    print(f't1:{t1},type:{type(t1)}')
    print('-' * 30)
    #符合正太分布的随机张量
    t2 = torch.randn(size=(2, 3))
    print(f't2:{t2},type:{type(t2)}')
    print('-' * 30)
    #随机整数张量
    t3 = torch.randint(low=11, high=20, size=(2, 3))
    print(f't3:{t3},type:{type(t3)}')
    print('-' * 30)




if __name__ == '__main__':
    dm02()