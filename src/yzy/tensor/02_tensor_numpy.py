"""
掌握
    张量 -》 numpy    张量对象.numpy（）
    numpy -》 张量     torch.tensor（numpy对象）
    从 标量张量里提取内容 标亮张量.item()
"""

import torch
import numpy as np

#1
def numpy_to_tensor():
    t1 = torch.randn(4, 4)
    print(f't1:{t1},type:{type(t1)}')
    n1 = t1.numpy()                     #共享内存
    print(f'n1:{n1},type:{type(n1)}')
    n2 = t1.numpy().copy()              #不共享内存
    print(f'n2:{n2},type:{type(n2)}')
def tensor_to_numpy():
    n1 = np.random.randn(4, 4)
    print(f'n1:{n1},type:{type(n1)}')
    t2= torch.from_numpy(n1)            #共享内存
    print(f't2:{t2},type:{type(t2)}')
    t1 = torch.tensor(n1)               #不共享内存
    print(f't1:{t1},type:{type(t1)}')
def get_tensor_content():
    t1 = torch.tensor([1,])
    print(f't1:{t1},type:{type(t1)}')
    print(f't1.item():{t1.item()}')
    pass

if __name__ == '__main__':
    numpy_to_tensor()
    print('-' * 30)
    tensor_to_numpy()
    print('-' * 30)
    get_tensor_content()