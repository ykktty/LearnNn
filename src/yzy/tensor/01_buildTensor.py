import torch
import numpy

#torch.tensor
def dm01():
    #标量张量
    t1 = torch.tensor(10)
    print(f't1:{t1},type:{type(t1)}')
    print('-' * 30)

    #二维张量
    t2 = torch.tensor([[1, 2], [3, 4]])
    print(f't2:{t2},type:{type(t2)}')
    print('-' * 30)

    #numpy数据转为张量
    data = numpy.random.randint(0,10,size=(2,3))
    print(f'data:{data},type:{type(data)}')
    t3 = torch.tensor( data, dtype=torch.float)
    print(f't3:{t3},type:{type(t3)}')
    print('-' * 30)

    #tensor方法指定形状创建张量 报错
    t4 = torch.tensor(2,3)
    print(f't4:{t4},type:{type(t4)}')
    print('-' * 30)




if __name__ == '__main__':
    dm01()