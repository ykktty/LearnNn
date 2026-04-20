import torch


#点乘
def dm01():
    t1= torch.tensor([[1,2,3],[4,5,6]])
    t2 = torch.tensor([[7,8,9],[10,11,12]])
    t3 = t1*t2
    print(f't3:{t3}')
    t4 = t1.mul(t2)
    print(f't4:{t4}')


#矩阵乘
def dm02():
    t1 = torch.tensor([[1,2,3],[4,5,6]])
    t2 = torch.tensor([[7,8],[9,10],[11,12]])

    t3 = t1 @ t2
    print(f't3:{t3}')
    t4 = t1.matmul(t2)
    print(f't4:{t4}')
    t5 = torch.mm(t1,t2)
    print(f't5:{t5}')

if __name__ == '__main__':
    dm01()
    print('-' * 30)
    dm02()