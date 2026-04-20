import torch

t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])

# 有dim参数
print(t1.sum(dim=0))    #列求和
print(t1.sum(dim=1))    #行求和
print(t1.sum())         #全局求和

print('-' * 10)

#max
#min
#mean

#没有dim参数
print(t1.pow(2))
print(t1 ** 2)
print(t1.sqrt())    #矩阵的平方根
print(f'矩阵的指数：{t1.exp()}')     #矩阵的指数

print(t1.log())
print(t1.log10())
