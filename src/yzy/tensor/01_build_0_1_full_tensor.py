"""
    ones.()
    ones_like.()
    zeros.()
    zeros_like.()
    full.()
    full_like.()
"""
import torch

shape = torch.tensor([[2, 3], [4, 5], [6, 7]])

# ones
t1 = torch.ones(2, 3)
print(f't1:{t1},type:{type(t1)}')
t2  = torch.ones_like(shape)
print(f't2:{t2},type:{type(t2)}')
print('-' * 30)
t3 = torch.zeros(2, 3)
print(f't3:{t3},type:{type(t3)}')
t4 = torch.zeros_like(shape)
print(f't4:{t4},type:{type(t4)}')
print('-' * 30)
t5 = torch.full(size=(2, 3), fill_value=10)
print(f't5:{t5},type:{type(t5)}')
t6 = torch.full_like(shape, fill_value=10)
print(f't6:{t6},type:{type(t6)}')
print('-' * 30)


