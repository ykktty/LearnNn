import torch

#创建指定类型的张量
t1 = torch.tensor([1,2,3],dtype=torch.float)
print(f't1:{t1},type:{type(t1)}，dtype:{t1.dtype}')
print('-' * 30)

#创建张量后类型转换
#方式1
t2 = t1.type(torch.int16)
print(f't2:{t2},type:{type(t2)}，dtype:{t2.dtype}')
print('-' * 30)

#方式2
print(t2.half())    #float16
print(t2.float())   #float32
print(t2.double())  #float64
print(t2.long())    #int64
print(t2.short())   #int16
print(t2.int())     #int32
