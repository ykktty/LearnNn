import torch

t1 = torch.tensor([1, 2, 3])

t2 = t1.add(10)
print(t2)
t3 = t1 + 10
print(t3)

t4 = t1.add_(10)
print(t4)
#t1 += 10
#print(t5)

t5 = t1.sub(3)
print(t5)
t6 = t1.mul(10)
print(t6)
t7 = t1.div(10)
print(t7)
t8 = t1.neg()
print(t8)
