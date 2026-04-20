"""
    优化思路
        SGD --》 Adam
        lr 0.001 --> 0.0001
        对数据标准化   标准化后提升明显 其他一般 甚至下降
        增加网络深度
        增加训练轮次
        。。。
"""




import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

from torchinfo import summary


#todo 1.构建数据集
def build_dataset():
    # 1.csv加载数据
    data = pd.read_csv(r'E:\work\datasets\手机价格分类\train.csv')
    # print(f'data_type:{data.dtypes}')
    # print(f'data_head:{data.head()}')
    # print(f'data_shape:{data.shape}')

    # 2.获取x特征列和y标签列
    x , y = data.iloc[:, :-1] , data.iloc[:, -1]
    # print(f'x_head:{x.head()},x_shape:{x.shape}')
    # print(f'y_head:{y.head()},y_shape:{y.shape}')

    # 3.将特征列转换为浮点型
    x = x.astype(np.float32)
    #print(f'x_head:{x.head()},x_shape:{x.shape}')

    # 4.切分训练集和测试集
    x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)



    #todo 优化：数据标准化 提升明显
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)



    # 5.将数据转换为张量

    # train_dataset = TensorDataset(torch.tensor(x_train.values), torch.tensor(y_train.values))
    # test_dataset = TensorDataset(torch.tensor(x_test.values), torch.tensor(y_test.values))
    train_dataset = TensorDataset(torch.tensor(x_train),torch.tensor(y_train.values))
    test_dataset = TensorDataset(torch.tensor(x_test),torch.tensor(y_test.values))


    return train_dataset,test_dataset,x_train.shape[1],len(np.unique(y))


#todo 2.构建模型
class PhonePriceModel(nn.Module):
    def __init__(self,input_dim,num_dim):

        super().__init__()

        self.linear1 = nn.Linear(input_dim,128)
        self.linear2 = nn.Linear(128,256)
        self.linear3 = nn.Linear(256,num_dim)



    def forward(self,x):

        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)

        return x

#todo 3. 训练
def train(train_dataset,input_dim,num_dim):

    # 1.tensor数据集转换为DataLoader
    train_loader = DataLoader(train_dataset,batch_size=16,shuffle=True)

    # 2.构建模型
    model = PhonePriceModel(input_dim,num_dim)

    # 3.构建损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.001)

    # 4.训练
    epochs = 50
    for epoch in range(epochs):
        total_loss , batch_num = 0,0
        start_time = time.time()
        for x,y in train_loader:
            model.train()
            y_pred = model(x)
            loss = criterion(y_pred,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_num += 1

        print(f'epoch:{epoch+1},loss:{total_loss/batch_num:.4f},time:{time.time()-start_time:.2f}s')

    # 5.保存模型
    torch.save(model.state_dict(),f'./model/phone_price_model.pth')

#todo 4. 测试

def evaluate(test_dataset,input_dim,num_dim):

    model = PhonePriceModel(input_dim,num_dim)
    model.load_state_dict(torch.load(f'./model/phone_price_model.pth'))

    test_loader = DataLoader(test_dataset,batch_size=8,shuffle=False)

    model.eval()
    correct = 0
    for x,y in test_loader:
        y_pred = model(x)
        y_pred = torch.argmax(y_pred,dim=1)
        print(f'y_pred:{y_pred},y:{y}')
        correct += torch.sum(y_pred == y)

    print(type(correct))
    print(type(correct.item()))


    print(f'准确率:{correct/len(test_dataset):.4f}')


if __name__ == '__main__':

     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

     train_dataset,test_dataset,input_dim,num_dim = build_dataset()
     #print(f'训练集:{train_dataset},测试集:{test_dataset},输入维度:{input_dim},标签维度:{num_dim}')

     #model = PhonePriceModel(input_dim,num_dim)

     #summary(model,input_size=(input_dim,))

     #train(train_dataset,input_dim,num_dim)

     evaluate(test_dataset,input_dim,num_dim)


