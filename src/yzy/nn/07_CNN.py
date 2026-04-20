import matplotlib.pyplot as plt
import torch
import torchvision
import time

from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import torch.nn as nn


BATCH_SIZE = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#准备数据集
def create_dataset():

    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=ToTensor())
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=ToTensor())
    return train_dataset, test_dataset

#搭建模型
class MyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,3,1,0)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,3,1,0)
        self.pool2 = nn.MaxPool2d(2,2)
        self.linear1 = nn.Linear(16*6*6,120)
        self.dropout1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(120,84)
        self.dropout2 = nn.Dropout(0.3)
        self.output = nn.Linear(84,10)

    def forward(self,x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.reshape(x.size(0),-1)
        x = torch.relu(self.linear1(x))
        x = self.dropout1(x)
        x = torch.relu(self.linear2(x))
        x = self.dropout2(x)
        x = self.output(x)

        return x

#训练
def train(train_dataset):
    dataloader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
    model = MyNet().to(DEVICE)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    epochs = 50
    for epoch in range(epochs):
        total_loss, total_num,total_correct = 0.0,0,0
        start_time = time.time()
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_pred = model(x)
            loss = criterion(y_pred,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_correct += (torch.argmax(y_pred,dim=-1) == y).sum().item()
            # total_num += x.size(0)
            # total_loss += loss.item() * x.size(0)
            total_num += len(y)
            total_loss += loss.item()*len(y)

        print(f'epoch: {epoch+1}, loss: {total_loss/total_num:.4f}, acc: {total_correct/total_num:.4f},time: {time.time()-start_time:.2f}s')
    torch.save(model.state_dict(),'./model/model.pth')

#测试
def evaluate(test_dataset):
    dataloader = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False)
    model = MyNet().to(DEVICE)
    model.load_state_dict(torch.load('./model/model.pth'))
    model.eval()
    total_correct,total_num = 0,0
    for x, y in dataloader:
        x,y = x.to(DEVICE),y.to(DEVICE)
        y_pred = model(x)
        total_correct += (torch.argmax(y_pred,dim=-1) == y).sum().item()
        total_num += len(y)
    print(f'acc: {total_correct/total_num:.4f}')



if __name__ == '__main__':
    train_dataset, test_dataset = create_dataset()
    #train(train_dataset)
    evaluate(test_dataset)
