import torch.nn
import torch

def dm01():
    #y_true = torch.tensor([[0,1,0],[1,0,0]],dtype=torch.float)
    y_true = torch.tensor([1,2])

    y_pred = torch.tensor([[0.1,0.9,0.2],[0.7,0.2,0.1]])

    criterion = torch.nn.CrossEntropyLoss()

    loss = criterion(y_pred,y_true)
    print(f'loss:{loss}')

if __name__ == '__main__':
    dm01()