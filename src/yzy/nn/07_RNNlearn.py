import torch
import jieba
import torch.nn as nn

def dm01():
    text = '朝暮与年岁并往，一同行至天光'

    words = jieba.lcut(text)
    #print( words)

    embed = nn.Embedding(len(words),4)

    for index , word in enumerate(words):
        print(index,word)
        #print(f'{type(index)}')

        word_vector = embed(torch.tensor( index))
        print(f'{word_vector}')

def dm02():
    rnn = nn.RNN(input_size=128,hidden_size=256,num_layers=1)

    x = torch.randn(size=(5,32,128))



    output,h1 = rnn(x,h0)
    print(output.shape)
    print(h1.shape)



if  __name__ == '__main__':
    dm01()