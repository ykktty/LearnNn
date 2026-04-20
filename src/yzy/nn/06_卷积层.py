import matplotlib.pyplot as plt
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def dm01():

    img = plt.imread('./datasets/20190628155524_uhzuu.jpg')
    print(f'img:{img},img.shape:{img.shape}')

    img2 = torch.tensor( img, dtype=torch.float)
    img2 = img2.permute(2,0,1)
    print(f'img2:{img2},img2.shape:{img2.shape}')

    img3 = img2.unsqueeze(dim=0)
    print(f'img3:{img3},img3.shape:{img3.shape}')

    conv = torch.nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=2, padding=0)

    conv_img = conv(img3)

    print(f'conv_img:{conv_img},conv_img.shape:{conv_img.shape}')

    img4 = conv_img[0]
    print(f'img4:{img4},img4.shape:{img4.shape}')

    img5 = img4.permute(1,2,0)

    featrue1 = img5[:,:,0].detach().numpy()
    plt.imshow(featrue1)
    plt.show()


if  __name__ == '__main__':
    dm01()