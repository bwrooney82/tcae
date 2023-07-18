import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from segmentation import ddx, segmentation
from channel_shuffle import channel_shuffle
from dataset import MyDataset
from Mymodel import Mymodel

'''前端处理'''

path = r"D:\Desktop\code\训练样本2.csv"  # 预处理后样本集

level = 3
value = 0.5
ddx_data = ddx(path, level, value,'Train')  # 计算加加速度，去噪

num_segment = 9
boundaries = segmentation(ddx_data, path, num_segment)  # 切分成segment

num_gens = 9
batch_size = 5
all_samples, all_boundaries = channel_shuffle(boundaries, path, num_gens)  # 进行channel shuffle，返回新样本以及新样本的分界点

'''制作数据集'''

all_samples = [torch.from_numpy(df.values).float() for df in all_samples]
all_boundaries = [torch.tensor(boundary) for boundary in all_boundaries]  # 转换成张量

traindataset = MyDataset(all_samples, all_boundaries)
train_dataloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)  # 封装成数据集 sample与breaks一一对应

num_epochs = 200
input_size = 9
hidden_size = 32
num_layers = 3
dilation = 2
sample_length = 128
lr = 0.001
latent_size = hidden_size // (2 ** (num_layers))

model = Mymodel(input_size=input_size, hidden_size=hidden_size, latent_size = latent_size, num_layers=num_layers,
                sample_length=sample_length, batch_size=batch_size)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_values = []
counter = 0

for epoch in range(num_epochs):
    epoch_loss = 0
    for batch_idx, batch_data in enumerate(train_dataloader):
        optimizer.zero_grad()
        input = batch_data['sample']
        output= model(batch_data)

        loss = criterion(output, input)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * input.size(0)

    avg_loss = epoch_loss / len(traindataset)
    loss_values.append(avg_loss)
    counter += 1
    if counter == 10:  # 每隔10个epoch打印一次损失
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, avg_loss))
        counter = 0  # 计数器重置为0
    if epoch == num_epochs - 1:  # 最后一个epoch
        Memory_Matrix = torch.cat(model.Memory_matrix)
        torch.save(Memory_Matrix, 'MM.pt')
        params = {
            'scale_norm': model.scale_norm.state_dict(),
            'channel attention': model.channel_attention.state_dict(),
            'encoder': model.encoder.state_dict(),
            'decoder': model.decoder.state_dict(),
            'conv': model.conv.state_dict(),
            'scale recover': model.scale_recover.state_dict()
        }
        torch.save(params, 'train_epoch_{}.pth'.format(num_epochs))
        torch.save(model.state_dict(), 'train_model_epoch_{}.pth'.format(num_epochs))
    else:
        model.Memory_matrix = []

plt.plot(loss_values)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()



'''计算异常分数'''
model = Mymodel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                sample_length=sample_length, batch_size=1,latent_size=latent_size)
model.load_state_dict(torch.load('train_model_epoch_{}.pth'.format(num_epochs)))
metric_dataloader = DataLoader(traindataset, batch_size=1, shuffle=True)
losses=[]
model.eval()
for batch_idx, batch_data in enumerate(metric_dataloader):
    with torch.no_grad():
        input = batch_data['sample']
        output = model(batch_data)
        loss = criterion(output, input)
        losses.append(loss.item())
mean = np.mean(losses)
std = np.std(losses)
upper_bound = mean + 3 * std
lower_bound = mean - 3 * std
print("异常分数阈值为{}，{}".format(upper_bound,lower_bound))