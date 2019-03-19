import torch
from torch import optim,nn
from model import Net
from torch.utils.data import DataLoader
from torch.nn.init import kaiming_normal_
from dataset import GF2
from dataset import Resample,Normalize,ToTensor
from torchvision import transforms
from ModelConfig import Config
from matplotlib import pyplot as plt
import numpy as np
config=Config()
dataset=GF2(config.train_ms_path,config.train_pan_path,config.patch_size,transfrom=transforms.Compose([Resample(),Normalize(),ToTensor()]))
dataloader=DataLoader(dataset,batch_size=config.batch_size,shuffle=True,num_workers=1)
dataset_val=GF2(config.val_ms_path,config.val_pan_path,config.patch_size,transfrom=transforms.Compose([Resample(),Normalize(),ToTensor()]),option='Val')
dataloader_val=DataLoader(dataset_val,batch_size=config.batch_size,shuffle=True,num_workers=1)
net=Net()
assert torch.cuda.is_available()
net.apply(lambda m:kaiming_normal_(m.weight.data,mode='fan_out') if isinstance(m,nn.Conv2d) else None)
net_paras=net.state_dict()
pretrain_paras={k:v for k,v in torch.load('best_model').items() if k in net_paras and v.shape==net_paras[k].shape}
#
# # net.apply(init_weight)
net_paras.update(pretrain_paras)
# net.eval()
# net.load_state_dict(torch.load('best_model_2'))
# net.eval()
# for para in net.state_dict().keys():
#     if para in pretrain_paras:
#         net.state_dict()[para].data=pretrain_paras[para].data
#     else:
#         if 'conv' in para:
#             kaiming_normal_(net.state_dict()[para].data,mode='fan_out')
    # else:
    #     if isinstance(m,nn.Conv2d)
    #     net.state_dict()[para]=kaiming_normal_(net.state_dict()[para].weight.data,mode='fan_out') if isinstance(m,nn.Conv2d) else None
# net.state_dict().update(reserved_paras)
# net.eval()
optimizer=optim.Adam(net.parameters(),lr=config.lr,weight_decay=config.weight_decay)
criterion=nn.MSELoss()
best_loss=float('inf')
epoches=0
epoch_train_loss=[]
epoch_val_loss=[]
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
plt.ion()
net.to("cuda:0")
while(True):
    epoches+=1
    train_loss, val_loss = [], []
    for data in dataloader:
        ms=data['ms'].to("cuda:0")
        pan=data['pan'].to("cuda:0")
        label=data['label'].to("cuda:0")
        fusion_result=net.forward(ms,pan)
        optimizer.zero_grad()
        loss=criterion(fusion_result,label)
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        # if loss<best_loss:
        #     best_loss=loss.item()
        #     torch.save(net.state_dict(),"best_model")
    epoch_train_loss.append(np.mean(train_loss))
    ###validate model
    with torch.no_grad():
        for data in dataloader_val:
            ms = data['ms'].to("cuda:0")
            pan = data['pan'].to("cuda:0")
            label = data['label'].to("cuda:0")
            fusion_result = net.forward(ms, pan)
            loss = criterion(fusion_result+label, label)
            val_loss.append(loss.item())
        if best_loss>np.mean(val_loss):
            best_loss=np.mean(val_loss)
            torch.save(net.state_dict(), "best_model")
        epoch_val_loss.append(np.mean(val_loss))
    print("Train loss of Epoch %d is %e, validate loss is %e" % (epoches, epoch_train_loss[-1], epoch_val_loss[-1]))
    plt.cla()
    ax.set_ylim(0, max(epoch_train_loss[0],epoch_val_loss[0]))
    ax.plot(np.arange(1,epoches+1),epoch_train_loss,label='Train')
    ax.plot(np.arange(1,epoches+1),epoch_val_loss,'--',label='Val')
    ax.legend(loc='best')
    plt.show()
    plt.pause(0.1)

