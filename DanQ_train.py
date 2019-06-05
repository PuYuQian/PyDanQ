import scipy.io
import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import visdom
import mkdir
import time
torch.manual_seed(1337)
np.random.seed(1337)
torch.cuda.manual_seed(1337)
torch.backends.cudnn.benchmark=True

## Hyper Parameters
EPOCH = 60
BATCH_SIZE = 100
LR = 0.001
save_model_time = '0525'

mkpath = 'model/model%s'% save_model_time
mkdir.mkdir(mkpath)
#patience = 5

print('starting loading the data')
np_valid_data = scipy.io.loadmat('valid.mat')

validX_data = torch.FloatTensor(np_valid_data['validxdata'])
validY_data = torch.FloatTensor(np_valid_data['validdata'])

params = {'batch_size': 100,'num_workers': 2}

valid_loader = Data.DataLoader(
    dataset=Data.TensorDataset(validX_data, validY_data), 
    shuffle=False,
    **params)

vis = visdom.Visdom(env='DanQ')

win = vis.line(
    X=np.array([0]),
    Y=np.array([0]),
    opts=dict(
        title='LOSS-EPOCH(%s)' % save_model_time,
        showlegend=True,),
    name="train")
vis.line(
    X=np.array([0]),
    Y=np.array([0]),
    win=win,
    update="new",
    name="val",
)

print('compling the network')
class DanQ(nn.Module):
    def __init__(self, ):
        super(DanQ, self).__init__()
        self.Conv1 = nn.Conv1d(in_channels=4, out_channels=320, kernel_size=26)
        #self.Conv1.weight.data = torch.Tensor(np.load('conv1_weights.npy'))
        #self.Conv1.bias.data = torch.Tensor(np.load('conv1_bias.npy'))
        self.Maxpool = nn.MaxPool1d(kernel_size=13, stride=13)
        self.Drop1 = nn.Dropout(p=0.2)
        self.BiLSTM = nn.LSTM(input_size=320, hidden_size=320, num_layers=2,
                                 batch_first=True,
                                 dropout=0.5,
                                 bidirectional=True)
        self.Linear1 = nn.Linear(75*640, 925)
        self.Linear2 = nn.Linear(925, 919)

    def forward(self, input):
        x = self.Conv1(input)
        x = F.relu(x)
        x = self.Maxpool(x)
        x = self.Drop1(x)
        x_x = torch.transpose(x, 1, 2)
        x, (h_n,h_c) = self.BiLSTM(x_x)
        #x, h_n = self.BiGRU(x_x)
        x = x.contiguous().view(-1, 75*640)
        x = self.Linear1(x)
        x = F.relu(x)
        x = self.Linear2(x)
        return x


danq = DanQ()
danq.cuda()
print(danq)

optimizer = optim.RMSprop(danq.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5,verbose=1)
loss_func = nn.BCEWithLogitsLoss()

print('starting training')
# training and validating
since = time.time()

train_losses = []
valid_losses = []

for epoch in range(EPOCH):
    danq.train()
    train_loss = 0
    for i in range(1,11):
        trainX_data = torch.load('pt_data/%s.pt' % str(i))
        trainY_data = torch.load('pt_label/%s.pt' % str(i))
        train_loader = Data.DataLoader(dataset=Data.TensorDataset(trainX_data, trainY_data), shuffle=True, **params)
        for step, (train_batch_x, train_batch_y) in enumerate(train_loader):

            train_batch_x = train_batch_x.cuda()
            train_batch_y = train_batch_y.cuda()

            out = danq(train_batch_x)
            loss = loss_func(out, train_batch_y)

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
    i = 1

    if epoch % 5 == 0:
       torch.save(danq, 'model/model{save_model_time}/danq_net_{epoch}.pkl'.format(save_model_time=save_model_time,epoch=int(epoch/5)))
       torch.save(danq.state_dict(), 'model/model{save_model_time}/danq_net_params_{epoch}.pkl'.format(save_model_time=save_model_time,epoch=int(epoch/5)))

    
    danq.eval()

    for valid_step, (valid_batch_x, valid_batch_y) in enumerate(valid_loader):

        valid_batch_x = valid_batch_x.cuda()
        valid_batch_y = valid_batch_y.cuda()

        val_out = danq(valid_batch_x)
        val_loss = loss_func(val_out, valid_batch_y)
        valid_losses.append(val_loss.item())
        
    train_loss = np.average(train_losses)
    valid_loss = np.average(valid_losses)
    
    scheduler.step(valid_loss)
    
    epoch_len = len(str(epoch))

    print_msg = (f'[{epoch:>{epoch_len}}/{EPOCH:>{epoch_len}}] ' +
                 f'train_loss: {train_loss:.5f} ' +
                 f'valid_loss: {valid_loss:.5f}')

    print(print_msg)
    
    vis.line(
        X=np.array([epoch]),
        Y=np.array([train_loss]),
        win=win,
        update="append",
        name="train"
    )
    vis.line(
        X=np.array([epoch]),
        Y=np.array([valid_loss]),
        win=win,
        update="append",
        name="val"
    )
    #save bestmodel
    bestmodel.bestmodel(danq,save_model_time,valid_loss)
    
    train_losses = []
    valid_losses = []

time_elapsed = time.time() - since
print('time:', time_elapsed)
torch.save(danq, 'model/model{save_model_time}/danq_net_final.pkl'.format(save_model_time=save_model_time))  # save entire net
torch.save(danq.state_dict(), 'model/model{save_model_time}/danq_net_params_final.pkl'.format(save_model_time=save_model_time))
