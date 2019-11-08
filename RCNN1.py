import torch
from torch import nn
import torch.utils.data as d
from torch.utils.data.dataset import Dataset
import numpy as np
from tensorboardX import SummaryWriter as S
import torch.nn.functional as F
from construct_data import trainData
from test_data import testData
import argparse
import time
start=time.clock()
train_data=trainData('细粒度评论.txt','细粒度标签.txt')
train_loader=d.DataLoader(dataset=train_data,batch_size=100,shuffle=True)
test_data=testData('验证集评论.txt','验证集标签.txt')
test_loader=d.DataLoader(dataset=test_data,batch_size=500,shuffle=True)
EPOCH = 1
TIME_STEP = 300
INPUT_SIZE = 200
LR=0.01
writer=S(log_dir='./模型')
class RCNN(nn.Module):
    def __init__(self):
        super(RCNN,self).__init__()
        self.lstm=nn.LSTM(input_size=INPUT_SIZE,
                          hidden_size=128,
                          num_layers=2,
                          batch_first=True,
                          dropout=0,
                          bidirectional=True)
        self.L1=nn.Linear(456,128,bias=False)#此处没有bias是因为避免全为0行出现值，从而影响提取最大值
        self.tanh=nn.Tanh()
        #self.Max1d=nn.MaxPool1d(self.length[0])
        self.L2=nn.Linear(128,4)
    def rep_text(self,a,b,c,d):
        #print(b[0])
        a=a.view(len(b),b[0],2,128)
        #M=torch.FloatTensor(c)
        mat_FW=torch.zeros((a.shape[0],b[0],128))
        for i in range(a.shape[0]):
            mat_FW[i][0]=d
            for l in range(1,b[i]):
                mat_FW[i][l]=a[i][l][0]
        #print('FW:',mat_FW,mat_FW.shape)
        mat_BW=torch.zeros((a.shape[0],b[0],128))
        for j in range(a.shape[0]):
            mat_BW[j][b[j]-1]=d
            for k in range(b[j]-1):
                mat_BW[j][k]=a[j][b[j]-2-k][1]
        #print('BW:',mat_BW,mat_BW.shape)
        data=torch.cat((mat_FW,c),2)
        #print('SHUJU:',data,data.shape)
        data=torch.cat((data,mat_BW),2)
        #print('SHUJU:',data,data.shape)
        return data
    def packedd(self,x,y):
        x=x.view(-1,300,INPUT_SIZE)
        #print(x.shape)
        x=x.detach().numpy()
        if torch.is_tensor(y) == True:
            y=y.numpy()
        num=[]
        for i in range(x.shape[0]):
            k=0
            for j in range(300):
                if x[i][j].any()==True:
                    k+=1
            num.append(k)
        lengths=sorted(num,reverse=True)
        lengths_2=lengths
        matrix=np.zeros((x.shape[0],lengths[0],INPUT_SIZE))
        label=[]
        for i in range(x.shape[0]):
            matrix[i][:max(num)]=x[num.index(max(num))][:max(num)]
            label.append(y[num.index(max(num))])
            elment=num.index(max(num))
            num[elment]=0
        label=torch.LongTensor(label)
        matrix=torch.FloatTensor(matrix)
        lengths=torch.LongTensor(lengths)
        x_packed=nn.utils.rnn.pack_padded_sequence(matrix, lengths=lengths, batch_first=True)
        return x_packed,label,lengths,matrix
    def forward(self,b_x,b_y):
        x_packed,label,length,mat=self.packedd(b_x,b_y)
        b_x=b_x.view(-1,300,INPUT_SIZE)
        np.random.seed(1)
        H_0=np.random.rand(128)
        h_f=torch.FloatTensor(H_0)
        H_0=np.tile(H_0,(4,b_x.shape[0],1))
        H_0=torch.FloatTensor(H_0)
        C_0=H_0
        hidd=(H_0,C_0)
        r_out,(h_n,h_c)=self.lstm(x_packed,hidd)
        out = torch.nn.utils.rnn.pad_packed_sequence(r_out, batch_first=True)
        out=out[0]
        data=self.rep_text(out,length,mat,h_f)
        out=self.L1(data)
        out=self.tanh(out)
        out=out.permute(0,2,1)
        Max1d=nn.MaxPool1d(int(length.detach().numpy()[0]))
        output=Max1d(out).squeeze(2)
        outputs=self.L2(output)
        return outputs,label
rcnn=RCNN()
print(rcnn)
optimizer=torch.optim.SGD(rcnn.parameters(),lr=LR)
weight=[1,1.2,1.2,1]
weight=torch.FloatTensor(weight)
loss_func=nn.CrossEntropyLoss()
ACC_BEST=0
is_train=True
if is_train==True:
    for epoch in range(EPOCH):
        rcnn.train()
        for step,(b_x,b_y) in enumerate(train_loader):
            output,labels=rcnn(b_x,b_y)
            loss=loss_func(output,labels)
            print(loss.item())
            writer.add_scalar('SS',loss,1050*epoch+step+1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step%50==0:
                for step,(test_x,test_y) in enumerate(test_loader):
                    test_outputs,labels=rcnn(test_x,test_y)
                    labels=labels.numpy()
                    print('标签：',labels)
                    pred_y=torch.max(test_outputs, 1)[1].data.numpy().squeeze()
                    accuracy=float((pred_y==labels).astype(int).sum())/float(labels.size)
                    print('准确率:',accuracy)
                    if accuracy > ACC_BEST:
                        ACC_BEST=accuracy
                        torch.save(rcnn.state_dict(),'canshu.pkl')
else:
    rcnn.eval()
    rcnn.load_state_dict(torch.load('canshu.pkl'))
    for step,(test_x,test_y) in enumerate(test_loader):
        test_output,labels=rnn(test_x,test_y)
        labels=labels.numpy()
        pred_y=torch.max(test_output, 1)[1].data.numpy().squeeze()
        accuracy=float((pred_y==labels).astype(int).sum()) / float(labels.size)
        print(accuracy)
print('End')
writer.close()
elapsed=(time.clock() - start)
print('程序执行时间：',elapsed)
