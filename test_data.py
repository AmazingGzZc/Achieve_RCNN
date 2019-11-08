import torch
import numpy as np
from gensim.models import word2vec
from torch.utils.data.dataset import Dataset
import torch.utils.data as d
model=word2vec.Word2Vec.load('all_vectors.model')
class testData(Dataset):
    def __init__(self,file_path,label_path):
        self.file_path=file_path
        self.label_path=label_path
        file1=open(self.file_path,'r')
        file2=open(self.label_path,'r')
        self.comment=[]
        for line in file1.readlines():
            self.comment.append(line)
        self.labels=[]
        for num in file2.readlines():
            self.labels.append(num.split(' ')[0])
    def __getitem__(self,index):
        data=self.comment[index]
        label=self.labels[index]
        data=data.split(' ')
        com=np.zeros((300,200))
        if len(data)<=300:
            i=0
            for word in data:
                if word !='\n':
                    com[i]=model[word]
                    i+=1
        else:
            i=0
            for word in data:
                if word != '\n':
                    com[i]=model[word]
                    i+=1
                    if i ==300:
                        break
        if label == '1':
            label=0
        elif label == '0':
            label=1
        elif label == '-1':
            label=2
        else:
            label=3
        return torch.FloatTensor(com),torch.Tensor([label]).squeeze()
    def __len__(self):
        return len(self.labels)
'''test_data=testData('验证集评论.txt','验证集标签.txt')
test_loader=d.DataLoader(dataset=test_data,batch_size=5,shuffle=True)
for step,(x,y) in enumerate(test_loader):
    print(x,x.shape)
    print(y,y.shape)'''
