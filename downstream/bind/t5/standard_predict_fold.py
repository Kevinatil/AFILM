#五折平均测试
from transformers import BertTokenizer, BertModel, T5Tokenizer, T5EncoderModel
from torch.utils.data import DataLoader
import re
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import DataLoader
import random
from sklearn import preprocessing
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix, mean_squared_error
import os
from sys import getsizeof
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import math
import matplotlib.pyplot as plt


device=torch.device('cuda')



class ContextPooling(nn.Module):
    def __init__(self,seq_len,in_dim=1024):
        super(ContextPooling,self).__init__()
        self.seq_len=seq_len
        self.conv=nn.Sequential(
            nn.Conv1d(in_dim,in_dim*2,3,stride=1,padding=1),
            nn.LayerNorm((in_dim*2,seq_len)),
            nn.LeakyReLU(True),

            nn.Conv1d(in_dim*2,in_dim*2,3,stride=1,padding=1),
            nn.LayerNorm((in_dim*2,seq_len)),
            nn.LeakyReLU(True),

            nn.Conv1d(in_dim*2,2,3,stride=1,padding=1),
            nn.LayerNorm((2,seq_len)),
            nn.LeakyReLU(True),
        )

    def _local_normal(self,s,center,r=0.1):
        PI=3.1415926
        std_=(r*self.seq_len*s[:,center]).unsqueeze(1) #[B,1]
        mean_=center
        place=torch.arange(self.seq_len).float().repeat(std_.shape[0],1).to(device) # [B,L]

        #print(std_)

        ret=pow(2*PI,-0.5)*torch.pow(std_,-1)*torch.exp(-torch.pow(place-mean_,2)/(1e-5+2*torch.pow(std_,2)))

        #ret-=torch.max(ret,dim=1)[0].unsqueeze(1)
        #ret=torch.softmax(ret,dim=1)

        ret/=torch.max(ret,dim=1)[0].unsqueeze(1)


        return ret

    def forward(self,feats): # feats: [B,L,1024]
        feats_=feats.permute(0,2,1)
        feats_=self.conv(feats_) #output: [B,2,L]
        s,w=feats_[:,0,:].squeeze(1),feats_[:,1,:].squeeze(1) #[B,L]
        s=torch.softmax(s,1)
        w=torch.softmax(w,1)

        out=[]

        for i in range(self.seq_len):
            w_=self._local_normal(s,i)*w
            w_=w_.unsqueeze(2) # [B,L,1]
            out.append((w_*feats).sum(1,keepdim=True)) # w_ [B,L,1], feats [B,L,1024]

        out=torch.cat(out,dim=1) # [B,L,1024]
        return out

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, output_size, dropout_prob):   
        super(SelfAttention, self).__init__()

        assert output_size%num_attention_heads==0

        self.num_attention_heads = num_attention_heads
        #self.attention_head_size = int(hidden_size / num_attention_heads)
        self.attention_head_size= int(output_size/num_attention_heads)
        self.all_head_size = int(self.num_attention_heads * self.attention_head_size)
        
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        # dropout
        self.dropout = nn.Dropout(dropout_prob)

    def transpose_for_scores(self, x):
        # INPUT:  x'shape = [bs, seqlen, hid_size]
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        
        mixed_query_layer = self.query(hidden_states)   # [bs, seqlen, hid_size]
        mixed_key_layer = self.key(hidden_states)       # [bs, seqlen, hid_size]
        mixed_value_layer = self.value(hidden_states)   # [bs, seqlen, hid_size]

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # [bs, 8, seqlen, 16]*[bs, 8, 16, seqlen]  ==> [bs, 8, seqlen, seqlen]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)   # [bs, 8, seqlen, seqlen]

        attention_probs = nn.Softmax(dim=-1)(attention_scores)    # [bs, 8, seqlen, seqlen]

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs).to(torch.float32)
        
        context_layer = torch.matmul(attention_probs, value_layer)   # [bs, 8, seqlen, 16]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()   # [bs, seqlen, 8, 16]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)   # [bs, seqlen, 128]
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

class SoluModel(nn.Module):
    def __init__(self, seq_len ,in_dim=1024, sa_out=1024, conv_out=1024):
        super(SoluModel, self).__init__()
        
        #self.self_attention=SelfAttention(in_dim,4,sa_out,0.6) # input: [B,L,1024] output: [B,L,1024]
        self.contextpooling=ContextPooling(seq_len,in_dim)

        self.conv=nn.Sequential( #input: [B,1024,L] output: [B,1024,L]
            nn.Conv1d(in_dim,in_dim*2,3,stride=1,padding=1),
            nn.LayerNorm((in_dim*2,seq_len)),
            nn.LeakyReLU(True),

            nn.Conv1d(in_dim*2,in_dim*2,3,stride=1,padding=1),
            nn.LayerNorm((in_dim*2,seq_len)),
            nn.LeakyReLU(True),

            nn.Conv1d(in_dim*2,conv_out,3,stride=1,padding=1),
            nn.LayerNorm((conv_out,seq_len)),
            nn.LeakyReLU(True),
        )

        self.cls_dim=sa_out+conv_out

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim, self.cls_dim // 4),
            nn.LeakyReLU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),

            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim//4, self.cls_dim // 4),
            nn.LeakyReLU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),

            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim//4, self.cls_dim // 64),
            nn.LeakyReLU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),
            
            nn.Linear(self.cls_dim // 64, 1),
            
            nn.Sigmoid())

        self.regressor = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim, self.cls_dim // 4),
            nn.LeakyReLU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),

            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim//4, self.cls_dim // 4),
            nn.LeakyReLU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),

            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim//4, self.cls_dim // 64),
            nn.LeakyReLU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),
            
            nn.Linear(self.cls_dim // 64, 1))
        
        self._initialize_weights()

    def forward(self, feats):
        out_sa=self.contextpooling(feats)+feats

        out_conv=self.conv(feats.permute(0,2,1))
        out_conv=out_conv.permute(0,2,1)+feats

        out=torch.cat([out_sa,out_conv],dim=2)
        out=torch.max(out,dim=1)[0].squeeze()

        cls_out = self.classifier(out)
        reg_out = self.regressor(out)

        #print(cls_out)

        return cls_out,reg_out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

                
                
class HuggingfaceClassifier(nn.Module):
    """
    Classifier implemented in HuggingfaceClassificationHead style.

    Args:
        d_model: feature dimensionality
        labels: number of classes
        inner_dim: dimensionality in the inner vector space.
        activation: activation function used in the feed-forward network
        dropout: dropout rate
    """

    def __init__(self,
                 d_model=1024,
                 labels=2,
                 inner_dim=None,
                 dropout=0.1):
        super().__init__()
        inner_dim = inner_dim or d_model * 2

        self._fc1 = nn.Linear(d_model, inner_dim)
        self._dropout = nn.Dropout(dropout)
        self._fc2 = nn.Linear(inner_dim, labels)
        self._activation = F.gelu
        self.sigmoid=nn.Sigmoid()
        
        self._initialize_weights()

    def forward(self, x):
        """
        Args:
            x: feature to predict labels
                :math:`(*, D)`, where D is the feature dimension

        Returns:
            - log probability of each classes
                :math: `(*, L)`, where L is the number of classes
        """
        x = self._dropout(x)
        x = self._fc1(x)
        x = self._activation(x)
        x = self._dropout(x)
        x = self._fc2(x)
        return self.sigmoid(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
                

class ScaleDataset(torch.utils.data.Dataset):
    def __init__(self,feats,cls_label):
        self.feats = torch.from_numpy(feats).float()
        self.cls_labels = torch.from_numpy(cls_label).float()

    def __getitem__(self, index):
        return self.feats[index], self.cls_labels[index]
    def __len__(self):
        return len(self.feats)


if __name__ == '__main__':

    BATCH_SIZE = 256
    backbone_name = 't5' #'T5-XL-UNI' #"T5-XL-UNI" 'BERT' 'T5-XL-UNI'
    model_dir="result_256_lr3e-05_k5_f1_t5/model_{}.pth.tar" #调整为模型路径

    model = HuggingfaceClassifier(1024, 1)
    model = model.to(device)

    if 1: #通过训练数据对测试数据进行标准化
        # load train data
        train_feats=np.load('../../../data/bind/t5_train_data_max.npy') #训练数据，用来进行测试数据归一化与标准化
        scaler=StandardScaler()
        scaler.fit(train_feats)
        del train_feats

        # load test data
        test_feats=np.load('../../../data/bind/t5_test_data_max.npy') #测试数据
        test_cls_label=np.load('../../../data/bind/t5_test_label.npy')
        test_feats=scaler.transform(test_feats)

    Test_dataset = ScaleDataset(test_feats,test_cls_label)
    test_loader = DataLoader(dataset=Test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model_n=5

    test_vc_all=[]
    test_precision_all=[]
    test_recall_all=[]
    test_f1_all=[]
    test_auc_all=[]

    # testing
    model.eval()
    with torch.no_grad():
        for model_i in range(model_n):

            print('fold',model_i)

            model_path=model_dir.format(model_i)
            model.load_state_dict(torch.load(model_path)['state_dict'])

            y_cls = []
            for step, (feat_, cls_label_) in enumerate(test_loader):
                feat_=feat_.to(device)
                if device.type=='cpu':
                    cls_output = model(feat_)#.data.numpy().squeeze()
                    cls_output=cls_output[:,1].data.numpy().squeeze()
                else:
                    cls_output = model(feat_)#.cpu().data.numpy().squeeze()
                    cls_output=cls_output.cpu().data.numpy().squeeze()
                y_cls.append(cls_output.reshape(-1,1))


            # metrics for classification
            y_cls = np.concatenate(y_cls,axis=0).flatten()
            y_cls_bool=(y_cls>0.5)
            test_cls_label=test_cls_label.flatten().astype(bool)
            cm_=confusion_matrix(test_cls_label,y_cls_bool)
            vc_=(cm_[0][0]+cm_[1][1])/(cm_[0][0]+cm_[0][1]+cm_[1][0]+cm_[1][1])
            precision_=precision_score(test_cls_label,y_cls_bool) #cm_[1][1]/(cm_[0][1]+cm_[1][1])
            recall_=recall_score(test_cls_label,y_cls_bool) #cm_[1][1]/(cm_[1][0]+cm_[1][1])
            f1_=f1_score(test_cls_label,y_cls_bool)
            auc_=roc_auc_score(test_cls_label,y_cls)

            print('fold: {}, vc: {:.4f}\n'.format(model_i,vc_))
            print('fold: {}, auc: {:.4f}, f1: {:.4f}, precision: {:.4f} recall: {:.4f}\n'.format(model_i,auc_,f1_,precision_,recall_))
            print('[[{},{}],\n[{},{}]]\n\n\n'.format(cm_[0][0],cm_[0][1],cm_[1][0],cm_[1][1]))

            test_vc_all.append(vc_)
            test_precision_all.append(precision_)
            test_recall_all.append(recall_)
            test_f1_all.append(f1_)
            test_auc_all.append(auc_)
        

    vc_mean=np.mean(test_vc_all)
    precision_mean=np.mean(test_precision_all)
    recall_mean=np.mean(test_recall_all)
    f1_mean=np.mean(test_f1_all)
    auc_mean=np.mean(test_auc_all)

    print('total, auc: {:.4f}, vc: {:.4f}, f1: {:.4f}, precision: {:.4f}, recall: {:.4f}'.format(auc_mean, vc_mean, f1_mean, precision_mean, recall_mean))
