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
from sklearn.model_selection import KFold
import math
import time
import random

os.environ['CUDA_VISIBLE_DEVICES']='1,0'
os.environ['CUDA_LAUNCH_BLOCKING']='1'

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
                 labels=6,
                 inner_dim=None,
                 dropout=0.1):
        super().__init__()
        inner_dim = inner_dim or d_model * 2

        self._fc1 = nn.Linear(d_model, inner_dim)
        self._dropout = nn.Dropout(dropout)
        self._fc2 = nn.Linear(inner_dim, labels)
        self._activation = F.gelu
        
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
        return x #F.softmax(x, dim=1)

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
    def __init__(self, feats, cls_label):
        self.feats = torch.from_numpy(feats).float()
        self.cls_labels = torch.from_numpy(cls_label).long() #.float()

    def __getitem__(self, index):
        return self.feats[index], self.cls_labels[index]
    def __len__(self):
        return len(self.feats)


#######################################
# new loss define
from torch.nn.modules.loss import _Loss

class WeightMSELoss(_Loss):

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(WeightMSELoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target, k):
        input=input.reshape(1,-1)
        target=target.reshape(1,-1)
        mask=(target>=1)
        #print(input)
        return (torch.sum((input*mask-target*mask)**2)*k+torch.sum((input*(~mask)-target*(~mask))**2))/len(input)

class RegFocalLoss(_Loss):

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(RegFocalLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, pred, target, gamma=1):
        pred=pred.reshape(1,-1)
        target=target.reshape(1,-1)
        se_=torch.abs(pred-target)
        a_=torch.pow(se_,gamma).detach()
        a_sum=torch.sum(a_).detach()
        a_=(a_/a_sum).detach()
        return torch.sum(torch.pow(se_,2)*a_)

class ClsFocalLoss(_Loss):

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(ClsFocalLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, pred, target, gamma=1, alpha=0.5):
        assert alpha<1 and alpha>0

        epsilon=1e-7
        pred=pred.reshape(-1,)
        target=target.reshape(-1,)

        pt_0=1-pred[target==0]
        pt_1=pred[target==1]

        loss_0=(-torch.pow(1-pt_0,gamma)*torch.log(pt_0+epsilon)).sum()
        loss_1=(-torch.pow(1-pt_1,gamma)*torch.log(pt_1+epsilon)).sum()

        loss=(1-alpha)*loss_0+alpha*loss_1

        return loss/len(pred)

#######################################


if __name__ == '__main__':
    BATCH_SIZE = 256
    EPOCH = 120

    lr=2e-5
    
    backbone_name='newg'

    cls_loss_func= nn.CrossEntropyLoss()

    k=10
    
    test_vc_all =[]
    test_precision_all=[]
    test_recall_all=[]
    test_f1_all=[]
    test_auc_all=[]

    res_dir = "max_result_{}_lr{}_k{}_f1_".format(BATCH_SIZE,lr,k) + backbone_name
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    embeds1=pickle.load(open('../../../data/bcell/{}_all_data-0.txt_embedding.pickle'.format(backbone_name),'rb'))
    embeds2=pickle.load(open('../../../data/bcell/{}_all_data-1.txt_embedding.pickle'.format(backbone_name),'rb'))
    embeds=embeds1+embeds2
    del embeds1,embeds2
    
    #feats=[]
    #for emb in embeds:
    #    feats.append(emb[0,:])
    #feats=np.array(feats)
    
    feats=[]
    for emb in embeds:
        feats.append(np.max(emb,axis=0))
    feats=np.array(feats)
    
    cls_label=np.load('../../../data/bcell/{}_all_label.npy'.format(backbone_name))
    
    print(feats.shape)
    print(cls_label.shape)
    
    kf=KFold(n_splits=k,shuffle=True,random_state=27)
    kf_idx=kf.split(feats,cls_label)

    counter=0

    for i_idx in kf_idx:
    #for i_ in range(k):
        #i_idx=kf_idx[i_]

        f = open(os.path.join(res_dir, 'result.txt'), 'a')
        print('fold',counter)
        f.write('fold {}, {}\n'.format(counter,time.ctime()))
        
        vc_best=0
        precision_best=0
        recall_best=0
        f1_best=0
        auc_best=0
        cm_best=None

        model = HuggingfaceClassifier(1024, 6) #SoluModel(SEQ_LEN)
        optimizer = torch.optim.Adam(model.parameters(),lr)
        model = model.to(device) #.cuda()
        
        
        # get data
        train_feats=feats[i_idx[0]]
        train_cls_label=cls_label[i_idx[0]]

        val_feats=feats[i_idx[1]]
        val_cls_label=cls_label[i_idx[1]]

        
        #shape_=train_feats.shape
        #train_feats=train_feats.reshape(shape_[0],-1)
        scaler=StandardScaler()
        scaler.fit(train_feats)
        train_feats=scaler.transform(train_feats)#.reshape(shape_)
        
        #shape_=val_feats.shape
        #val_feats=scaler.transform(val_feats.reshape(shape_[0],-1)).reshape(shape_)
        val_feats=scaler.transform(val_feats)

        Train_dataset = ScaleDataset(train_feats,train_cls_label)
        train_loader_single = DataLoader(dataset=Train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        Val_dataset = ScaleDataset(val_feats,val_cls_label)
        val_loader_single = DataLoader(dataset=Val_dataset, batch_size=256, shuffle=False)

        # training and testing
        for epoch in range(EPOCH):
            total_loss = []

            # training
            model.train()
            for step, (feat_, cls_label_) in enumerate(train_loader_single):
                cls_label_ = cls_label_.to(device)
                feat_=feat_.to(device)
                cls_output = model(feat_)
                cls_output=cls_output.squeeze()

                loss = cls_loss_func(cls_output, cls_label_)
                total_loss.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if step % 40 == 0:
                    print(">>>>>loss:", sum(total_loss) / len(total_loss))

            # testing
            model.eval()
            with torch.no_grad():
                y_cls = []
                for step, (feat_, cls_label_) in enumerate(val_loader_single):
                    feat_=feat_.to(device) #.cuda()
                    if device.type=='cpu':
                        cls_output = model(feat_)#.data.numpy().squeeze()
                        cls_output = F.softmax(cls_output, dim=1)
                        cls_output=cls_output.data.numpy().squeeze()
                    else:
                        cls_output = model(feat_)#.cpu().data.numpy().squeeze()
                        cls_output = F.softmax(cls_output, dim=1)
                        cls_output=cls_output.cpu().data.numpy().squeeze()
                    y_cls.append(cls_output)

                # metrics for classification
                y_cls = np.concatenate(y_cls,axis=0)
                y_cls_bool=np.argmax(y_cls,axis=1)
                val_cls_label=val_cls_label.flatten() #.astype(bool) # -1 will turn into True
                
                cm_=confusion_matrix(val_cls_label,y_cls_bool)
                vc_=(cm_.trace())/(cm_.sum())
                precision_=precision_score(val_cls_label,y_cls_bool,average='weighted') #cm_[1][1]/(cm_[0][1]+cm_[1][1])
                recall_=recall_score(val_cls_label,y_cls_bool,average='weighted') #cm_[1][1]/(cm_[1][0]+cm_[1][1])
                f1_=f1_score(val_cls_label,y_cls_bool,average='weighted')
                auc_=roc_auc_score(val_cls_label,y_cls,multi_class='ovo',average='weighted',labels=[0,1,2,3,4,5])

                print('Epoch: {}, auc: {:.3f}, accuracy: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1: {:.3f}\n'.format(epoch,auc_,vc_,precision_,recall_,f1_))
                f.write('fold: {}, Epoch: {}, auc: {:.3f}, accuracy: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1: {:.3f}\n'.format(counter,epoch,auc_,vc_,precision_,recall_,f1_))

                #if f1_ > f1_best:
                if vc_ > vc_best:
                    auc_best=auc_
                    vc_best=vc_
                    precision_best=precision_
                    recall_best=recall_
                    f1_best=f1_
                    cm_best=cm_

                    # save model
                    save_path = os.path.join(res_dir, 'model_{}.pth.tar'.format(counter))
                    torch.save({'state_dict': model.state_dict()}, save_path)
                    #print('Epoch {}, f1 best: {}\n'.format(epoch,f1_best))
                    #f.write('Epoch {}, f1 best: {}\n'.format(epoch,f1_best))
                    print('Epoch {}, acc best: {}\n'.format(epoch,vc_best))
                    f.write('Epoch {}, acc best: {}\n'.format(epoch,vc_best))

        f.write('\nfold {}, auc: {:.4f}, accuracy: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}\n'.format(counter,auc_best,vc_best,
                                                                                                                precision_best,recall_best,
                                                                                                                f1_best))
        f.write('fold {}, \n[[{},{},{},{},{},{}],\n[{},{},{},{},{},{}],\n[{},{},{},{},{},{}],\n[{},{},{},{},{},{}],\n[{},{},{},{},{},{}],\n[{},{},{},{},{},{}],\n]\n\n\n'.format(counter,
        cm_best[0][0],cm_best[0][1],cm_best[0][2],cm_best[0][3],cm_best[0][4],cm_best[0][5],
        cm_best[1][0],cm_best[1][1],cm_best[1][2],cm_best[1][3],cm_best[1][4],cm_best[1][5],
        cm_best[2][0],cm_best[2][1],cm_best[2][2],cm_best[2][3],cm_best[2][4],cm_best[2][5],
        cm_best[3][0],cm_best[3][1],cm_best[3][2],cm_best[3][3],cm_best[3][4],cm_best[3][5],
        cm_best[4][0],cm_best[4][1],cm_best[4][2],cm_best[4][3],cm_best[4][4],cm_best[4][5],
        cm_best[5][0],cm_best[5][1],cm_best[5][2],cm_best[5][3],cm_best[5][4],cm_best[5][5]))

        test_precision_all.append(precision_best)
        test_recall_all.append(recall_best)
        test_f1_all.append(f1_best)
        test_auc_all.append(auc_best)
        test_vc_all.append(vc_best)
        f.close()

        counter+=1

    precision_mean=np.mean(test_precision_all)
    recall_mean=np.mean(test_recall_all)
    f1_mean=np.mean(test_f1_all)
    auc_mean=np.mean(test_auc_all)
    vc_mean=np.mean(test_vc_all)
    f = open(os.path.join(res_dir, 'result.txt'), 'a')
    f.write('total, auc: {:.4f}, accuracy: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}\n'.format(auc_mean,vc_mean,
                                                                                                            precision_mean,recall_mean,
                                                                                                            f1_mean))
    f.write('end time: {}'.format(time.ctime()))

    f.close()
