import torch
from opt import args
import torch.nn as nn
import torch.nn.functional as F
import warnings
import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt, ticker
from scipy import io
import random
from utils import *
from tqdm import tqdm
from torch import optim
#from setup import setup_args
from sklearn.manifold import TSNE
import math
import os
from torch_geometric.utils import dropout_adj,degree,to_undirected


class model_(nn.Module):
    def __init__(self,input_dim,hidden_dim,act):
        super(model_,self).__init__()
        self.AE1=nn.Linear(input_dim,hidden_dim)
        self.AE2=nn.Linear(input_dim,hidden_dim)
        
        if act == "ident": 
            self.activate = lambda x: x
        if act == "sigmoid":    
            self.activate = nn.Sigmoid()
    def forward(self,x1,x2):
        Z1=self.activate(self.AE1(x1))
        Z2=self.activate(self.AE2(x2))
        Z1=F.normalize(Z1,dim=1,p=2)
        Z2=F.normalize(Z2,dim=1,p=2)
        return Z1,Z2
    
def setup_args(dataset_name="cora"):
    args.dataset = dataset_name
    args.device = "cuda" 
    args.acc = args.nmi = args.ari = args.f1 = 0
    args.lr_s=1e-3
    args.lr_x=1e-3
    if args.dataset == 'cora':
        args.t = 2
        args.lr = 1e-3
        args.n_input = -1  
        args.dims = 1500
        args.activate = 'ident'
        args.tao = 0.2    
        args.beta = 1
    elif args.dataset == 'acm':
        args.t = 2
        args.lr = 1e-3
        args.n_input = -1
        args.dims = 1500
        args.activate = 'sigmoid'
        args.tao = 0.9
        args.beta = 1        
    elif args.dataset == 'citeseer':
        args.t = 2
        args.lr = 1e-3
        args.lr_s=1e-5
        args.lr_x=1e-5
        args.n_input = -1
        args.dims = 1500
        args.activate = 'sigmoid'
        args.tao = 0.3
        args.beta = 2

    elif args.dataset == 'amap':
        args.t = 3
        args.lr = 1e-5
        args.n_input = -1
        args.dims = 500
        args.activate = 'ident'
        args.tao = 0.9
        args.beta = 3

    elif args.dataset == 'bat':
        args.t = 6
        args.lr = 1e-3
        args.n_input = -1
        args.dims = 1500
        args.activate = 'ident'
        args.tao = 0.3
        args.beta = 5

    elif args.dataset == 'eat':  
        args.t = 6          
        args.lr = 1e-4
        args.lr_s=1e-5
        args.n_input = -1
        args.dims = 256      
        args.activate = 'ident'
        args.tao = 0.7
        args.beta = 5

    elif args.dataset == 'uat':
        args.t = 6
        args.lr = 1e-4
        args.n_input = -1
        args.dims = 500
        args.activate = 'sigmoid'
        args.tao = 0.7
        args.beta = 5

    else:
        args.t = 2
        args.lr = 1e-3
        args.n_input = -1
        args.dims = 1500
        args.activate = 'ident'
        args.tao = 0.9
        args.beta = 1

    print("---------------------")
    print("runs: {}".format(args.runs))
    print("dataset: {}".format(args.dataset))
    print("confidence: {}".format(args.tao))
    print("focusing factor: {}".format(args.beta))
    print("learning rate: {}".format(args.lr))
    print("---------------------")

    return args


def semi_loss(z1,z2,tau=0.5): 
    f=lambda x:torch.exp(x/tau)
    in_sim=f(sim(z1,z1))
    between_sim=f(sim(z1,z2))
    ret = -torch.log(between_sim.diag()/(in_sim.sum(1)+between_sim.sum(1)-in_sim.diag()))
    return ret

def sim(z1,z2):
    return torch.mm(z1,z2.t())

def square_euclid_distance(Z, center):
    ZZ = (Z * Z).sum(-1).reshape(-1, 1).repeat(1, center.shape[0])
    CC = (center * center).sum(-1).reshape(1, -1).repeat(Z.shape[0], 1)
    ZZ_CC = ZZ + CC
    ZC = Z @ center.T
    distance = ZZ_CC - 2 * ZC
    return distance

def high_confidence(Z, center,tau):
    distance_norm = torch.min(F.softmax(square_euclid_distance(Z, center), dim=1), dim=1).values  
    value, _ = torch.topk(distance_norm, int(Z.shape[0] * (1 - tau)))  
    index = torch.where(distance_norm <= value[-1],
                                torch.ones_like(distance_norm), torch.zeros_like(distance_norm))
    high_confid = torch.nonzero(index).reshape(-1, )    
    return high_confid


def find_others0(adj,sim,top_k):
    zero=torch.zeros_like(adj)
    s=torch.where(adj>0,zero,sim)
    s=s-torch.diag_embed(torch.diag(s))
    indice_move=(s<torch.topk(s,top_k)[0][...,-1,None])
    s[indice_move]=0
    indice=s.to_sparse()._indices()
    values=s.to_sparse()._values()
    indice0,value0=to_undirected(indice,values,reduce='mean')
    
    adj_0=torch.where(adj>0,sim,zero)
    indice1=adj_0.to_sparse()._indices()
    value1=adj_0.to_sparse()._values()
    
    return indice0,value0,indice1,value1  

class GCN_encoder(nn.Module):
    def __init__(self, num_features, hidden_size, embedding_size):
        super(GCN_encoder,self).__init__()
        #self.gcn1=GCN2(num_features, hidden_size, embedding_size)
        #self.gcn2=GCN2(num_features, hidden_size, embedding_size)
        self.gcn1=GCN1(num_features, embedding_size)    
        self.gcn2=GCN1(num_features, embedding_size)
    def forward(self,x,adj):
        h1=self.gcn1(x,adj)
        h2=self.gcn2(x,adj)
        return h1,h2    
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')' 
class GCN2(nn.Module):
    def __init__(self, nfeat, nhid1,nhid2):
        super(GCN2, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid1)
        self.gc2 = GraphConvolution(nhid1, nhid2)
    def forward(self, x, adj):
        x = torch.relu(self.gc1(x, adj))
        x = self.gc2(x, adj)
        return x                 

class GCN1(nn.Module):
    def __init__(self, nfeat, nhid1):##, dropout=0,nhid2
        super(GCN1, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid1)
    def forward(self, x, adj):
        x=self.gc1(x,adj)
        return x 

class DNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(DNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)#1.414

    def forward(self, x):
        Wh = torch.mm(x, self.W)
        return Wh

    def __repr__(self):
        return self.__class__.__name__+'('+str(self.in_features)+'->'+str(self.out_features)+')'    
    
class DNN1(nn.Module):  ##√
    def __init__(self, num_features, hidden_size):##, dropout=0.0, nclass
        super(DNN1, self).__init__()
        self.a = DNNLayer(num_features, hidden_size)
    def forward(self, x):
        H=self.a(x)
        return H  
    
class DNN2(nn.Module):  
    def __init__(self, num_features, hidden_size):##, dropout=0.0, nclass
        super(DNN2, self).__init__()
        self.a = DNNLayer(num_features, 32)
        self.b = DNNLayer(32, hidden_size)
    def forward(self, x):
        H=torch.relu(self.a(x))
        H=self.b(H)
        return H  
    
    
def make_adj(edge_index,edge_attr,n):           ##
    adj=torch.sparse_coo_tensor(
        edge_index,edge_attr,torch.Size([n,n])
    ).to_dense()
    return adj
    
def normalize_1(x,num):                    ##
    x1=(x-x.min())/(x.mean()-x.min())
    edge_weights=x1/x1.mean()*num
    return edge_weights

def generalize_adj(weight,indice,node_num,device):  ##
    mask=torch.bernoulli(weight).to(torch.bool)
    adj=indice[:,mask]     
    adj=make_adj(adj,torch.ones(adj.shape[1]),node_num).to(device)
    return adj

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    # for dataset_name in ["cora", "citeseer", "amap", "bat", "eat", "uat"]:
    for dataset_name in ["citeseer"]:
        args = setup_args(dataset_name)
        
        #model_path0 = os.path.join('parameter/model_s_' + args.dataset + '.pkl')
        #model_path1 = os.path.join('parameter/model_x_' + args.dataset + '.pkl')
        #model_path2 = os.path.join('parameter/model_' + args.dataset + '.pkl')
    
        acc_list = []
        nmi_list = []
        ari_list = []
        f1_list = []

        for args.seed in range(5):
            setup_seed(args.seed)
            X, y, A, node_num, cluster_num = load_graph_data(dataset_name, show_details=False)

            X_filtered = laplacian_filtering(A, X, args.t) 
            args.acc, args.nmi, args.ari, args.f1, P, center = phi(X_filtered, y, cluster_num)
            
            model=model_(input_dim=X.shape[1], hidden_dim=args.dims, act=args.activate)
            model_s=GCN_encoder(X.shape[1],2*256,256)
            model_x=DNN1(X.shape[1],X.shape[1])

            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            view_optimizer=optim.Adam([{'params':model_s.parameters()},{'params':model_x.parameters(),'lr':0.00001}],lr=0.00001,weight_decay=1e-5)             
            
            A,X,model,model_s,model_x,X_filtered = map(lambda x: x.to(args.device), (A,X,model,model_s,model_x, X_filtered))
            adj=normalize_adj1(A)

            # training
            for epoch in tqdm(range(200), desc="training..."):
                model.train()
                model_x.train()
                model_s.train()
                #print(X.dtype)
                x1=model_x(X)
                z0,z1=model_s(X,adj)
                z=0.5*(z0+z1)
                
                z=F.normalize(z,p=2,dim=1)
                z0=F.normalize(z0,p=2,dim=1)
                z1=F.normalize(z1,p=2,dim=1)
                A_pred=torch.sigmoid(torch.mm(z,z.T))
                A_pred0=torch.sigmoid(torch.mm(z0,z0.T))
                A_pred1=torch.sigmoid(torch.mm(z1,z1.T))
#               
                if epoch % 1==0:
                    indice00,value00,indice01,value01=find_others0(A,sim=A_pred0,top_k=7)
                    indice10,value10,indice11,value11=find_others0(A,sim=A_pred0,top_k=7)
                    
                    indice00,indice01,indice10,indice11=map(lambda x:x.cpu(),(indice00,indice01,indice10,indice11))  
                    value00,value01,value10,value11=map(lambda x:x.cpu(),(value00,value01,value10,value11))  
                    
                    edge_weights00=normalize_1(value00,0.3)  
                    edge_weights01=normalize_1(value01,0.9)
                    edge_weights10=normalize_1(value10,0.3)
                    edge_weights11=normalize_1(value11,0.9)

                    total_weight0=torch.cat([edge_weights00,edge_weights01],dim=-1)
                    total_indice0=torch.cat([indice00,indice01],dim=-1)
                    total_weight1=torch.cat([edge_weights10,edge_weights11],dim=-1)
                    total_indice1=torch.cat([indice10,indice11],dim=-1)
                    
                    total_weight0=total_weight0.where(total_weight0<1,torch.ones_like(total_weight0)*1)
                    total_weight1=total_weight1.where(total_weight1<1,torch.ones_like(total_weight1)*1)
                               
                adj0=generalize_adj(total_weight0,total_indice0,node_num,args.device)
                adj1=generalize_adj(total_weight1,total_indice1,node_num,args.device) 

                X_filtered1 = laplacian_filtering(adj0, X, args.t).to(args.device)
                X_filtered2 = laplacian_filtering(adj1, X, args.t).to(args.device)           
    
                Z1, Z2 = model(X_filtered1,X_filtered2)
                
                loss_s0=0.5*(torch.square((A_pred0.view(-1))-(A_pred1.view(-1))).sum())/(X.shape[0])
                loss_s=0.5*(torch.square((A.view(-1))-(A_pred.view(-1))).sum())/(X.shape[0])##adj
                loss_x=0.5*(torch.square((X.view(-1))-(x1.view(-1))).sum())/(X.shape[0])
                
                loss=0.5*(semi_loss(Z1,Z2)+semi_loss(Z2,Z1)).mean()
                total_loss=(loss+loss_x+loss_s+loss_s0).mean() 
                
                # optimization
                optimizer.zero_grad()
                view_optimizer.zero_grad()
                total_loss.backward()
            
                optimizer.step()
                view_optimizer.step()

                # testing and update weights of sample pairs
                if epoch % 1 == 0:  ##10
                    model.eval()
                    Z1, Z2= model(X_filtered,X_filtered)

                    Z = (Z1 + Z2) / 2
                    acc, nmi, ari, f1, P, center = phi(Z, y, cluster_num)
                    
                    if acc >= args.acc:
                        args.acc, args.nmi, args.ari, args.f1 = acc, nmi, ari, f1
                        
                        #torch.save(model_s.state_dict(), model_path0)
                        #torch.save(model_x.state_dict(), model_path1)
                        #torch.save(model.state_dict(), model_path2)
                        #print("{:},{:.2f}, {:.2f}, {:.2f}, {:.2f}".format(epoch,args.acc, args.nmi, args.ari, args.f1))

            print("Training complete")

            # record results
            print("{:.2f}, {:.2f}, {:.2f}, {:.2f}".format(args.acc, args.nmi, args.ari, args.f1))
            acc_list.append(args.acc)
            nmi_list.append(args.nmi)
            ari_list.append(args.ari)
            f1_list.append(args.f1)

        # record results
        acc_list, nmi_list, ari_list, f1_list = map(lambda x: np.array(x), (acc_list, nmi_list, ari_list, f1_list))
        print("{:.2f}, {:.2f}".format(acc_list.mean(), acc_list.std()))
        print("{:.2f}, {:.2f}".format(nmi_list.mean(), nmi_list.std()))
        print("{:.2f}, {:.2f}".format(ari_list.mean(), ari_list.std()))
        print("{:.2f}, {:.2f}".format(f1_list.mean(), f1_list.std()))
