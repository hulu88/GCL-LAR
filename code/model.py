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
import copy
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
        args.lr = 1e-5
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

    # other new datasets
    else:
        args.t = 2
        args.lr = 1e-5
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
def same_cluster(p,z,cluster_layer,tau,adj,y):
    index=(p==p.unsqueeze(1)).float()

    z=F.normalize(z,p=2,dim=1)
    cluster_layer=F.normalize(cluster_layer,p=2,dim=1)
    dist=torch.sum(torch.pow(z.unsqueeze(1)-cluster_layer,2),2)  
    dist=torch.softmax(dist,dim=-1)
    d=torch.min(dist,dim=-1)[0]
    threshold=torch.topk(d,int(d.shape[0]*(1-tau)))[0][-1]
    reliable=torch.nonzero(d<=threshold).reshape(-1,)  
    
    threshold1=torch.topk(d,int(d.shape[0]*(0.2)))[0][-1]
    unreliable=torch.nonzero(d>threshold1).reshape(-1,)  
    
    temp=copy.deepcopy(adj)
    mat=np.ix_(reliable,reliable)
    mask=torch.zeros_like(index)
    mask[mat]=index[mat]
    
    mat_1=np.ix_(unreliable,unreliable)
    zero=torch.zeros_like(index)
    temp[mat_1]=zero[mat_1]
    
    adj1=mask*adj        
    temp[reliable]=adj1[reliable]   
  
    return adj1


def find1thNei(adj,node):
    d=torch.squeeze(torch.nonzero(adj[node]))
    if d.dim()==0:
        d=torch.unsqueeze(d,dim=0)
    d=d.numpy()
    return d

def show_reliable(y,adj):
    num2=num=0
    for i in range(adj.shape[0]): 
        fir_nei=find1thNei(adj.cpu(),i)

        num=num+fir_nei.shape[0]  
        for j in fir_nei:
            if y[j]==y[i]:
                num2=num2+1
    print(f"num={num}")  
    print(f"num1={adj.sum()}")
    acc=num2/num
    print(f"acc={acc}")
    return acc
def nei_con_loss(z1,z2,adj,tau=0.5):  
    '''neighbor contrastive loss'''
    adj = adj - torch.diag_embed(adj.diag())  
    adj[adj > 0] = 1                  
    nei_count = torch.sum(adj, 1) * 2 + 1  
    nei_count = torch.squeeze(torch.tensor(nei_count))

    f = lambda x: torch.exp(x / tau)
    intra_view_sim = f(sim(z1, z1))
    inter_view_sim = f(sim(z1, z2))

    loss = (inter_view_sim.diag() + (intra_view_sim.mul(adj)).sum(1) + (inter_view_sim.mul(adj)).sum(1)) / (
        intra_view_sim.sum(1) + inter_view_sim.sum(1) - intra_view_sim.diag())
    loss = loss / nei_count  

    return -torch.log(loss)

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
        output = torch.mm(adj, support)#spmm-->mm
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')' 
class GCN2(nn.Module):
    def __init__(self, nfeat, nhid1,nhid2):##,
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
    
class DNN1(nn.Module):
    def __init__(self, num_features, hidden_size):##, dropout=0.0, nclassv
        super(DNN1, self).__init__()
        self.a = DNNLayer(num_features, hidden_size)
    def forward(self, x):
        H=self.a(x)
        return H  
    
def make_adj(edge_index,edge_attr,n):
    adj=torch.sparse_coo_tensor(
        edge_index,edge_attr,torch.Size([n,n])
    ).to_dense()
    return adj
    
def normalize_1(x,num):
    x1=(x-x.min())/(x.mean()-x.min())
    edge_weights=x1/x1.mean()*num
    return edge_weights

def generalize_adj(weight,indice,node_num,device):
    mask=torch.bernoulli(weight).to(torch.bool)
    adj=indice[:,mask]     
    adj=make_adj(adj,torch.ones(adj.shape[1]),node_num).to(device)
    return adj

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    pretrain_path0="parameter/model_s_acm_1.pkl"
    pretrain_path1="parameter/model_x_acm_1.pkl"
    pretrain_path2="parameter/test_model_acm_1.pkl"

    for dataset_name in ["acm"]:
        args = setup_args(dataset_name)

        acc_list = []
        nmi_list = []
        ari_list = []
        f1_list = []

        for args.seed in [0]:
            setup_seed(args.seed)
            X, y, A, node_num, cluster_num = load_graph_data(dataset_name, show_details=False)
            
            model=model_(input_dim=X.shape[1], hidden_dim=args.dims, act=args.activate)
            model_s=GCN_encoder(X.shape[1],2*256,256)
            model_x=DNN1(X.shape[1],X.shape[1])

            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            
            model_s.load_state_dict(torch.load(pretrain_path0,map_location='cpu'))
            model_x.load_state_dict(torch.load(pretrain_path1,map_location='cpu'))
            model.load_state_dict(torch.load(pretrain_path2,map_location='cpu'))
                
            X_filtered = laplacian_filtering(A, X, args.t) 
            
            with torch.no_grad():
                Z1,Z2=model(X_filtered,X_filtered)
                Z=(Z1+Z2)/2            
            
            args.acc, args.nmi, args.ari, args.f1, P, center = phi(Z, y, cluster_num)            

            P1=torch.tensor(P)
            AA=same_cluster(P1,Z,center,0.7,A,y).to(args.device) 
            
            
            A,X,model,model_s,model_x,X_filtered = map(lambda x: x.to(args.device), (A,X,model,model_s,model_x, X_filtered))
            adj=normalize_adj1(A)

            # training
            for epoch in range(200):
                model.train()
                model_x.train()
                model_s.train()
                x1=model_x(X)
                z0,z1=model_s(X,adj)
                z=0.5*(z0+z1)
                
                z=F.normalize(z,p=2,dim=1)
                z0=F.normalize(z0,p=2,dim=1)
                z1=F.normalize(z1,p=2,dim=1)
                A_pred=torch.sigmoid(torch.mm(z,z.T))
                A_pred0=torch.sigmoid(torch.mm(z0,z0.T))
                A_pred1=torch.sigmoid(torch.mm(z1,z1.T))
                
                if epoch % 5==0:
                    indice00,value00,indice01,value01=find_others0(A,sim=A_pred0,top_k=1)
                    indice10,value10,indice11,value11=find_others0(A,sim=A_pred0,top_k=1)
                    
                    indice00,indice01,indice10,indice11=map(lambda x:x.cpu(),(indice00,indice01,indice10,indice11))  
                    value00,value01,value10,value11=map(lambda x:x.cpu(),(value00,value01,value10,value11))  
                    
                    edge_weights00=normalize_1(value00,0.3)  
                    edge_weights01=normalize_1(value01,0.3)
                    edge_weights10=normalize_1(value10,0.3)
                    edge_weights11=normalize_1(value11,0.3)

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
                loss=0.5*(nei_con_loss(Z1,Z2,AA)+nei_con_loss(Z2,Z1,AA)).mean()  
        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if epoch % 1 == 0:  
                    model.eval()
                    Z1, Z2= model(X_filtered,X_filtered)

                    Z = (Z1 + Z2) / 2
                    acc, nmi, ari, f1, P, center = phi(Z, y, cluster_num)

                    if epoch % 1==0:
                        P1=torch.tensor(P)
                        AA=same_cluster(P1,Z.detach().cpu(),center,0.7,A.detach().cpu(),y).to(args.device)
                    if acc >= args.acc:
                        args.acc, args.nmi, args.ari, args.f1 = acc, nmi, ari, f1                        
                        print("{:},{:.2f}, {:.2f}, {:.2f}, {:.2f}".format(epoch,args.acc, args.nmi, args.ari, args.f1))

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

