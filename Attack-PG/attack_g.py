# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 15:56:52 2023

@author: JetLee
"""

import torch
import numpy as np
import time
from ExplanationEvaluation.datasets.dataset_loaders import load_dataset
import random
import os
from ExplanationEvaluation.models.GNN_paper import NodeGCN,GraphGCN
from ExplanationEvaluation.utils.graph import index_edge
from ExplanationEvaluation.utils.plotting import plot, plot_2,plot_3
import pickle as pkl
from ratio_test import likelyhood

dataset = "REDDIT-BINARY"
#options: "mutag","REDDIT-BINARY"
attack = "kill-hot"
#options: "random";"kill-hot";"loss";"deduction"

task = 'graph'
if dataset == "mutag":
    model = GraphGCN(14, 2)
elif dataset == "REDDIT-BINARY":
    model = GraphGCN(11, 2)
    
o_graphs, features, labels, _, _, _ = load_dataset(dataset)
o_features = features
labels = torch.tensor(labels)
    
path = "./checkpoints/GNN/{}/best_model".format(dataset)
checkpoint = torch.load(path)
model.load_state_dict(checkpoint['model_state_dict'])

from ExplanationEvaluation.explainers.PGExplainer import PGExplainer
from ExplanationEvaluation.explainers.PGAttacker import Deduction_PGAttacker,Loss_PGAttacker


beta = 0.7
gamma = 0.7

k = 10
xi = 2
N=4

fools=[]
ex_time=[]
hot_time=[]
cold_time=[]

f_labels = []
for indx in  range(0,40,1):
    maxfool=0
    graphs = torch.tensor(np.array([o_graphs[indx]]))
    features = torch.tensor(np.array([o_features[indx]]))
    N=features.shape[1]
    extime_s = time.time()
    explainer = PGExplainer(model, graphs, features, task,epochs=50)
    f_labels.append(labels[indx])
    explainer.prepare([0])
        

    idx = indx
    graph, expl = explainer.explain(0)
    if len(expl)<4*k:
        continue
    extime_e = time.time()
    ex_time.append(extime_e-extime_s)
        
    adj = np.zeros((N,N))
    mask = np.zeros((N,N))
    degree = np.zeros((N))
    
    
    for i in range(len(graph[0])):
        k1 = graph[0,i]
        k2 = graph[1,i]
        adj[k1][k2]=1
        degree[k1]+=1
        mask[k1][k2]=expl[i]

    bi_expl = np.zeros(expl.shape)
    for i in range(len(graph[0])):
        k1 = graph[0,i]
        k2 = graph[1,i]
        if k1<=k2:
            bi_expl[i] = (mask[k1][k2]+mask[k2][k1])/2
        else:
            bi_expl[i] = 0
    expl = torch.tensor(bi_expl)
    
    
    o_id = np.flip(np.argsort(np.array(expl.detach()).reshape(-1))[-k:])
    o_bid = []
    
    o_bid=[]
    for j in range(len(o_id)):
        i = index_edge(graph,[graph[0,o_id[j]],graph[1,o_id[j]]])
        o_bid.append(i)
        if  graph[1,o_id[j]]!=graph[0,o_id[j]]:
            i = index_edge(graph,[graph[1,o_id[j]],graph[0,o_id[j]]])
            o_bid.append(i)
        
    r_graphs = [[],[]]
    for i in range(N):
        for j in range(i,N):
            if adj[i][j]==0:
                r_graphs[0].append(i)
                r_graphs[1].append(j)
                r_graphs[0].append(j)
                r_graphs[1].append(i)
                
    for i in range(len(o_bid)):
        r_graphs[0].append(graph[0,o_bid[i]])
        r_graphs[1].append(graph[1,o_bid[i]])
        
    r_graphs = torch.tensor(np.array([r_graphs]))
    
    
    
    if attack=="deduction" or attack == "loss":
        
        o_fliter = np.ones(expl.shape)
        o_bias = np.zeros(expl.shape)
        o_fliter[o_bid] = 0
        o_bias[o_bid] = 1
        hot_s = time.time()
        
        if attack == "deduction":
            attacker=Deduction_PGAttacker(model, graphs, features, task,beta=beta,N=N,r_epochs=10)
        else:
            attacker=Loss_PGAttacker(model, graphs, features, task,gamma=gamma,r_epochs=10)
            
        h_graph,hot_mask = attacker.learn_deletion([0],o_fliter,o_bias)
        h_graph = np.array(h_graph.detach())
        hot_mask = np.array(hot_mask.detach())
        
        add_mask = np.zeros((N,N))
        for i in range(hot_mask.shape[0]):
            add_mask[h_graph[0,i]][h_graph[1,i]] = hot_mask[i]
        
        for i in range(hot_mask.shape[0]):
            if h_graph[0,i]<=h_graph[1,i]:
                hot_mask[i] = add_mask[h_graph[0,i]][h_graph[1,i]]+add_mask[h_graph[1,i]][h_graph[0,i]]
            else:
                hot_mask[i] = 0
                
        hot_id = np.flip(np.argsort((hot_mask).reshape(-1)))
        hot_e = time.time()
        hot_time.append(hot_e-hot_s)
    
    
        r_graph=r_graphs[0].clone().detach()
    
        if attack == "deduction":
            attacker=Deduction_PGAttacker(model, r_graphs, features, task,beta=beta,N=N,r_epochs=2)
        else:
            attacker=Loss_PGAttacker(model, r_graphs, features, task,gamma=gamma,r_epochs=10)
            
        r_o_fliter = np.ones(r_graph.shape[1])
        r_o_bias = np.zeros(r_graph.shape[1])
        r_o_bid = []
        for i in range(r_graph.shape[1]):
            
            for j in range(len(o_bid)):
            
                if r_graph[0,i]==graph[0,o_bid[j]] and r_graph[1,i]==graph[1,o_bid[j]]:
                    r_o_fliter[i]=0
                    r_o_bias[i]=1
                    r_o_bid.append(i)
                    
        cold_s = time.time()
        c_graph,cold_mask = attacker.learn_addition([0],r_o_fliter,r_o_bias)
        c_graph = np.array(c_graph.detach())
        cold_mask = np.array(cold_mask.detach())
        
        del_mask = np.zeros((N,N))
        for i in range(cold_mask.shape[0]):
            del_mask[c_graph[0,i]][c_graph[1,i]] = cold_mask[i]
        
        for i in range(cold_mask.shape[0]):
            if c_graph[0,i]<=c_graph[1,i]:
                cold_mask[i] = del_mask[c_graph[0,i]][c_graph[1,i]]+del_mask[c_graph[1,i]][c_graph[0,i]]
            else:
                cold_mask[i] = 0
                
        cold_id = np.flip(np.argsort((cold_mask).reshape(-1)))
        cold_e = time.time()
        cold_time.append(cold_e-cold_s)
    
    
        
        for t in range(xi+1):
            rec_d = np.ones(expl.shape)
            rec_a = np.ones(r_graph.shape[1])
            new_d = degree
            cnt2=0
            hots = []
            for i in range(len(hot_id)):
                a = hot_id[i]
                pair = np.array(graph.T[a])
                npair = pair.copy()
                npair[0]=pair[1]
                npair[1]=pair[0]
                b = index_edge(graph,pair)
                if cnt2>=t*2:
                    break
                if np.isin([hot_id[i]],o_bid)==True or rec_d[hot_id[i]]==0:
                    continue
                
                rec_d[a]=0
                rec_d[b]=0
                
                n1 = pair[0]
                n2 = pair[1]
                new_d[n1]-=1
                if n1!=n2:
                    new_d[n2]-=1
            
                a1 = index_edge(graphs[0],pair)
                b1 = index_edge(graphs[0],npair)
                hots.append(a1)
                if a1 != b1:
                    hots.append(b1)
                cnt2+=2
            new_graphs=  np.delete(graphs[0].permute(1,0),hots,axis =0)
            for i in range(len(cold_id)):
                if cnt2>=2*xi:
                    break
                if np.isin([cold_id[i]],r_o_bid)==True or rec_a[cold_id[i]]==0:
                    continue
                a = cold_id[i]
                pair = r_graph.T[a]
                new_graphs = torch.cat((new_graphs,torch.tensor([[pair[0],pair[1]]])))
                if pair[0]!=pair[1]:
                    new_graphs = torch.cat((new_graphs,torch.tensor([[pair[1],pair[0]]])))
                    
                pair[1] = pair[0]+pair[1]
                pair[0] = pair[1]-pair[0]
                pair[1] = pair[1]-pair[0]
                b = index_edge(r_graph,pair)
                rec_a[a]=0
                rec_a[b]=0
                
                n1 = pair[0]
                n2 = pair[1]
                new_d[n1]+=1
                if n1!=n2:
                    new_d[n2]+=1
                
                cnt2+=2
                #print(new_graphs.shape)
            new_graphs= torch.tensor([np.array(new_graphs.permute(1,0))])
            ls = likelyhood(degree, new_d)
            predict_before = np.argmax(np.array(model(features,graphs[0]).detach()))
            predict_after = np.argmax(np.array(model(features,new_graphs[0]).detach()))
            if ls >0.000157 or predict_before!=predict_after:
                continue
        
            aexplainer = explainer
            aexplainer.graphs=new_graphs
            n_graph, n_expl = aexplainer.explain(0)
            n_expl = np.array(n_expl.detach())
            bi_n_expl = n_expl.copy()
            for i in range(n_expl.shape[0]):
                r_pair = [n_graph[1,i],n_graph[0,i]]
                if n_graph[0,i]<=n_graph[1,i]:
                    n_expl[i] += bi_n_expl[index_edge(n_graph, r_pair)]
                else:
                    n_expl[i] = 0
            n_o_id=[]
            for i in range(len(n_expl)):
                for j in range(len(o_id)):
                    if n_graph[0,i]==graph[0,o_id[j]] and n_graph[1,i]==graph[1,o_id[j]]:
                        n_o_id.append(i)
                        
            now_id = np.flip(np.argsort((n_expl).reshape(-1))[-k:])
            cnt2=0
            fool=0
            real_fool=0
            for i in range(len(now_id)):
                if np.isin([now_id[i]],n_o_id)==False :
                    fool+=1
                    
            n_graph = np.array(n_graph.detach())
            print(fool)
            maxfool = max(maxfool, fool)
        fools.append(maxfool)
        
    elif attack=="kill-hot":
            hot_id = np.flip(np.argsort(np.array(expl.detach()).reshape(-1)))
        
            rec = np.ones(expl.shape)
    
            new_d = degree
            cnt2=0
            hots = []
            for i in range(len(hot_id)):
                if cnt2>=2*xi:
                    break
                if np.isin([hot_id[i]],o_bid)==True or rec[hot_id[i]]==0:
                    continue
                a = hot_id[i]
                pair = graph.T[a]
                #print(pair)
                pair[1] = pair[0]+pair[1]
                pair[0] = pair[1]-pair[0]
                pair[1] = pair[1]-pair[0]
                #print(pair)
                b = index_edge(graph,pair)
                rec[a]=0
                rec[b]=0
            
                n1 = pair[0]
                n2 = pair[1]
                new_d[n1]-=1
                if n1!=n2:
                    new_d[n2]-=1
            
                hots.append(a)
                if a!=b:
                    
                    hots.append(b)
                cnt2+=2
            #print(graphs.permute(1,0).shape)
            new_graphs=  np.delete(graphs[0].permute(1,0),hots,axis =0)
            new_graphs= [new_graphs.permute(1,0)]
            ls = likelyhood(degree, new_d)
            predict_before = np.argmax(np.array(model(features,graphs[0]).detach()))
            predict_after = np.argmax(np.array(model(features,new_graphs[0]).detach()))
            if ls>0.000157 or predict_before!=predict_after:
                continue
            
            aexplainer = explainer
            aexplainer.graphs=new_graphs
            n_graph, n_expl = aexplainer.explain(0)
            n_expl = np.array(n_expl.detach())
            bi_n_expl = n_expl.copy()
            for i in range(n_expl.shape[0]):
                r_pair = [n_graph[1,i],n_graph[0,i]]
                if n_graph[0,i]<=n_graph[1,i]:
                    n_expl[i] += bi_n_expl[index_edge(n_graph, r_pair)]
                else:
                    n_expl[i] = 0
            n_o_id=[]
            for i in range(len(n_expl)):
                for j in range(len(o_id)):
                    if n_graph[0,i]==graph[0,o_id[j]] and n_graph[1,i]==graph[1,o_id[j]]:
                        n_o_id.append(i)
            now_id = np.flip(np.argsort((n_expl).reshape(-1))[-k:])
            cnt2=0
            fool=0
            real_fool=0
            for i in range(len(now_id)):
                if np.isin([now_id[i]],n_o_id)==False :
                    if i<k:
                        fool+=1
                    else:
                        break
            n_graph = np.array(n_graph.detach())
            print(fool)
        
            fools.append(fool)
    
    elif attack=="random":
        new_graphs = graphs
        new_d = degree
        cb =0 
        cnt2=0
        rec = np.zeros((N,N))
        add = []
        delete = []
        while cnt2<2*xi:
            
            a = random.randint(0,N-2)
            b = random.randint(a+1,N-1)
            if cb>=40:
                break
            flag=0
            for i in range(len(o_id)):
                if graphs[0,0,i]==a and graphs[0,1,i]==b:
                    flag=1
                    break;
                if graphs[0,0,i]==b and graphs[0,1,i]==a:
                    flag=1
                    break
            if flag ==1:
                cb+=1
                continue
            if rec[a][b]==1 :#or adj[a][b]==0:
                continue
            if adj[a][b]==0:
                add.append([a,b])
                add.append([b,a])
                new_d[a]+=1
                new_d[b]+=1
            else:
                new_d[a]-=1
                new_d[b]-=1
                for i in range(len(graph)):
                    if a==graph[0,i] and b == graph[1,i]:
                        delete.append(i)
                    if a==graph[1,i] and b == graph[0,i]:
                        delete.append(i)
            rec[a][b]==1
            cnt2+=2
        new_graphs=  np.delete(graphs[0].permute(1,0),delete,axis =0)
        new_graphs = torch.cat((new_graphs,torch.tensor(add,dtype = torch.int32)))
        new_graphs = [new_graphs.permute(1,0)]
        ls = likelyhood(degree, new_d)
        predict_before = np.argmax(np.array(model(features,graphs[0]).detach()))
        predict_after = np.argmax(np.array(model(features,new_graphs[0]).detach()))
        if ls>0.000157 or predict_before!=predict_after:
            continue
            
            
        aexplainer = explainer
        aexplainer.graphs=new_graphs
        n_graph, n_expl = aexplainer.explain(0)
        n_expl = np.array(n_expl.detach())
        bi_n_expl = n_expl.copy()
        for i in range(n_expl.shape[0]):
            r_pair = [n_graph[1,i],n_graph[0,i]]
            if n_graph[0,i]<=n_graph[1,i]:
                n_expl[i] += bi_n_expl[index_edge(n_graph, r_pair)]
            else:
                n_expl[i] = 0
                
        n_o_id=[]
        for i in range(len(n_expl)):
            for j in range(len(o_id)):
                if n_graph[0,i]==graph[0,o_id[j]] and n_graph[1,i]==graph[1,o_id[j]]:
                    n_o_id.append(i)
                
        now_id = np.flip(np.argsort((n_expl).reshape(-1))[-k:])
        cnt2=0
        fool=0
        real_fool=0
        for i in range(len(now_id)):
            if np.isin([now_id[i]],n_o_id)==False :
                if i<k:
                    fool+=1
                else:
                    break
        n_graph = np.array(n_graph.detach())
        print(fool)
    
        fools.append(fool)
    print(fools)
print(sum(fools)/len(fools))