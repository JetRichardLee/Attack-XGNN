# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 15:56:52 2023

@author: JetLee
"""
import os

import torch
import numpy as np
from ExplanationEvaluation.datasets.dataset_loaders import load_dataset
import random
import torch_geometric as ptgeom

from ExplanationEvaluation.models.GNN_paper import NodeGCN,NodeGCNL
from ExplanationEvaluation.utils.graph import index_edge
from ogb.nodeproppred import PygNodePropPredDataset

from ratio_test import likelyhood
import time

dataset=  'products'
#options: 'syn1';'syn2';'syn3';'products'
attack = "random"
#options: "random";"kill-hot";"loss";"deduction"

#the model and test cases applied
if dataset =='products':
    task = 'ogb_node'
    model = NodeGCNL(100, 47)
    test_indices = range(0,400,10)
else:
    task = 'node'
    if dataset =="syn1":
        model = NodeGCN(10, 4)
        test_indices = range(300,700,5)
        #not starting from indice zero, where many nodes with label 0 locate 
        #because 0 label node has no meaningful explanation
    elif dataset =="syn2":
        model = NodeGCN(10, 8)
        test_indices = range(400,560,1)
    elif dataset =="syn3":
        model = NodeGCN(10, 2)
        test_indices = range(0,320,2)


graphs, features, labels, _, _, _ = load_dataset(dataset)
graphs = torch.tensor(graphs)
features = torch.tensor(features)
labels = torch.tensor(labels)


path = "./checkpoints/GNN/{}/best_model".format(dataset)
checkpoint = torch.load(path)
model.load_state_dict(checkpoint['model_state_dict'])

from ExplanationEvaluation.explainers.PGExplainer import PGExplainer
from ExplanationEvaluation.explainers.PGAttacker import Deduction_PGAttacker,Loss_PGAttacker


#tuning parameter setting
beta = 0.7
gamma = 0.7 
r_epochs=40

k = 25
xi = 10

rate =[]
fools=[]
ex_time=[]
hot_time=[]
cold_time=[]


for indx in test_indices:
    print(indx)
    maxfool=0
    sumfool =0 
    #extime_s = time.time()
    #Initialize an explainer to explain the graph first
    explainer = PGExplainer(model, graphs, features, task)
    explainer.prepare([indx])

    idx = indx
    graph, expl = explainer.explain(idx)
    
    #extime_e = time.time()
    #print("explain:")
    #print(extime_e-extime_s)
    
    #ex_time.append(extime_e-extime_s)
    
    #if the subgraph around the node is too small to pick an explanation, omit it
    if len(expl)<k*2:
            continue
    
    
    #rate.append(len(graph[0]))
    node = []
    ansexl=None
    ansgra=None
    
    
    
    print("mapping node")
    #mapping the node in the graph to the subgraph
    maps = np.zeros((len(features)))
    N=len(features)
    maps = np.zeros((len(features)),dtype=np.int32)
    maps[graph[0,0]]=1
    node_index = graph[0]
    node_index = np.sort(node_index)
    node.append(node_index[0])
    for i in range(1,len(graph[0])):
        if node_index[i]!=node_index[i-1]:
            node.append(node_index[i])
            maps[node_index[i]]=len(node)

        
    print("summary graph")
    #summary the information of the subgraph
    adj = np.zeros((len(node),len(node)))
    mask = np.zeros((len(node),len(node)))
    degree = np.zeros((len(node)))
            
    for i in range(len(graph[0])):
        adj[maps[graph[0,i]]-1][maps[graph[1,i]]-1]=1
        degree[maps[graph[0,i]]-1]+=1           
        mask[maps[graph[0,i]]-1][maps[graph[1,i]]-1]=expl[i]
            
    #the expl is initally for directed edges. Need to transfer to undirected ones. 
    bi_expl = np.zeros(expl.shape)
    for i in range(len(graph[0])):
        if graph[0,i]<=graph[1,i]:
            bi_expl[i] = mask[maps[graph[0,i]]-1][maps[graph[1,i]]-1]+mask[maps[graph[1,i]]-1][maps[graph[0,i]]-1]
        else:
            bi_expl[i]=0
    expl = torch.tensor(bi_expl)
    
    #select the top k edges as the explanatory edges Es
    o_id = np.flip(np.argsort(np.array(expl.detach()).reshape(-1))[-k:])
        
    o_bid=[]
    for j in range(len(o_id)):
        i = index_edge(graph,[graph[0,o_id[j]],graph[1,o_id[j]]])
        o_bid.append(i)
        i = index_edge(graph,[graph[1,o_id[j]],graph[0,o_id[j]]])
        o_bid.append(i)
    
    #construct a complementary graph EA for the E
    if attack=="deduction" or attack == "loss":
        
        r_graphs = [[],[]]
        #note that in the implement, we also added existent edges connected to the index Node v into the EA
        #This approximation is to ensure the connectivity of the neighborhood on v
        #and we won't consider adding them in the addition step still
        
        print("regraph")
        #if the subgraph is too large, we only add nonexistent edges involved with the index Node v
        if len(node)>=1500:
            for i in range(len(node)):
                if i%1000==0:
                    print("{}/{}".format(i, len(node)))
                if node[i] != indx:
                    r_graphs[0].append(node[i])
                    r_graphs[1].append(indx)
                    r_graphs[0].append(indx)
                    r_graphs[1].append(node[i])
        else:
            for i in range(len(node)):
                if i%1000==0:
                    print("{}/{}".format(i, len(node)))
                for j in range(i+1,len(node)):
                    if adj[i][j]==0 or node[i]== indx or node[j] == indx:
                        r_graphs[0].append(node[i])
                        r_graphs[1].append(node[j])
                        r_graphs[0].append(node[j])
                        r_graphs[1].append(node[i])
    
        for i in range(len(o_id)):
            r_graphs[0].append(graph[0,o_id[i]])
            r_graphs[1].append(graph[1,o_id[i]])
            r_graphs[0].append(graph[1,o_id[i]])
            r_graphs[1].append(graph[0,o_id[i]])
    
        r_graphs = torch.tensor(np.array(r_graphs))
        
    
        #we train the hot_mask(deletion mask) MD
        p_del_s = time.time()
        print("preparing deletion")
        #setting the filter matrix f and bias matrix b 
        #o_bid is the index of Es in directed version (containing 2*k directed edges)
        o_fliter = np.ones(expl.shape)
        o_bias = np.zeros(expl.shape)
        o_bid=[]
        for j in range(len(o_id)):
            i = index_edge(graph,[graph[0,o_id[j]],graph[1,o_id[j]]])
            o_fliter[i]=0
            o_bias[i]=1
            o_bid.append(i)
            i = index_edge(graph,[graph[1,o_id[j]],graph[0,o_id[j]]])
            o_fliter[i]=0
            o_bias[i]=1
            o_bid.append(i)
        
        #initialize an attacker and learn the mask
        if attack == "deduction":
            attacker=Deduction_PGAttacker(model, graphs, features, task,beta=beta)
        else:
            attacker=Loss_PGAttacker(model, graphs, features, task,gamma=gamma)
        h_graph,hot_mask = attacker.learn_deletion([indx],o_fliter,o_bias)
        h_graph = np.array(h_graph.detach())
        hot_mask = np.array(hot_mask.detach())
        
        add_mask = np.zeros((len(node),len(node)))
        for i in range(hot_mask.shape[0]):
            add_mask[maps[h_graph[0,i]]-1][maps[h_graph[1,i]]-1] = hot_mask[i]
        
        for i in range(hot_mask.shape[0]):
            if h_graph[0,i]<=h_graph[1,i]:
                hot_mask[i] = add_mask[maps[h_graph[0,i]]-1][maps[h_graph[1,i]]-1]+add_mask[maps[h_graph[1,i]]-1][maps[h_graph[0,i]]-1]
            else:
                hot_mask[i] = 0
        
        #print(hot_mask)
        #store the deletion candidates sorted by the MD
        #we store candidates much more than k in case there's any forbidden edge like ones in Es,
        #although theorically they won't appear here
        hot_id = np.flip(np.argsort((hot_mask)))
        p_del_e = time.time()
    
        print("prepared deletion")
        #print(p_del_e-p_del_s)
        #hot_time.append(p_del_e-p_del_s)
    
    
        #we train the cold_mask(addition mask) MA
        #steps are the same with the MD training
        print("preparing addition")
        cold_s = time.time()
        if task == 'node':
            r_graph = ptgeom.utils.k_hop_subgraph([indx], 3, r_graphs)[1]
        else:
            r_graph = ptgeom.utils.k_hop_subgraph([indx], 2, r_graphs)[1]
        
        if attack == "deduction":
            attacker=Deduction_PGAttacker(model, r_graphs, features, task,beta=beta)
        else:
            attacker=Loss_PGAttacker(model, r_graphs, features, task,gamma=gamma)
            
        r_o_fliter = np.ones(r_graph.shape[1])
        r_o_bias = np.zeros(r_graph.shape[1])
        r_o_bid = []
        
        for j in range(len(o_id)):
            
            i = index_edge(r_graph,[graph[0,o_id[j]],graph[1,o_id[j]]])
            r_o_fliter[i]=0
            r_o_bias[i]=1
            r_o_bid.append(i)
            i = index_edge(r_graph,[graph[1,o_id[j]],graph[0,o_id[j]]])
            r_o_fliter[i]=0
            r_o_bias[i]=1
            r_o_bid.append(i)
            
        c_graph,cold_mask = attacker.learn_addition([indx],r_o_fliter,r_o_bias)
        c_graph = np.array(c_graph.detach())
        cold_mask = np.array(cold_mask.detach())
        for i in range(cold_mask.shape[0]):
            if i%2==0:
                cold_mask[i]=cold_mask[i]+cold_mask[i+1]
                if c_graph[0,i]>c_graph[1,i]:
                    cold_mask[i+1]=cold_mask[i]
                    cold_mask[i]=0
                else:
                    cold_mask[i+1]=0
                    
        cold_id = np.flip(np.argsort((cold_mask).reshape(-1)))
        cold_e = time.time()
        print("prepared addition")
        #print(cold_e-cold_s)
        cold_time.append(cold_e-cold_s)
    
        #use the numeration to find the best combination on Eadd and Edel
        for t in range(xi+1):
            #t to represent the budget for deletion
            deletion_rec = np.ones(expl.shape)
            addition_rec = np.ones(r_graph.shape[1])
            new_degree = degree
            cnt2=0
            hots = []
            for i in range(len(hot_id)):
                if cnt2>=t*2:
                    break
                #in o_bid-> this edge is in Es, so cannot delete
                #deletion_rec==0 -> this edge is already deleted
                if np.isin([hot_id[i]],o_bid)==True or deletion_rec[hot_id[i]]==0:
                    continue
                a = hot_id[i]
                #find the other directed edge of this candidate
                pair = np.array(graph.T[a])
                npair = pair.copy()
                npair[0] =pair[1]
                npair[1] =pair[0]
            #print(pai]
            #print(pair)
                b = index_edge(graph,npair)
                #delete both of them
                deletion_rec[a]=0
                deletion_rec[b]=0
            
                n1 = np.where(np.array(node,dtype = np.int32) == int(pair[0]))[0][0]
                n2 = np.where(np.array(node,dtype = np.int32) == int(pair[1]))[0][0]
                
                new_degree[n1]-=1
                new_degree[n2]-=1
                a1 = index_edge(graphs,pair)
                b1 = index_edge(graphs,npair)
                hots.append(a1)
                hots.append(b1)
                cnt2+=2
            new_graphs=  np.delete(graphs.permute(1,0),hots,axis =0)
            
            for i in range(len(cold_id)):
                # ensure the budget limit
                if cnt2>=xi*2:
                    break
                #in r_o_bid-> this edge is in Es, so cannot modified
                #addition_rec==0 -> this edge is already added
                if np.isin([cold_id[i]],r_o_bid)==True or addition_rec[cold_id[i]]==0:
                    continue
                
                a = cold_id[i]
                pair = r_graph.T[a]
                n1 = np.where(np.array(node,dtype = np.int32) == int(pair[0]))[0][0]
                n2 = np.where(np.array(node,dtype = np.int32) == int(pair[1]))[0][0]
                #adj[n1,n2]==1 or adj[n2,n1]==1 -> this edge already exists
                if adj[n1,n2]==1 or adj[n2,n1]==1:
                    continue
                #add this edge into the new graph
                new_graphs = torch.cat((new_graphs,torch.tensor([[pair[0],pair[1]]])))
                new_graphs = torch.cat((new_graphs,torch.tensor([[pair[1],pair[0]]])))
                
                pair[1] = pair[0]+pair[1]
                pair[0] = pair[1]-pair[0]
                pair[1] = pair[1]-pair[0]
                b = index_edge(r_graph,pair)
                
                new_degree[n1]+=1
                new_degree[n2]+=1
                
                addition_rec[a]=0
                addition_rec[b]=0
                cnt2+=2
            new_graphs= new_graphs.permute(1,0)
            
            
            #ensure the likelyhood test is passed and the prediction on the attacked graph is hold the same
            ls = likelyhood(degree,new_degree,5)
            predict_before = np.argmax(np.array(model(features,graphs).detach()[indx]))
            predict_after = np.argmax(np.array(model(features,new_graphs).detach()[indx]))
            if ls >0.000157 or predict_before!=predict_after:
                continue
        
            #there is two scenario for XGNN attack in PG Explainer: 
                #1. the pg explainer learned again on the attacked graph
                #2. only the graph to be explained is attacked and the weight of the explainer is hold the same
            # we conduct our experiment in the scenario 2, "attack graph data while or after the explainer training"
            #1 
            #aexplainer=PGExplainer(model, new_graphs, features, task)
            #aexplainer.prepare([indx])
            #2
            aexplainer = explainer
            aexplainer.graphs = new_graphs
            
            #obtained the attacked mask
            n_graph, n_expl = aexplainer.explain(idx)
            
            n_graph = np.array(n_graph.detach())
            t_n_expl = n_expl
            t_n_graph = n_graph
            n_expl = np.array(n_expl.detach())
            
            #mapping to the undirected edge version
            bi_n_expl = n_expl.copy()
            for i in range(n_expl.shape[0]):
                r_pair = [n_graph[1,i],n_graph[0,i]]
                if n_graph[0,i]<=n_graph[1,i]:
                    n_expl[i] += bi_n_expl[index_edge(n_graph, r_pair)]
                else:
                    n_expl[i] = 0
            #mapping Es's index into the attacked subgraph
            #note the attacked subgraph may not contain all of them since connectivity may be changed
            n_o_id=[]
            for i in range(n_expl.shape[0]):
                for j in range(len(o_id)):
                    if n_graph[0,i]==graph[0,o_id[j]] and n_graph[1,i]==graph[1,o_id[j]]:
                        n_o_id.append(i)
                        
            #count the top k edges changed by index
            now_id = np.flip(np.argsort((n_expl).reshape(-1))[-k:])
            fool=0
            for i in range(len(now_id)):
                if np.isin([now_id[i]],n_o_id)==False :
                    fool+=1
                    
            #rec the best attack strategy
            if fool >= maxfool:
                ansexl = t_n_expl
                ansgra = t_n_graph
            maxfool = max(maxfool, fool)
            
        fools.append(maxfool)
        
    elif attack=="kill-hot":
        o_bid=[]
        for j in range(len(o_id)):
            i = index_edge(graph,[graph[0,o_id[j]],graph[1,o_id[j]]])
            o_bid.append(i)
            i = index_edge(graph,[graph[1,o_id[j]],graph[0,o_id[j]]])
            o_bid.append(i)
        hot_id = np.flip(np.argsort(np.array(expl.detach()).reshape(-1)))
        
        del_rec = np.ones(expl.shape)
    
        new_degree = degree
        cnt2=0
        hots = []
        for i in range(len(hot_id)):
            if cnt2>=xi*2:
                break
            if np.isin([hot_id[i]],o_bid)==True or del_rec[hot_id[i]]==0:
                continue
            a = hot_id[i]
            pair1 = graph.T[a]
            pair2 = [pair1[1],pair1[0]]
            b = index_edge(graph,pair2)
            del_rec[a]=0
            del_rec[b]=0
                
            n1 = np.where(node == pair1[0])
            n2 = np.where(node == pair1[1])
            new_degree[n1]-=1
            new_degree[n2]-=1
            
            a1 = index_edge(graphs,pair1)
            b1 = index_edge(graphs,pair2)
            hots.append(a1)
            hots.append(b1)
            cnt2+=2
        new_graphs=  np.delete(graphs.permute(1,0),hots,axis =0)
        new_graphs= new_graphs.permute(1,0)
        ls = likelyhood(degree,new_degree)
        predict_before = np.argmax(np.array(model(features,graphs).detach()[indx]))
        predict_after = np.argmax(np.array(model(features,new_graphs).detach()[indx]))
        
        if ls >0.000157 or predict_before!=predict_after:
            fools.append(0)
            continue
            
        #aexplainer=PGExplainer(model, new_graphs, features, task)
        #aexplainer.prepare([indx])
        aexplainer = explainer
        aexplainer.graphs = new_graphs
        n_graph, n_expl = aexplainer.explain(idx)
        t_n_expl = n_expl
        t_n_graph = n_graph
        n_expl = np.array(n_expl.detach())
            
        n_graph = np.array(n_graph.detach())
        bi_n_expl = n_expl.copy()
        for i in range(n_expl.shape[0]):
            r_pair = [n_graph[1,i],n_graph[0,i]]
            if n_graph[0,i]<=n_graph[1,i]:
                n_expl[i] += bi_n_expl[index_edge(n_graph, r_pair)]
            else:
                n_expl[i] = 0
                    
        n_o_id=[]
                
        for i in range(n_expl.shape[0]):
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
                
        if fool >= maxfool:
            ansexl = t_n_expl
            ansgra = t_n_graph
        maxfool = max(maxfool, fool)
        
        fools.append(maxfool)
        
    elif attack =="random": 
        o_bid=[]
        for j in range(len(o_id)):
            i = index_edge(graph,[graph[0,o_id[j]],graph[1,o_id[j]]])
            o_bid.append(i)
            i = index_edge(graph,[graph[1,o_id[j]],graph[0,o_id[j]]])
            o_bid.append(i)
            
        new_graphs = graphs
        new_degree = degree
        cb =0 
        cnt2=0
        add = []
        delete = []
        while cnt2<xi*2:
            
            a = random.randint(0,N-2)
            b = random.randint(a+1,N-1)
            if cb>=100:
                break
            flag=0
            for i in range(len(o_bid)):
                if graph[0,o_bid[i]]==a and graph[1,o_bid[i]]==b:
                    flag=1
                    break
                if graph[0,o_bid[i]]==b and graph[1,o_bid[i]]==a:
                    flag=1
                    break
            if flag ==1:
                cb+=1
                print("!")
                continue
                
            a1 = index_edge(graphs, [a,b])
            b1 = index_edge(graphs, [b,a])
                
            if a1==None:
                if [a,b] in add:
                    continue
                add.append([a,b])
                add.append([b,a])
                if maps[a]!=0:
                    new_degree[maps[a]-1]+=1
                if maps[b]!=0:
                    new_degree[maps[b]-1]+=1
            else:
                if a1 in delete:
                    continue
                if maps[a]!=0:
                    new_degree[maps[a]-1]-=1
                if maps[b]!=0:
                    new_degree[maps[b]-1]-=1
                delete.append(a1)
                delete.append(b1)
                    
            cnt2+=2
        new_graphs=  np.delete(graphs.permute(1,0),delete,axis =0)
        new_graphs = torch.cat((new_graphs,torch.tensor(add,dtype = torch.int32)))
        new_graphs = new_graphs.permute(1,0)
            
        ls = likelyhood(degree,new_degree)
        predict_before = np.argmax(np.array(model(features,graphs).detach()[indx]))
        predict_after = np.argmax(np.array(model(features,new_graphs).detach()[indx]))
        if ls >0.000157 or predict_before!=predict_after:
            continue
            
        #aexplainer=PGExplainer(model, new_graphs, features, task)
        #aexplainer.prepare([indx])
        aexplainer = explainer
        aexplainer.graphs = new_graphs
        n_graph, n_expl = aexplainer.explain(idx)
        t_n_expl = n_expl
        t_n_graph = n_graph
        n_expl = np.array(n_expl.detach())
                
        bi_n_expl = n_expl.copy()
        for i in range(n_expl.shape[0]):
            r_pair = [n_graph[1,i],n_graph[0,i]]
            if n_graph[0,i]<=n_graph[1,i]:
                n_expl[i] += bi_n_expl[index_edge(n_graph, r_pair)]
            else:
                n_expl[i] = 0
                
        n_o_id=[]
        for i in range(n_expl.shape[0]):
            for j in range(len(o_id)):
                if n_graph[0,i]==graph[0,o_id[j]] and n_graph[1,i]==graph[1,o_id[j]]:
                    n_o_id.append(i)
                        
        now_id = np.flip(np.argsort((n_expl).reshape(-1))[-k:])
        fool=0
        for i in range(len(now_id)):
            if np.isin([now_id[i]],n_o_id)==False :
                if i<k:
                    fool+=1
                else:
                    break
        n_graph = np.array(n_graph.detach())
        fools.append(fool)
            
    #print(ex_time)
    #print(hot_time)
    #print(cold_time)
    #f.write(str(fools)+'\n')
    #f.write(str(cnt_f)+'\n')
    print(fools)
print(fools)
print("average attack edges:")
print(sum(fools)/len(fools))
#f.close()
    
