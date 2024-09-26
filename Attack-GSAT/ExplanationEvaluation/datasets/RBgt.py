# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 12:36:54 2023

@author: 31271
"""
import numpy as np
import scipy.sparse as sp
import torch
import scipy
import pickle as pkl
from scipy.sparse import coo_matrix

def get_graph_data_gt(path):

    pri = path

    file_edges = pri+'A.txt'
    file_graph_indicator = pri+'graph_indicator.txt'
    file_graph_labels = pri+'graph_labels.txt'

    edges = np.loadtxt( file_edges,delimiter=',').astype(np.int32)
    edge_labels = np.zeros(edges.shape[0],dtype = np.int32)
    graph_indicator = np.loadtxt(file_graph_indicator,delimiter=',').astype(np.int32)
    graph_labels = np.loadtxt(file_graph_labels,delimiter=',').astype(np.int32)
    
    

    graph_id = 1
    starts = [1]
    node2graph = {}
    for i in range(len(graph_indicator)):
        if graph_indicator[i]!=graph_id:
            graph_id = graph_indicator[i]
            starts.append(i+1)
        node2graph[i+1]=len(starts)-1
    starts.append(len(graph_indicator)+1)
    graphid  = 0
    edge_lists = []
    edge_label_lists = []
    edge_list = []
    edge_label_list = []
    hot_id = []
    hot_degree = []
    shot_degree = []
    degree_c = np.zeros(starts[2]-starts[1],dtype = np.int32)
    for (s,t),l in list(zip(edges,edge_labels)):
        sgid = node2graph[s]
        tgid = node2graph[t]
        if sgid == -1 or tgid == -1:
            continue
        if sgid!=tgid:
            print('edges connecting different graphs, error here, please check.')
            print(s,t,'graph id',sgid,tgid)
            exit(1)
        gid = sgid
        
        if gid !=  graphid:
            hot_id.append(np.argmax(degree_c,axis=0))
            edge_lists.append(edge_list)
            edge_label_lists.append(edge_label_list)
            edge_list = []
            edge_label_list = []
            degree_c = np.zeros(starts[gid+1]-starts[gid],dtype = np.int32)
            graphid = gid
        start = starts[gid]
        degree_c[s-start]+=1
        degree_c[t-start]+=1
        
        edge_list.append((s-start,t-start))
        edge_label_list.append(l)
    
    hot_id.append(np.argmax(degree_c,axis=0))
    hot_degree.append(np.max(degree_c))
    np.savetxt('./hot_id.txt',hot_id,fmt="%d")
    np.savetxt('./graph_labels.txt',graph_labels,fmt="%d")
    print(len(hot_id))
    print(len(graph_labels))
    return 


import os
dir_path = os.path.dirname(os.path.realpath(__file__))
path_graph = dir_path+'/REDDIT-BINARY/REDDIT-BINARY_'
get_graph_data_gt(path_graph)