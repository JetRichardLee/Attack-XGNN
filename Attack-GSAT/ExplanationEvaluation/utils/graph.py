import torch

def index_edge(graph, pair):
    #print(graph)
    #print(pair)
    for i in range(len(graph[0])):
        if graph[0,i]==pair[0] and graph[1,i]==pair[1]:
            return i
    return None
