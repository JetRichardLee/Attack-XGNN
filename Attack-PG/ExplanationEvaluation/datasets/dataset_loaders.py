import pickle as pkl
import numpy as np
import os
from numpy.random.mtrand import RandomState

from ogb.graphproppred import GraphPropPredDataset
from ogb.nodeproppred import NodePropPredDataset
from ExplanationEvaluation.datasets.utils import preprocess_features, preprocess_adj, adj_to_edge_index,load_ogbn_dataset, load_real_dataset_2,load_real_dataset


def load_graph_dataset(_dataset, shuffle=True):
    """Load a graph dataset and optionally shuffle it.

    :param _dataset: Which dataset to load. Choose from "ba2" or "mutag"
    :param shuffle: Boolean. Wheter to suffle the loaded dataset.
    :returns: np.array
    """
    # Load the chosen dataset from the pickle file.
    if _dataset == "ba2":
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = dir_path + '/pkls/' + "BA-2motif" + '.pkl'
        with open(path, 'rb') as fin:
            adjs, features, labels = pkl.load(fin)

    elif _dataset == "mutag":
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = dir_path + '/pkls/' + "Mutagenicity" + '.pkl'
        if not os.path.exists(path): # pkl not yet created
            print("Mutag dataset pickle is not yet created, doing this now. Can take some time")
            adjs, features, labels = load_real_dataset(path, dir_path + '/Mutagenicity/Mutagenicity_')
            print("Done with creating the mutag dataset")
        else:
            with open(path, 'rb') as fin:
                adjs, features, labels = pkl.load(fin)
                
    elif _dataset == "REDDIT-BINARY":
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = dir_path + '/pkls/' + "REDDIT-BINARY" + '.pkl'
        if not os.path.exists(path): # pkl not yet created
            print("REDDIT-BINARY dataset pickle is not yet created, doing this now. Can take some time")
            adjs, features, labels = load_real_dataset_2(path, dir_path + '/REDDIT-BINARY/REDDIT-BINARY_')
            print("Done with creating the mutag dataset")
        else:
            with open(path, 'rb') as fin:
                adjs, features, labels = pkl.load(fin)
    else:
        print("Unknown dataset")
        raise NotImplemented

    n_graphs = adjs.shape[0]
    indices = np.arange(0, n_graphs)
    if shuffle:
        prng = RandomState(42) # Make sure that the permutation is always the same, even if we set the seed different
        indices = prng.permutation(indices)

    # Create shuffled data
    adjs = adjs[indices]
    features = features[indices].astype('float32')
    labels = labels[indices]

    # Create masks
    train_indices = np.arange(0, int(n_graphs*0.8))
    val_indices = np.arange(int(n_graphs*0.8), int(n_graphs*0.9))
    test_indices = np.arange(int(n_graphs*0.9), n_graphs)
    train_mask = np.full((n_graphs), False, dtype=bool)
    train_mask[train_indices] = True
    val_mask = np.full((n_graphs), False, dtype=bool)
    val_mask[val_indices] = True
    test_mask = np.full((n_graphs), False, dtype=bool)
    test_mask[test_indices] = True

    # Transform to edge index
    edge_index = adj_to_edge_index(adjs)

    return edge_index, features, labels, train_mask, val_mask, test_mask


def _load_node_dataset(_dataset):
    """Load a node dataset.

    :param _dataset: Which dataset to load. Choose from "syn1", "syn2", "syn3" or "syn4"
    :returns: np.array
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = dir_path + '/pkls/' + _dataset + '.pkl'
    with open(path, 'rb') as fin:
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, edge_label_matrix  = pkl.load(fin)
    labels = y_train
    labels[val_mask] = y_val[val_mask]
    labels[test_mask] = y_test[test_mask]
    return adj, features, labels, train_mask, val_mask, test_mask

def _load_ogb_dataset(_dataset,shuffle=True):
    """Load a node dataset.

    :param _dataset: Which dataset to load. Choose from "products"
    :returns: np.array
    """

    if _dataset == "products":
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = dir_path + '/pkls/' + "products" + '.pkl'
        if not os.path.exists(path): # pkl not yet created
            print("Products dataset pickle is not yet created, doing this now. Can take some time")
            edge_index, features, labels = load_ogbn_dataset(path, "F:/OgbDataset/ogbn-products.pkl")
            print("Done with creating the products dataset")
        
        with open(path, 'rb') as fin:
            edge_index, features, labels  = pkl.load(fin)
            n_nodes = labels.shape[0]
            labels = np.squeeze(labels)
        indices = np.arange(0, n_nodes)
        if shuffle:
            prng = RandomState(42) # Make sure that the permutation is always the same, even if we set the seed different
            indices = prng.permutation(indices)
        train_indices = indices[0:int(n_nodes*0.8)]
        val_indices =indices[int(n_nodes*0.8):int(n_nodes*0.9)]
        test_indices = indices[int(n_nodes*0.9):n_nodes]
        train_mask = np.full((n_nodes), False, dtype=bool)
        train_mask[train_indices] = True
        val_mask = np.full((n_nodes), False, dtype=bool)
        val_mask[val_indices] = True
        test_mask = np.full((n_nodes), False, dtype=bool)
        test_mask[test_indices] = True
    return edge_index, features, labels, train_mask, val_mask, test_mask


def load_dataset(_dataset, skip_preproccessing=False, shuffle=True):
    """High level function which loads the dataset
    by calling others spesifying in nodes or graphs.

    Keyword arguments:
    :param _dataset: Which dataset to load. Choose from "syn1", "syn2", "syn3", "syn4", "ba2" or "mutag"
    :param skip_preproccessing: Whether or not to convert the adjacency matrix to an edge matrix.
    :param shuffle: Should the returned dataset be shuffled or not.
    :returns: multiple np.arrays
    """
    print(f"Loading {_dataset} dataset")
    if _dataset[:3] == "syn": # Load node_dataset
        adj, features, labels, train_mask, val_mask, test_mask = _load_node_dataset(_dataset)
        preprocessed_features = preprocess_features(features).astype('float32')
        if skip_preproccessing:
            graph = adj
        else:
            graph = preprocess_adj(adj)[0].astype('int64').T
        labels = np.argmax(labels, axis=1)
        return graph, preprocessed_features, labels, train_mask, val_mask, test_mask
    if _dataset == "products": # Load node_dataset
        
        return _load_ogb_dataset(_dataset)
    else: # Load graph dataset
        return load_graph_dataset(_dataset, shuffle=False)
