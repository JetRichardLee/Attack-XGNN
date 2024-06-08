# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 17:00:58 2023

@author: 31271
"""

from ExplanationEvaluation.configs.selector import Selector
from ExplanationEvaluation.tasks.training import train_node, train_graph, eval_node, eval_graph

import torch
import numpy as np

_dataset = 'products' # One of: bashapes, bacommunity, treecycles, treegrids, ba2motifs, mutag

# Parameters below should only be changed if you want to run any of the experiments in the supplementary
_folder = 'replication' # One of: replication, batchnorm
_model = 'gnn' if _folder == 'replication' else 'ori'

# PGExplainer
config_path = f"./ExplanationEvaluation/configs/{_folder}/models/model_{_model}_{_dataset}.json"

config = Selector(config_path)
extension = (_folder == 'extension')

config = Selector(config_path).args

torch.manual_seed(config.model.seed)
torch.cuda.manual_seed(config.model.seed)
np.random.seed(config.model.seed)

_dataset = config.model.dataset
_explainer = config.model.paper

if _dataset[:3] == "syn":
    train_node(_dataset, _explainer, config.model)
elif _dataset == "products":
    train_node(_dataset, _explainer, config.model)
elif _dataset == "ba2" or _dataset == "mutag" or  _dataset == "REDDIT-BINARY":
    train_graph(_dataset, _explainer, config.model)

if _dataset[:3] == "syn":
    eval_node(_dataset, _explainer, config.model)
elif _dataset == "products":
    eval_node(_dataset, _explainer, config.model)
elif _dataset == "ba2" or _dataset == "mutag" or  _dataset == "REDDIT-BINARY":
    eval_graph(_dataset, _explainer, config.model)