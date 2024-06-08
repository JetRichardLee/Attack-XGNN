import torch
import torch_geometric as ptgeom
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Data
from tqdm import tqdm

from ExplanationEvaluation.explainers.BaseExplainer import BaseExplainer
from ExplanationEvaluation.utils.graph import index_edge


class Deduction_PGAttacker(BaseExplainer):
    """
    A class encaptulating our deduction_based attacker against the PGExplainer .
    This implement is modified from the official implement of PGExplainer (https://arxiv.org/abs/2011.04573)
    
    
    Original PG parameters
    :param model_to_explain: graph classification model who's predictions we wish to explain.
    :param graphs: the collections of edge_indices representing the graphs.
    :param features: the collcection of features for each node in the graphs.
    :param task: str "node" or "graph".
    :param epochs: amount of epochs to train our explainer.
    :param lr: learning rate used in the training of the explainer.
    :param temp: the temperture parameters dictacting how we sample our random graphs.
    :param reg_coefs: reguaization coefficients used in the loss. The first item in the tuple restricts the size of the explainations, the second rescticts the entropy matrix mask.
    :params sample_bias: the bias we add when sampling random graphs.
    Set the same as the target PG Explainer
    
    Original PG functions
    :function _create_explainer_input: utility;
    :function _sample_graph: utility; sample an explanatory subgraph.
    :function pg_loss: calculate the loss of the explainer during training.
    
    Attacker parameters
    :param beta: tuning parameters. The assumed lower bound of the target mask
    :param N: the number of samples applied
    
    Attacker functions 
    :function D_loss: the loss we applied on the existent edges besides the explanatory edges
    :function A_loss: the loss we applied on the non-existent edges
    :function learn_deletion: conduct a mask-learning process to learn the MD on E (ED+ES)
    :function learn_addition: conduct a mask-learning process to learn the MA on EA+ES
    
    
        """
    def __init__(self, model_to_explain, graphs, features, task, epochs=30, lr=0.003, temp=(5.0, 2.0), reg_coefs=(0.05, 1.0),sample_bias=0,beta=0.5,N=4,r_epochs=20):
        super().__init__(model_to_explain, graphs, features, task)

        self.epochs = epochs
        self.lr = lr
        self.temp = temp
        self.reg_coefs = reg_coefs
        self.sample_bias = sample_bias
        self.beta = beta
        self.N=N
        self.r_epochs = r_epochs
        if self.type == "graph":
            self.expl_embedding = self.model_to_explain.embedding_size * 2
        else:
            self.expl_embedding = self.model_to_explain.embedding_size * 3


    def _create_explainer_input(self, pair, embeds, node_id):
        """
        Given the embeddign of the sample by the model that we wish to explain, this method construct the input to the mlp explainer model.
        Depending on if the task is to explain a graph or a sample, this is done by either concatenating two or three embeddings.
        :param pair: edge pair
        :param embeds: embedding of all nodes in the graph
        :param node_id: id of the node, not used for graph datasets
        :return: concatenated embedding
        """
        rows = pair[0]
        cols = pair[1]
        row_embeds = embeds[rows]
        col_embeds = embeds[cols]
        if self.type == 'node'or self.type == 'ogb_node':
            node_embed = embeds[node_id].repeat(rows.size(0), 1)
            input_expl = torch.cat([row_embeds, col_embeds, node_embed], 1)
        else:
            # Node id is not used in this case
            input_expl = torch.cat([row_embeds, col_embeds], 1)
        return input_expl


    def _sample_graph(self, sampling_weights, temperature=1.0, bias=0.0, training=True):
        """
        Implementation of the reparamerization trick to obtain a sample graph while maintaining the posibility to backprop.
        :param sampling_weights: Weights provided by the mlp
        :param temperature: annealing temperature to make the procedure more deterministic
        :param bias: Bias on the weights to make samplign less deterministic
        :param training: If set to false, the samplign will be entirely deterministic
        :return: sample graph
        """
        if training:
            bias = bias + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1-bias)) * torch.rand(sampling_weights.size()) + (1-bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = (gate_inputs + sampling_weights) / temperature
            graph =  torch.sigmoid(gate_inputs)
        else:
            graph = torch.sigmoid(sampling_weights)
        return graph
    
    def pg_loss(self, masked_pred, original_pred, mask, reg_coefs):
        """
        Returns the loss score based on the given mask.
        :param masked_pred: Prediction based on the current explanation
        :param original_pred: Predicion based on the original graph
        :param edge_mask: Current explanaiton
        :param reg_coefs: regularization coefficients
        :return: loss
        """
        size_reg = reg_coefs[0]
        entropy_reg = reg_coefs[1]

        # Regularization losses
        size_loss = torch.sum(mask) * size_reg
        mask_ent_reg = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent_loss = entropy_reg * torch.mean(mask_ent_reg)
        # Explanation loss
        cce_loss = torch.nn.functional.cross_entropy(masked_pred, original_pred)

        return cce_loss + size_loss + mask_ent_loss
    
    
    def D_loss(self, masks,mask_preds, original_pred, mask, reg_coefs):
        """
        Returns the deletion loss score based on the given sampled masks.
        :param masks: the sampled masks on E. 
                        masks[0] is the mask fixing Es as 0
                        masks [1:N] is the N sampled masks where Es is in [beta:1]
        :param mask_preds: Prediction based on the masks
        :param original_pred: Predicion based on the original graph
        :param mask: Current mask on E
        :param reg_coefs: regularization coefficients of the target PG explainer
        :return: loss
        """
        
        loss = -self.pg_loss(mask_preds[0], original_pred, masks[0], reg_coefs)*self.N
        for i in range(1,len(masks)):
            loss+= self.pg_loss(mask_preds[i], original_pred, masks[i], reg_coefs)    
            
        return loss
    
    def A_loss(self, masks,mask_preds, original_pred, mask, reg_coefs):
        """
        Returns the addition loss score based on the given sampled masks.
        :param masks: the sampled masks on EA+Es. 
                        masks[0] is the mask fixing Es as 0
                        masks [1:N] is the N sampled masks where Es is in [beta:1]
        :param mask_preds: Prediction based on the masks
        :param original_pred: Predicion based on the original graph
        :param mask: Current mask on EA+Es
        :param reg_coefs: regularization coefficients of the target PG explainer
        :return: loss
        """
        loss = self.pg_loss(mask_preds[0], original_pred, masks[0], reg_coefs)*self.N
        for i in range(1,len(masks)):
            loss-= self.pg_loss(mask_preds[i], original_pred, masks[i], reg_coefs)    
 
        return loss


    def learn_deletion(self, indices=None,fliter =None, bias=None):
        """
        We adopt the same stochastic way to reversely learning our masks MD
        """
        # Creation of the explainer_model is done here to make sure that the seed is set
        self.explainer_model = nn.Sequential(
            nn.Linear(self.expl_embedding, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        if indices is None: # Consider all indices
            indices = range(0, self.graphs.size(0))

        self.explainer_model.train()

        optimizer = Adam(self.explainer_model.parameters(), lr=self.lr)
        temp_schedule = lambda e: self.temp[0]*((self.temp[1]/self.temp[0])**(e/self.epochs))

        if self.type == 'node'or self.type == 'ogb_node':
            embeds = self.model_to_explain.embedding(self.features, self.graphs).detach()

        # Start training loop
        for e in tqdm(range(0, self.r_epochs)):
            optimizer.zero_grad()
            loss = torch.FloatTensor([0]).detach()
            t = temp_schedule(e)

            for n in indices:
                n = int(n)
                if self.type == 'node':
                    feats = self.features
                    graph = ptgeom.utils.k_hop_subgraph(n, 3, self.graphs)[1]
                elif self.type == 'ogb_node':
                    feats = self.features
                    graph = ptgeom.utils.k_hop_subgraph(n, 2, self.graphs)[1]
                else:
                    feats = self.features[n].detach()
                    graph = self.graphs[n].detach()
                    embeds = self.model_to_explain.embedding(feats, graph).detach()

                #obtain the current MD
                input_expl = self._create_explainer_input(graph, embeds, n).unsqueeze(0)
                sampling_weights = self.explainer_model(input_expl)
                mask = self._sample_graph(sampling_weights, t, bias=self.sample_bias).squeeze()

                #obtain N+1 sample with different b setted in 0 and [beta,1]
                beta = self.beta
                N=self.N
                masks = [mask*torch.FloatTensor(fliter)+torch.FloatTensor(bias)*0.001]
                preds = [self.model_to_explain(feats, graph, edge_weights=masks[0])]
                fracs = (0.999-beta)/(N-1)
                #Note the beta we adopted is not set to completely 1 or 0 but 0.999 and 0.001 instead,
                #since mask value with completely 0 or 1 will make the entropy loss to be Nan
                
                for i in range(N):
                    masks.append(mask*torch.FloatTensor(fliter)+torch.FloatTensor(bias)*(beta+fracs*i))
                    preds.append(self.model_to_explain(feats, graph, edge_weights=masks[i+1]))

                original_pred = self.model_to_explain(feats, graph)

                #if self.type == 'node': # we only care for the prediction of the node
                if self.type == 'node'or self.type == 'ogb_node':
                    for i in range(len(preds)):
                        preds[i]=preds[i][n].unsqueeze(dim=0)
                        #print(preds[i].shape)
                    original_pred = original_pred[n]
                    #print(original_pred.shape)
                id_loss = self.D_loss(masks, preds,  torch.argmax(original_pred).unsqueeze(0), mask, self.reg_coefs)
                #id_loss = self.hot_loss(masked_pred_0,masked_pred_1,masked_pred_2,masked_pred_3,masked_pred_4,masked_pred_5,masked_pred_6,masked_pred_7,masked_pred_8, torch.argmax(original_pred).unsqueeze(0), mask, self.reg_coefs)
                loss += id_loss

            loss.backward()
            optimizer.step()
            
        input_expl = self._create_explainer_input(graph, embeds, indices).unsqueeze(dim=0)
        sampling_weights = self.explainer_model(input_expl)
        mask = self._sample_graph(sampling_weights, training=False).squeeze()
        #print("!mask:")
        #print(mask)
        #expl_graph_weights = torch.zeros(graph.size(1)) # Combine with original graph
        #for i in range(0, mask.size(0)):
        #    pair = graph.T[i]
        #    t = index_edge(graph, pair)
        #    expl_graph_weights[t] = mask[i]
        return graph,mask
    
    def learn_addition(self, indices=None,fliter =None, bias=None):
        """
        We adopt the same stochastic way to reversely learning our masks MA
        """
        # Creation of the explainer_model is done here to make sure that the seed is set
        self.explainer_model = nn.Sequential(
            nn.Linear(self.expl_embedding, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        if indices is None: # Consider all indices
            indices = range(0, self.graphs.size(0))

        self.explainer_model.train()

        optimizer = Adam(self.explainer_model.parameters(), lr=self.lr)
        temp_schedule = lambda e: self.temp[0]*((self.temp[1]/self.temp[0])**(e/self.epochs))

        if self.type == 'node'or self.type == 'ogb_node':
            embeds = self.model_to_explain.embedding(self.features, self.graphs).detach()

        for e in tqdm(range(0, self.r_epochs)):
            optimizer.zero_grad()
            loss = torch.FloatTensor([0]).detach()
            t = temp_schedule(e)

            for n in indices:
                n = int(n)
                if self.type == 'node':
                    feats = self.features
                    graph = ptgeom.utils.k_hop_subgraph(n, 3, self.graphs)[1]
                elif self.type == 'ogb_node':
                    feats = self.features
                    graph = ptgeom.utils.k_hop_subgraph(n, 2, self.graphs)[1]
                else:
                    feats = self.features[n].detach()
                    graph = self.graphs[n].detach()
                    embeds = self.model_to_explain.embedding(feats, graph).detach()

                # Sample possible explanation
                input_expl = self._create_explainer_input(graph, embeds, n).unsqueeze(0)
                sampling_weights = self.explainer_model(input_expl)
                mask = self._sample_graph(sampling_weights, t, bias=self.sample_bias).squeeze()
                #print("mask:")
                #print(mask)
                beta = self.beta
                N=self.N
                masks = [mask*torch.FloatTensor(fliter)+torch.FloatTensor(bias)*0.001]
                preds = [self.model_to_explain(feats, graph, edge_weights=masks[0])]
                fracs = (0.999-beta)/(N-1)
                for i in range(N):
                    masks.append(mask*torch.FloatTensor(fliter)+torch.FloatTensor(bias)*(beta+fracs*i))
                    preds.append(self.model_to_explain(feats, graph, edge_weights=masks[i+1]))
                original_pred = self.model_to_explain(feats, graph)

                #if self.type == 'node': # we only care for the prediction of the node
                if self.type == 'node'or self.type == 'ogb_node':
                    for i in range(len(preds)):
                        preds[i]=preds[i][n].unsqueeze(dim=0)
                    original_pred = original_pred[n]
                    
                id_loss = self.A_loss(masks, preds, torch.argmax(original_pred).unsqueeze(0), mask, self.reg_coefs)
                #id_loss = self.cold_loss(masked_pred_0,masked_pred_1,masked_pred_2,masked_pred_3,masked_pred_4,masked_pred_5,masked_pred_6,masked_pred_7,masked_pred_8, torch.argmax(original_pred).unsqueeze(0), mask, self.reg_coefs)
                loss += id_loss

            loss.backward()
            optimizer.step()
            
        input_expl = self._create_explainer_input(graph, embeds, indices).unsqueeze(dim=0)
        sampling_weights = self.explainer_model(input_expl)
        mask = self._sample_graph(sampling_weights, training=False).squeeze()
        return graph,mask
    
    def prepare(self, indices=None):
        return 
    
    def explain(self, index):
        return 

class Loss_PGAttacker(BaseExplainer):
    """
    A class encaptulating our loss_based attacker against the PGExplainer .
    This implement is modified from the official implement of PGExplainer (https://arxiv.org/abs/2011.04573)
    
    
    Original PG parameters
    :param model_to_explain: graph classification model who's predictions we wish to explain.
    :param graphs: the collections of edge_indices representing the graphs.
    :param features: the collcection of features for each node in the graphs.
    :param task: str "node" or "graph".
    :param epochs: amount of epochs to train our explainer.
    :param lr: learning rate used in the training of the explainer.
    :param temp: the temperture parameters dictacting how we sample our random graphs.
    :param reg_coefs: reguaization coefficients used in the loss. The first item in the tuple restricts the size of the explainations, the second rescticts the entropy matrix mask.
    :params sample_bias: the bias we add when sampling random graphs.
    Set the same as the target PG Explainer
    
    Original PG functions
    :function _create_explainer_input: utility;
    :function _sample_graph: utility; sample an explanatory subgraph.
    :function pg_loss: calculate the loss of the explainer during training.
    
    Attacker parameters
    :param gamma: tuning parameters. The assumed value of the target mask
    
    Attacker functions 
    :function D_loss: the loss we applied on the existent edges besides the explanatory edges
    :function A_loss: the loss we applied on the non-existent edges
    :function learn_deletion: conduct a mask-learning process to learn the MD on E (ED+ES)
    :function learn_addition: conduct a mask-learning process to learn the MA on EA+ES
    
    
        """
    def __init__(self, model_to_explain, graphs, features, task, epochs=30, lr=0.01, temp=(5.0, 2.0), reg_coefs=(0.05, 1.0),sample_bias=0,gamma=0.7,r_epochs=70):
        super().__init__(model_to_explain, graphs, features, task)

        self.epochs = epochs
        self.lr = lr
        self.temp = temp
        self.reg_coefs = reg_coefs
        self.sample_bias = sample_bias
        self.gamma = gamma
        self.r_epochs = r_epochs
        if self.type == "graph":
            self.expl_embedding = self.model_to_explain.embedding_size * 2
        else:
            self.expl_embedding = self.model_to_explain.embedding_size * 3


    def _create_explainer_input(self, pair, embeds, node_id):
        """
        Given the embeddign of the sample by the model that we wish to explain, this method construct the input to the mlp explainer model.
        Depending on if the task is to explain a graph or a sample, this is done by either concatenating two or three embeddings.
        :param pair: edge pair
        :param embeds: embedding of all nodes in the graph
        :param node_id: id of the node, not used for graph datasets
        :return: concatenated embedding
        """
        rows = pair[0]
        cols = pair[1]
        row_embeds = embeds[rows]
        col_embeds = embeds[cols]
        if self.type == 'node'or self.type == 'ogb_node':
            node_embed = embeds[node_id].repeat(rows.size(0), 1)
            input_expl = torch.cat([row_embeds, col_embeds, node_embed], 1)
        else:
            # Node id is not used in this case
            input_expl = torch.cat([row_embeds, col_embeds], 1)
        return input_expl


    def _sample_graph(self, sampling_weights, temperature=1.0, bias=0.0, training=True):
        """
        Implementation of the reparamerization trick to obtain a sample graph while maintaining the posibility to backprop.
        :param sampling_weights: Weights provided by the mlp
        :param temperature: annealing temperature to make the procedure more deterministic
        :param bias: Bias on the weights to make samplign less deterministic
        :param training: If set to false, the samplign will be entirely deterministic
        :return: sample graph
        """
        if training:
            bias = bias + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1-bias)) * torch.rand(sampling_weights.size()) + (1-bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = (gate_inputs + sampling_weights) / temperature
            graph =  torch.sigmoid(gate_inputs)
        else:
            graph = torch.sigmoid(sampling_weights)
        return graph


    def D_loss(self, masked_pred_gamma, original_pred, mask_gamma, reg_coefs):
        """
        Returns the loss score based on the given MD*f+b.
        :param masked_pred_gamma: Prediction based on the MD*f+b
        :param original_pred: Predicion based on the original graph
        :param masked_gamma: MD*f+b
        :param reg_coefs: regularization coefficients
        :return: loss
        """
        size_reg = reg_coefs[0]
        entropy_reg = reg_coefs[1]

        # Regularization losses
        size_loss = torch.sum(mask_gamma) * size_reg
        mask_ent_reg = -mask_gamma * torch.log(mask_gamma) - (1 - mask_gamma) * torch.log(1 - mask_gamma)
        mask_ent_loss = entropy_reg * torch.mean(mask_ent_reg)

        # Explanation loss
        cce_loss =  torch.nn.functional.cross_entropy(masked_pred_gamma, original_pred)
        return cce_loss + size_loss + mask_ent_loss

    def A_loss(self, masked_pred_gamma, original_pred, mask_gamma, reg_coefs):
        """
        Returns the loss score based on the given MA*f+b.
        :param masked_pred_gamma: Prediction based on the MA*f+b.
        :param original_pred: Predicion based on the original graph
        :param mask_gamma: MA*f+b.
        :param reg_coefs: regularization coefficients
        :return: loss
        """
        size_reg = reg_coefs[0]
        entropy_reg = reg_coefs[1]

        # Regularization losses
        size_loss = torch.sum(mask_gamma) * size_reg
        mask_ent_reg = -mask_gamma * torch.log(mask_gamma) - (1 - mask_gamma) * torch.log(1 - mask_gamma)
        mask_ent_loss = entropy_reg * torch.mean(mask_ent_reg)

        # Explanation loss
        cce_loss = torch.nn.functional.cross_entropy(masked_pred_gamma, original_pred)
        return -cce_loss - size_loss - mask_ent_loss

    def learn_deletion(self, indices=None,fliter =None, bias=None):
        """
        We adopt the same stochastic way to reversely learning our masks MD
        """
        # Creation of the explainer_model is done here to make sure that the seed is set
        self.explainer_model = nn.Sequential(
            nn.Linear(self.expl_embedding, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        if indices is None: # Consider all indices
            indices = range(0, self.graphs.size(0))
            
        self.explainer_model.train()

        # Create optimizer and temperature schedule
        optimizer = Adam(self.explainer_model.parameters(), lr=self.lr)
        temp_schedule = lambda e: self.temp[0]*((self.temp[1]/self.temp[0])**(e/self.r_epochs))

        # If we are explaining a graph, we can determine the embeddings before we run
        if self.type == 'node'or self.type == 'ogb_node':
            embeds = self.model_to_explain.embedding(self.features, self.graphs).detach()

        # Start training loop
        for e in tqdm(range(0, self.r_epochs)):
            optimizer.zero_grad()
            loss = torch.FloatTensor([0]).detach()
            t = temp_schedule(e)

            for n in indices:
                n = int(n)
                if self.type == 'node':
                    # Similar to the original paper we only consider a subgraph for explaining
                    feats = self.features
                    graph = ptgeom.utils.k_hop_subgraph(n, 3, self.graphs)[1]
                elif self.type == 'ogb_node':
                    feats = self.features
                    graph = ptgeom.utils.k_hop_subgraph(n, 2, self.graphs)[1]
                else:
                    feats = self.features[n].detach()
                    graph = self.graphs[n].detach()
                    embeds = self.model_to_explain.embedding(feats, graph).detach()
                gamma =self.gamma
                # Sample possible explanation
                input_expl = self._create_explainer_input(graph, embeds, n).unsqueeze(0)
                sampling_weights = self.explainer_model(input_expl)
                mask = self._sample_graph(sampling_weights, t, bias=self.sample_bias).squeeze()
                #print("mask:")
                #print(mask)
                mask_gamma = mask*torch.FloatTensor(fliter)+torch.FloatTensor(bias)*gamma
                masked_pred_gamma = self.model_to_explain(feats, graph, edge_weights=mask_gamma)

                original_pred = self.model_to_explain(feats, graph)

                #if self.type == 'node': # we only care for the prediction of the node
                if self.type == 'node'or self.type == 'ogb_node':
                    masked_pred_gamma = masked_pred_gamma[n].unsqueeze(dim=0)
                    original_pred = original_pred[n]

                id_loss = self.D_loss(masked_pred_gamma, torch.argmax(original_pred).unsqueeze(0), mask_gamma, self.reg_coefs)
                loss += id_loss

            loss.backward()
            optimizer.step()
            
        input_expl = self._create_explainer_input(graph, embeds, indices).unsqueeze(dim=0)
        sampling_weights = self.explainer_model(input_expl)
        mask = self._sample_graph(sampling_weights, training=False).squeeze()
        #print("!mask:")
        #print(mask)
        #expl_graph_weights = torch.zeros(graph.size(1)) # Combine with original graph
        #for i in range(0, mask.size(0)):
        #    pair = graph.T[i]
        #    t = index_edge(graph, pair)
        #    expl_graph_weights[t] = mask[i]
        return graph,mask
    
    def learn_addition(self, indices=None,fliter =None, bias=None):
        """
        We adopt the same stochastic way to reversely learning our masks MA
        """
        # Creation of the explainer_model is done here to make sure that the seed is set
        self.explainer_model = nn.Sequential(
            nn.Linear(self.expl_embedding, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        if indices is None: # Consider all indices
            indices = range(0, self.graphs.size(0))

        # Make sure the explainer model can be trained
        self.explainer_model.train()

        # Create optimizer and temperature schedule
        optimizer = Adam(self.explainer_model.parameters(), lr=self.lr)
        temp_schedule = lambda e: self.temp[0]*((self.temp[1]/self.temp[0])**(e/self.r_epochs))

        # If we are explaining a graph, we can determine the embeddings before we run
        #if self.type == 'node':
        if self.type == 'node'or self.type == 'ogb_node':
            embeds = self.model_to_explain.embedding(self.features, self.graphs).detach()

        # Start training loop
        for e in tqdm(range(0, self.r_epochs)):
            optimizer.zero_grad()
            loss = torch.FloatTensor([0]).detach()
            t = temp_schedule(e)

            for n in indices:
                n = int(n)
                if self.type == 'node':
                    # Similar to the original paper we only consider a subgraph for explaining
                    feats = self.features
                    graph = ptgeom.utils.k_hop_subgraph(n, 3, self.graphs)[1]
                elif self.type == 'ogb_node':
                    feats = self.features
                    graph = ptgeom.utils.k_hop_subgraph(n, 2, self.graphs)[1]
                else:
                    feats = self.features[n].detach()
                    graph = self.graphs[n].detach()
                    embeds = self.model_to_explain.embedding(feats, graph).detach()
                gamma = self.gamma
                # Sample possible explanation
                input_expl = self._create_explainer_input(graph, embeds, n).unsqueeze(0)
                sampling_weights = self.explainer_model(input_expl)
                mask = self._sample_graph(sampling_weights, t, bias=self.sample_bias).squeeze()
                #print("~!")
                #print(mask)
                mask_gamma = mask*torch.FloatTensor(fliter)+torch.FloatTensor(bias)*gamma
                masked_pred_gamma = self.model_to_explain(feats, graph, edge_weights=mask_gamma)
                
                original_pred = self.model_to_explain(feats, graph)

                #if self.type == 'node': # we only care for the prediction of the node
                if self.type == 'node'or self.type == 'ogb_node':
                    masked_pred_gamma = masked_pred_gamma[n].unsqueeze(dim=0)
                    original_pred = original_pred[n]

                id_loss = self.A_loss(masked_pred_gamma, torch.argmax(original_pred).unsqueeze(0), mask_gamma, self.reg_coefs)
                loss += id_loss

            loss.backward()
            optimizer.step()
            
        input_expl = self._create_explainer_input(graph, embeds, indices).unsqueeze(dim=0)
        sampling_weights = self.explainer_model(input_expl)
        mask = self._sample_graph(sampling_weights, training=False).squeeze()


        return graph,mask
    
    def prepare(self, indices=None):
        return 
    
    def explain(self, index):
        return 