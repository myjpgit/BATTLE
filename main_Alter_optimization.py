import argparse
import copy
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from data_loader import load_data
from model_optimization import GCN, GCL, Encoder_Decoder
from graph_learners import *
from utils import *
from sklearn.cluster import KMeans
import dgl
from torch_geometric.utils import to_dense_adj
import random
import torch.nn as nn
import logging
import os
import sys
from RWR import random_walk_with_restart, gdc

# device=torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device=torch.device("cpu")
EOS = 1e-10

class Experiment:
    def __init__(self):
        super(Experiment, self).__init__()


    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)
        random.seed(seed)
        dgl.seed(seed)
        dgl.random.seed(seed)

    def loss_cls_upper(self, model, Adj, anchor_adj, features, new_embedding, labels, edge_index, adj_original, args):
        new_features, new_structure = model(features, Adj, anchor_adj)
        diff_structure = torch.pow(anchor_adj - new_structure, 2)
        structure_errors = torch.sqrt(torch.sum(diff_structure, 1))
        structure_loss = F.binary_cross_entropy_with_logits(torch.sigmoid(new_structure), torch.tensor(adj_original).to(device))
        criterion = nn.CosineEmbeddingLoss()
        Tar = torch.ones(len(features)).to(device)
        attribute_loss = criterion(new_features, features, Tar)
        s = to_dense_adj(edge_index)[0]
        self.attr_stru_alpha = torch.std(s).detach() / (torch.std(features).detach() + torch.std(s).detach())
        loss = self.attr_stru_alpha * structure_loss + (1-self.attr_stru_alpha) * attribute_loss

        # diff_attribute = torch.pow(features - new_features, 2)
        # attribute_errors = torch.sqrt(torch.sum(diff_attribute, 1))
        # diff_structure = torch.pow(anchor_adj - new_structure, 2)
        # structure_errors = torch.sqrt(torch.sum(diff_structure, 1))
        # s = to_dense_adj(edge_index)[0]
        # self.attr_stru_alpha = torch.std(s).detach() / (torch.std(features).detach() + torch.std(s).detach())
        # if args.attr_stru_alpha:
        #     score = args.attr_stru_alpha * attribute_errors + (1 - args.attr_stru_alpha) * structure_errors
        # else:
        #     score = self.attr_stru_alpha * attribute_errors + (1 - self.attr_stru_alpha) * structure_errors
        #     # print(score)
        # outlier_scores = -score + max(score)
        # loss = torch.mean(score)
        return loss
    
    def loss_cls_lower(self, model, Adj, anchor_adj, features, new_embedding, labels, edge_index, adj_original, args):
        new_features, new_structure = model(features, Adj, anchor_adj)
        diff_attribute = torch.pow(features - new_features, 2)
        attribute_errors = torch.sqrt(torch.sum(diff_attribute, 1))
        diff_structure = torch.pow(anchor_adj - new_structure, 2)
        structure_errors = torch.sqrt(torch.sum(diff_structure, 1))
        s = to_dense_adj(edge_index)[0]
        self.attr_stru_alpha = torch.std(s).detach() / (torch.std(features).detach() + torch.std(s).detach())
        score = self.attr_stru_alpha * attribute_errors + (1 - self.attr_stru_alpha) * structure_errors
        # if args.attr_stru_alpha:
        #     score = args.attr_stru_alpha * attribute_errors + (1 - args.attr_stru_alpha) * structure_errors
        # else:
        #     score = self.attr_stru_alpha * attribute_errors + (1 - self.attr_stru_alpha) * structure_errors
        outlier_scores = -score + max(score)
        # outlier_scores = score
        loss = torch.mean(score)
        # outlier_scores = outlier_scores + labels
        # return loss, outlier_scores
        return loss, outlier_scores

    def delete_anomaly(self, new_embedding, edge_index, adj_original):
        # source_node_embedding = new_embedding[edge_index[0]]
        # target_node_embedding = new_embedding[edge_index[1]]
        # adj_score = source_node_embedding * (target_node_embedding.t())
        adj_score = new_embedding @ new_embedding.T
        adj_score = adj_score.sigmoid() * (torch.from_numpy(adj_original).to(device))
        # adj_score = adj_score.sigmoid()
        return adj_score
    
    def train_Upper(self, optimizer_Upper, optimizer_learner, model_Upper, graph_learner, model_Lower, features, anchor_adj, anchor_Global_Anomaly_adj, anchor_local_Anomaly_adj, labels, edge_index, adj_original, args):
        model_Upper.train()
        graph_learner.train()

        # print('Before Upper Update')
        # print(graph_learner.state_dict()['Adj'][0][0:10])
        # print(model_Upper.state_dict()['encoder.gnn_encoder_layers.1.linear.weight'][3][2:12])
        # print(model_Lower.state_dict()['shared_encoder.layers.1.linear.weight'][5][6:16])          
        
        loss_Upper, Adj, new_embedding = self.loss_gcl(model_Upper, graph_learner, features, anchor_adj, anchor_Global_Anomaly_adj, anchor_local_Anomaly_adj)
        
        Adj = normalize(Adj, 'sym', args.sparse) # normalize anchor_adj
        f_adj = Adj
        new_embeddings = new_embedding.detach()

        if args.sparse:
            f_adj.edata['w'] = f_adj.edata['w'].detach()
        else:
            f_adj = f_adj.detach()

        if args.sparse:
            anchor_adj.edata['w'] = anchor_adj.edata['w'].detach()
        else:
            anchor_adj = anchor_adj.detach()
        
        loss_Lower_Upper = self.loss_cls_upper(model_Lower, f_adj, anchor_adj, features, new_embeddings, labels, edge_index, adj_original, args)

        loss = args.loss_gamma * loss_Upper + (1-args.loss_gamma) * loss_Lower_Upper

        optimizer_Upper.zero_grad()
        optimizer_learner.zero_grad()
        loss.backward()
        optimizer_Upper.step()      
        optimizer_learner.step()  
        # print('After Upper Update')
        # print(graph_learner.state_dict()['Adj'][0][0:10])
        # print(model_Upper.state_dict()['encoder.gnn_encoder_layers.1.linear.weight'][3][2:12])
        # print(model_Lower.state_dict()['shared_encoder.layers.1.linear.weight'][5][6:16])                                           
        
        return loss_Upper, Adj

    def train_Lower(self, optimizer_Lower, model_Upper, graph_learner, model_Lower, features, anchor_adj, anchor_Global_Anomaly_adj, anchor_local_Anomaly_adj, labels, edge_index, adj_original, args):
        model_Lower.train()
        # print('Before Lower Update')
        # print(graph_learner.state_dict()['Adj'][0][0:10])
        # print(model_Upper.state_dict()['encoder.gnn_encoder_layers.1.linear.weight'][3][2:12])
        # print(model_Lower.state_dict()['shared_encoder.layers.1.linear.weight'][5][6:16])
        _, Adj, new_embedding = self.loss_gcl(model_Upper, graph_learner, features, anchor_adj, anchor_Global_Anomaly_adj, anchor_local_Anomaly_adj)
        
        Adj = normalize(Adj, 'sym', args.sparse) # normalize anchor_adj
        f_adj = Adj
        # new_embeddings = new_embedding.detach()

        if args.sparse:
            f_adj.edata['w'] = f_adj.edata['w'].detach()
        else:
            f_adj = f_adj.detach()
        
        loss_Lower, outlier_scores = self.loss_cls_lower(model_Lower, f_adj, anchor_adj, features, new_embedding, labels, edge_index, adj_original, args)
        
        optimizer_Lower.zero_grad()
        loss_Lower.backward()
        optimizer_Lower.step()
        # print('After Lower Update')
        # print(graph_learner.state_dict()['Adj'][0][0:10])
        # print(model_Upper.state_dict()['encoder.gnn_encoder_layers.1.linear.weight'][3][2:12])
        # print(model_Lower.state_dict()['shared_encoder.layers.1.linear.weight'][5][6:16])
        return loss_Lower, outlier_scores


    # def loss_gcl(self, model, graph_learner, features, anchor_adj, anchor_topological_Anomaly_adj, anchor_EdgeAttr_Anomaly_adj):
    def loss_gcl(self, model, graph_learner, features, anchor_adj, anchor_Global_Anomaly_adj, anchor_local_Anomaly_adj):

        # view 1: anchor_topological_Anomaly graph
        if args.maskfeat_rate_anchor:
            mask_v1, _ = get_feat_mask(features, args.maskfeat_rate_anchor)
            features_v1 = features * (1 - mask_v1)
        else:
            features_v1 = copy.deepcopy(features)

        # z1, _ = model(features_v1, anchor_topological_Anomaly_adj, 'anchor')
        z1, _ = model(features_v1, anchor_Global_Anomaly_adj, 'anchor')
        # z1, _ = model(features_v1, anchor_adj, 'anchor')

        # view 2: learned graph
        if args.maskfeat_rate_learner:
            mask, _ = get_feat_mask(features, args.maskfeat_rate_learner)
            features_v2 = features * (1 - mask)
        else:
            features_v2 = copy.deepcopy(features)

        learned_adj = graph_learner(features)
        if not args.sparse:
            learned_adj = symmetrize(learned_adj)
            learned_adj = normalize(learned_adj, 'sym', args.sparse)

        z2, _ = model(features_v2, learned_adj, 'learner')

        # view 3: anchor_Attributes_Anomaly graph
        if args.maskfeat_rate_anchor:
            mask_v3, _ = get_feat_mask(features, args.maskfeat_rate_anchor)  # Data Augmentation
            features_v3 = features * (1 - mask_v3)
        else:
            features_v3 = copy.deepcopy(features)

        z3, _ = model(features_v3, anchor_local_Anomaly_adj, 'anchor')

        # # compute loss
        # if args.contrast_batch_size:
        #     node_idxs = list(range(features.shape[0]))
        #     # random.shuffle(node_idxs)
        #     batches = split_batch(node_idxs, args.contrast_batch_size)
        #     loss = 0
        #     for batch in batches:
        #         weight = len(batch) / features.shape[0]
        #         loss += model.calc_loss(z1[batch], z2[batch]) * weight
        # else:
        #     loss = model.calc_loss(z1, z2)

        # return loss, learned_adj

        # compute loss
        if args.contrast_batch_size:
            node_idxs = list(range(features.shape[0]))
            # random.shuffle(node_idxs)
            batches = split_batch(node_idxs, args.contrast_batch_size)
            loss = 0
            for batch in batches:
                weight = len(batch) / features.shape[0]
                loss += model.calc_loss(z1[batch], z2[batch], z3[batch]) * weight
        else:
            # loss = model.calc_loss(x1, x2) # Calculating contrastive loss
            # loss = model.calc_loss(z1, z2) # Calculating contrastive loss
            loss = model.calc_loss(args, z1, z2, z3)  # Calculating contrastive loss

        # return loss, learned_adj # return the contrastive loss and learned adjacency matrix
        implicit_adj = torch.sigmoid(z2 @ z2.T)
        # implicit_adj = z2 @ z2.T
        return loss, implicit_adj, z2 # return the contrastive loss and learned adjacency matrix


    def train(self, args):

        torch.cuda.set_device(args.gpu)

        # if args.gsl_mode == 'structure_refinement':
        #     features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, adj_original = load_data(args)
        # elif args.gsl_mode == 'structure_inference':
        #     features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, _ = load_data(args)

        data, features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, adj_original, edge_index = load_data(args, 0.1, 0.3)  # 加入了原始图的边索引
        # print(train_mask)
        # print(val_mask)
        # print(test_mask)
        # print(labels)

        test_accuracies = []
        validation_accuracies = []
        test_precision_50 = []
        test_recall_50 = []
        test_precision_100 = []
        test_recall_100 = []
        test_precision_200 = []
        test_recall_200 = []
        test_precision_300 = []
        test_recall_300 = []

        for trial in range(args.ntrials):

            self.setup_seed(trial)

            if args.sparse:
                anchor_adj_raw = adj_original
                anchor_local_Anomaly_adj_raw = (random_walk_with_restart(adj_original, 0.3, args.P)) * edge_feature(args, data.num_nodes, edge_index)
                anchor_Global_Anomaly_adj_raw = gdc(adj_original, args.a) * edge_feature(args, data.num_nodes, edge_index)
                # RWR_adj = random_walk_with_restart(adj_original, 0.1, args.P)
                # anchor_Global_Anomaly_adj_raw = (RWR_adj + adj_original - RWR_adj * adj_original) * edge_feature(args, data.num_nodes, edge_index)
                # anchor_Global_Anomaly_adj_raw = adj_originals
            else:
                anchor_adj_raw = torch.from_numpy(adj_original) 
                anchor_local_Anomaly_adj_raw = torch.from_numpy((random_walk_with_restart(adj_original, 0.3, args.P)) * edge_feature(args, data.num_nodes, edge_index)) #*adj_original
                anchor_Global_Anomaly_adj_raw = torch.from_numpy(gdc(adj_original, args.a) * edge_feature(args, data.num_nodes, edge_index)).float()
                # RWR_adj = random_walk_with_restart(adj_original, 0.1, args.P)
                # anchor_Global_Anomaly_adj_raw = torch.from_numpy((RWR_adj + adj_original - RWR_adj * adj_original) * edge_feature(args, data.num_nodes, edge_index))
                # print(anchor_Global_Anomaly_adj_raw)
                # anchor_Global_Anomaly_adj_raw = torch.from_numpy(adj_original)
            
            anchor_adj = normalize(anchor_adj_raw, 'sym', args.sparse) # normalize anchor_adj
            anchor_local_Anomaly_adj = normalize(anchor_local_Anomaly_adj_raw, 'sym', args.sparse)
            anchor_Global_Anomaly_adj = normalize(anchor_Global_Anomaly_adj_raw, 'sym', args.sparse)

            if args.sparse:
                anchor_adj_torch_sparse = copy.deepcopy(anchor_adj)  # Deep copy of anchor_adj
                anchor_adj = torch_sparse_to_dgl_graph(anchor_adj)
                anchor_local_Anomaly_adj_torch_sparse = copy.deepcopy(anchor_local_Anomaly_adj)
                anchor_local_Anomaly_adj = torch_sparse_to_dgl_graph(anchor_local_Anomaly_adj)
                anchor_Global_Anomaly_adj_torch_sparse = copy.deepcopy(anchor_Global_Anomaly_adj)
                anchor_Global_Anomaly_adj = torch_sparse_to_dgl_graph(anchor_Global_Anomaly_adj)

            if args.type_learner == 'fgp':
                graph_learner = FGP_learner(features.cpu(), args.k, args.sim_function, 6, args.sparse)
            elif args.type_learner == 'mlp':
                graph_learner = MLP_learner(2, features.shape[1], args.k, args.sim_function, 6, args.sparse,
                                     args.activation_learner)
            elif args.type_learner == 'att':
                graph_learner = ATT_learner(2, features.shape[1], args.k, args.sim_function, 6, args.sparse,
                                          args.activation_learner)
            # elif args.type_learner == 'gnn':
            #     graph_learner = GNN_learner(2, features.shape[1], args.k, args.sim_function, 6, args.sparse,
            #                          args.activation_learner, torch_sparse_to_dgl_graph(anchor_adj))
            
            anchor_adj = anchor_adj.to(device)
            adj = anchor_adj
            
            if args.sparse:
                adj.edata['w'] = adj.edata['w'].detach()
            else:
                adj = adj.detach()

            model_Upper = GCL(nlayers=args.nlayers, in_dim=nfeats, hidden_dim=args.hidden_dim,
                         emb_dim=args.rep_dim, proj_dim=args.proj_dim,
                         dropout=args.dropout, dropout_adj=args.dropedge_rate, sparse=args.sparse)


            model_Lower = Encoder_Decoder(in_dim=nfeats, hid_dim=args.hidden_dim_cls, out_dim=args.proj_dim, num_layers=args.nlayers_cls, dropout=args.dropout_cls,
                                dropout_adj=args.dropedge_cls, adj = adj, act=F.relu, sparse=args.sparse)

            optimizer_Upper = torch.optim.Adam(model_Upper.parameters(), lr=args.lr, weight_decay=args.w_decay)
            optimizer_learner = torch.optim.Adam(graph_learner.parameters(), lr=args.lr, weight_decay=args.w_decay)
            optimizer_Lower = torch.optim.Adam(model_Lower.parameters(), lr=args.lr_cls, weight_decay=args.w_decay_cls)

            if torch.cuda.is_available():
                model_Upper = model_Upper.to(device)
                model_Lower = model_Lower.to(device)
                graph_learner = graph_learner.to(device)
                # train_mask = train_mask.to(device)
                # val_mask = val_mask.to(device)
                # test_mask = test_mask.to(device)
                features = features.to(device)
                labels = labels.to(device)
                if not args.sparse:
                    anchor_adj = anchor_adj.to(device)
                    anchor_local_Anomaly_adj = anchor_local_Anomaly_adj.to(device)
                    anchor_Global_Anomaly_adj = anchor_Global_Anomaly_adj.to(device)

            if args.downstream_task == 'classification':
                best_val = 0
                best_val_test = 0
                best_precision_test_50 = 0
                best_recall_test_50 = 0
                best_precision_test_100 = 0
                best_recall_test_100 = 0
                best_precision_test_200 = 0
                best_recall_test_200 = 0
                best_precision_test_300 = 0
                best_recall_test_300 = 0
                best_epoch = 0

            for epoch in range(1, args.epochs + 1):


                loss_Lower, outlier_scores = self.train_Lower(optimizer_Lower, model_Upper, graph_learner, model_Lower, features, anchor_adj, anchor_Global_Anomaly_adj, anchor_local_Anomaly_adj, labels, edge_index, adj_original, args)
                loss_Upper, Adj = self.train_Upper(optimizer_Upper, optimizer_learner, model_Upper, graph_learner, model_Lower, features, anchor_adj, anchor_Global_Anomaly_adj, anchor_local_Anomaly_adj, labels, edge_index, adj_original, args)
                
                auc = evaluate_AUC(torch.tensor(outlier_scores), torch.tensor(labels))
                precision50 = evaluate_precision(torch.tensor(outlier_scores), torch.tensor(labels), 50)
                recall50 = evaluate_recall(torch.tensor(outlier_scores), torch.tensor(labels), 50)
                precision100 = evaluate_precision(torch.tensor(outlier_scores), torch.tensor(labels), 100)
                recall100 = evaluate_recall(torch.tensor(outlier_scores), torch.tensor(labels), 100)
                precision200 = evaluate_precision(torch.tensor(outlier_scores), torch.tensor(labels), 200)
                recall200 = evaluate_recall(torch.tensor(outlier_scores), torch.tensor(labels), 200)
                precision300 = evaluate_precision(torch.tensor(outlier_scores), torch.tensor(labels), 300)
                recall300 = evaluate_recall(torch.tensor(outlier_scores), torch.tensor(labels), 300)

                # Structure Bootstrapping (大改)
                #(new_embedding 计算每条边的概率（ai,j = sigmoid(zi, zj)）然后在原始图中删除得分最低的n条边)
                if (1 - args.tau) and (args.c == 0 or epoch % args.c == 0):
                    if args.sparse:
                        # delete_anomaly_adj = self.delete_anomaly(new_embedding, edge_index, adj_original)
                        # delete_anomaly_adj_torch_sparse = dgl_graph_to_torch_sparse(delete_anomaly_adj)
                        learned_adj_torch_sparse = dgl_graph_to_torch_sparse(Adj)
                        anchor_local_Anomaly_adj_torch_sparse = anchor_local_Anomaly_adj_torch_sparse * args.tau \
                                                  + learned_adj_torch_sparse * (1 - args.tau)
                        anchor_local_Anomaly_adj = torch_sparse_to_dgl_graph(anchor_local_Anomaly_adj_torch_sparse)

                        anchor_Global_Anomaly_adj_torch_sparse = anchor_Global_Anomaly_adj_torch_sparse * args.tau \
                                                  + learned_adj_torch_sparse * (1 - args.tau)
                        anchor_Global_Anomaly_adj = torch_sparse_to_dgl_graph(anchor_Global_Anomaly_adj_torch_sparse)

                    else:
                        # delete_anomaly_adj = self.delete_anomaly(new_embedding, edge_index, adj_original)
                        # delete_anomaly_topological_adj = self.delete_anomaly(anchor_topological_Anomaly_adj_embedding, edge_index, adj_original)
                        # delete_anomaly_EdgeAttr_adj = self.delete_anomaly(anchor_EdgeAttr_Anomaly_adj_embedding, edge_index, adj_original)

                        # delete_anomaly_topological_adj = normalize(delete_anomaly_topological_adj, 'sym', args.sparse)
                        # delete_anomaly_EdgeAttr_adj = normalize(delete_anomaly_EdgeAttr_adj, 'sym', args.sparse)

                        anchor_local_Anomaly_adj = anchor_local_Anomaly_adj * args.tau + Adj.detach() * (1 - args.tau)
                        anchor_Global_Anomaly_adj = anchor_Global_Anomaly_adj * args.tau + Adj.detach() * (1 - args.tau)
                        # anchor_EdgeAttr_Anomaly_adj = anchor_EdgeAttr_Anomaly_adj * args.tau + Adj.detach() * (1 - args.tau)

                print("Epoch {:05d} | Upper Loss {:.4f} | Lower Loss {:.4f}".format(epoch, loss_Upper.item(), loss_Lower.item()))
                print("Test_AUC {:.4f} | Test_precision_50 {:.4f} | Test_recall_50 {:.4f} | Test_precision_100 {:.4f} | Test_recall_100 {:.4f} | Test_precision_200 {:.4f} | Test_recall_200 {:.4f} | Test_precision_300 {:.4f} | Test_recall_300 {:.4f}".format(auc, precision50, recall50, precision100, recall100, precision200, recall200, precision300, recall300))


                if auc > best_val_test:
                    best_val_test = auc
                if precision50 > best_precision_test_50:
                    # best_val = val_auc
                    best_precision_test_50 = precision50
                    best_recall_test_50 = recall50
                if precision100 > best_precision_test_100:
                    best_precision_test_100 = precision100
                    best_recall_test_100 = recall100
                if precision200 > best_precision_test_200:
                    best_precision_test_200 = precision200
                    best_recall_test_200 = recall200
                if precision300 > best_precision_test_300:
                    best_precision_test_300 = precision300
                    best_recall_test_300 = recall300
                    best_epoch = epoch

            if args.downstream_task == 'classification':
                # validation_accuracies.append(best_val.item())
                test_accuracies.append(best_val_test.item())
                test_precision_50.append(best_precision_test_50.item())
                test_recall_50.append(best_recall_test_50.item())
                test_precision_100.append(best_precision_test_100.item())
                test_recall_100.append(best_recall_test_100.item())
                test_precision_200.append(best_precision_test_200.item())
                test_recall_200.append(best_recall_test_200.item())
                test_precision_300.append(best_precision_test_300.item())
                test_recall_300.append(best_recall_test_300.item())
                print("Trial: ", trial + 1)
                # print("Best val ACC: ", best_val.item())
                print("Best test ACC: ", best_val_test.item())
                print("Best test Precision@50: ", best_precision_test_50.item())
                print("Best test Recall@50: ", best_recall_test_50.item())
                print("Best test Precision@100: ", best_precision_test_100.item())
                print("Best test Recall@100: ", best_recall_test_100.item())
                print("Best test Precision@200: ", best_precision_test_200.item())
                print("Best test Recall@200: ", best_recall_test_200.item())
                print("Best test Precision@300: ", best_precision_test_300.item())
                print("Best test Recall@300: ", best_recall_test_300.item())
            elif args.downstream_task == 'clustering':
                print("Final ACC: ", acc)
                print("Final NMI: ", nmi)
                print("Final F-score: ", f1)
                print("Final ARI: ", ari)

        if args.downstream_task == 'classification' and trial != 0:
            self.print_results(test_accuracies, test_precision_50, test_recall_50, test_precision_100, test_recall_100, test_precision_200, test_recall_200, test_precision_300, test_recall_300)


    def print_results(self, test_accu, test_pre_50, test_re_50, test_pre_100, test_re_100, test_pre_200, test_re_200, test_pre_300, test_re_300):
        # s_val = "Val accuracy: {:.4f} +/- {:.4f}".format(np.mean(validation_accu), np.std(validation_accu))
        s_test = "Test accuracy: {:.4f} +/- {:.4f}".format(np.mean(test_accu),np.std(test_accu))
        s_pre_50 = "Test precision@50: {:.4f} +/- {:.4f}".format(np.mean(test_pre_50),np.std(test_pre_50))
        s_recall_50 = "Test recall@50: {:.4f} +/- {:.4f}".format(np.mean(test_re_50),np.std(test_re_50))
        s_pre_100 = "Test precision@100: {:.4f} +/- {:.4f}".format(np.mean(test_pre_100),np.std(test_pre_100))
        s_recall_100 = "Test recall@100: {:.4f} +/- {:.4f}".format(np.mean(test_re_100),np.std(test_re_100))
        s_pre_200 = "Test precision@200: {:.4f} +/- {:.4f}".format(np.mean(test_pre_200),np.std(test_pre_200))
        s_recall_200 = "Test recall@200: {:.4f} +/- {:.4f}".format(np.mean(test_re_200),np.std(test_re_200))
        s_pre_300 = "Test precision@300: {:.4f} +/- {:.4f}".format(np.mean(test_pre_300),np.std(test_pre_300))
        s_recall_300 = "Test recall@300: {:.4f} +/- {:.4f}".format(np.mean(test_re_300),np.std(test_re_300))
        print(s_test)
        print(s_pre_50)
        print(s_recall_50)
        print(s_pre_100)
        print(s_recall_100)
        print(s_pre_200)
        print(s_recall_200)
        print(s_pre_300)
        print(s_recall_300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Experimental setting
    parser.add_argument('-dataset', type=str, default='BlogCatalog',
                        choices=['ACM', 'BlogCatalog', 'Flickr', 'Cora'])
    parser.add_argument('-ntrials', type=int, default=5)
    parser.add_argument('-sparse', type=int, default=0)
    parser.add_argument('-gsl_mode', type=str, default="structure_inference",
                        choices=['structure_inference', 'structure_refinement'])
    parser.add_argument('-eval_freq', type=int, default=5)
    parser.add_argument('-downstream_task', type=str, default='classification',
                        choices=['classification', 'clustering'])
    parser.add_argument('-gpu', type=int, default=0)

    # GCL Module - Framework
    parser.add_argument('-epochs', type=int, default=1000)
    parser.add_argument('-lr', type=float, default=0.01)
    parser.add_argument('-w_decay', type=float, default=0.0)
    parser.add_argument('-hidden_dim', type=int, default=512)
    parser.add_argument('-rep_dim', type=int, default=64)
    parser.add_argument('-proj_dim', type=int, default=64)
    parser.add_argument('-dropout', type=float, default=0.5)
    parser.add_argument('-contrast_batch_size', type=int, default=0)
    parser.add_argument('-nlayers', type=int, default=2)

    # GCL Module -Augmentation
    parser.add_argument('-maskfeat_rate_learner', type=float, default=0.0)
    parser.add_argument('-maskfeat_rate_anchor', type=float, default=0.0)
    parser.add_argument('-dropedge_rate', type=float, default=0.5)

    # GSL Module
    parser.add_argument('-type_learner', type=str, default='fgp', choices=["fgp", "att", "mlp", "gnn"])
    parser.add_argument('-a', type=float, default=0.2)
    parser.add_argument('-P', type=int, default=20) # Third
    parser.add_argument('-k', type=int, default=30)
    parser.add_argument('-sim_function', type=str, default='cosine', choices=['cosine', 'minkowski'])
    parser.add_argument('-gamma', type=float, default=0.9)
    parser.add_argument('-activation_learner', type=str, default='relu', choices=["relu", "tanh"])
    parser.add_argument('-loss_alpha', type=float, default=0.5)
    parser.add_argument('-loss_gamma', type=float, default=0.8)
    parser.add_argument('-attr_stru_alpha', type=float, default=0)
    parser.add_argument('-edge_featurte', type=str, default='mean', choices=["degree_i", "degree_j", "common_neighbor", "admic_adar", "jaccard", "PA", "mean"])

    # Evaluation Network (Classification)
    parser.add_argument('-epochs_cls', type=int, default=200)
    parser.add_argument('-lr_cls', type=float, default=0.001)
    parser.add_argument('-w_decay_cls', type=float, default=0.0005)
    parser.add_argument('-hidden_dim_cls', type=int, default=32)
    parser.add_argument('-dropout_cls', type=float, default=0.5)
    parser.add_argument('-dropedge_cls', type=float, default=0.25)
    parser.add_argument('-nlayers_cls', type=int, default=2)
    parser.add_argument('-patience_cls', type=int, default=10)

    # Structure Bootstrapping
    parser.add_argument('-tau', type=float, default=1)
    parser.add_argument('-c', type=int, default=0)

    args = parser.parse_args()

    experiment = Experiment()
    experiment.train(args)
