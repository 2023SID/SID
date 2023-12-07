import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
from construct_subgraph import construct_mini_batch_giant_graph
from sklearn.metrics import average_precision_score, roc_auc_score

################################################################################################
################################################################################################
################################################################################################

def compute_ap_score(pred_pos, pred_neg, neg_samples):
        y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu().detach()
        y_true = torch.cat([torch.ones_like(pred_pos), torch.zeros_like(pred_neg)], dim=0).cpu().detach()
        acc = average_precision_score(y_true, y_pred)
        if neg_samples > 1:
            auc = torch.sum(pred_pos.squeeze() < pred_neg.squeeze().reshape(neg_samples, -1), dim=0)
            auc = 1 / (auc+1)
        else:
            auc = roc_auc_score(y_true, y_pred)
        return acc, auc 
    
################################################################################################
################################################################################################
################################################################################################
"""
Module: Time-encoder
"""

class TimeEncode(nn.Module):
    """
    out = linear(time_scatter): 1-->time_dims
    out = cos(out)
    """
    def __init__(self, dim):
        super(TimeEncode, self).__init__()
        self.dim = dim
        self.w = nn.Linear(1, dim)
        self.reset_parameters()
    
    def reset_parameters(self, ):
        self.w.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.dim, dtype=np.float32))).reshape(self.dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(self.dim))

        self.w.weight.requires_grad = False
        self.w.bias.requires_grad = False
    
    @torch.no_grad()
    def forward(self, t):
        output = torch.cos(self.w(t.reshape((-1, 1))))
        return output



################################################################################################
################################################################################################
################################################################################################
"""
Module: MLP-Mixer
"""

class FeedForward(nn.Module):
    """
    2-layer MLP with GeLU (fancy version of ReLU) as activation
    """
    def __init__(self, dims, expansion_factor, dropout=0, use_single_layer=False):
        super().__init__()

        self.dims = dims
        self.use_single_layer = use_single_layer
        
        self.expansion_factor = expansion_factor
        self.dropout = dropout

        if use_single_layer:
            self.linear_0 = nn.Linear(dims, dims)
        else:
            self.linear_0 = nn.Linear(dims, int(expansion_factor * dims))
            self.linear_1 = nn.Linear(int(expansion_factor * dims), dims)

        self.reset_parameters()

    def reset_parameters(self):
        self.linear_0.reset_parameters()
        if self.use_single_layer==False:
            self.linear_1.reset_parameters()

    def forward(self, x):
        x = self.linear_0(x)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        if self.use_single_layer==False:
            x = self.linear_1(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class MixerBlock(nn.Module):
    """
    out = X.T + MLP_Layernorm(X.T)     # apply token mixing
    out = out.T + MLP_Layernorm(out.T) # apply channel mixing
    """
    def __init__(self, per_graph_size, dims, 
                 token_expansion_factor=0.5, 
                 channel_expansion_factor=4, 
                 dropout=0, 
                 module_spec=None, use_single_layer=False):
        super().__init__()
        
        if module_spec == None:
            self.module_spec = ['token', 'channel']
        else:
            self.module_spec = module_spec.split('+')

        if 'token' in self.module_spec:
            self.token_layernorm = nn.LayerNorm(dims)
            self.token_forward = FeedForward(per_graph_size, token_expansion_factor, dropout, use_single_layer)
            
        if 'channel' in self.module_spec:
            self.channel_layernorm = nn.LayerNorm(dims)
            self.channel_forward = FeedForward(dims, channel_expansion_factor, dropout, use_single_layer)
        

    def reset_parameters(self):
        if 'token' in self.module_spec:
            self.token_layernorm.reset_parameters()
            self.token_forward.reset_parameters()

        if 'channel' in self.module_spec:
            self.channel_layernorm.reset_parameters()
            self.channel_forward.reset_parameters()
        
    def token_mixer(self, x):
        x = self.token_layernorm(x).permute(0, 2, 1)
        x = self.token_forward(x).permute(0, 2, 1)
        return x
    
    def channel_mixer(self, x):
        x = self.channel_layernorm(x)
        x = self.channel_forward(x)
        return x

    def forward(self, x):
        if 'token' in self.module_spec:
            x = x + self.token_mixer(x)
        if 'channel' in self.module_spec:
            x = x + self.channel_mixer(x)
        return x
    
class FeatEncode(nn.Module):
    """
    Return [raw_edge_feat | TimeEncode(edge_time_stamp)]
    """
    def __init__(self, time_dims, feat_dims, out_dims):
        super().__init__()
        
        self.time_encoder = TimeEncode(time_dims)
        self.feat_encoder = nn.Linear(time_dims + feat_dims, out_dims) 
        self.reset_parameters()

    def reset_parameters(self):
        self.time_encoder.reset_parameters()
        self.feat_encoder.reset_parameters()
        
    def forward(self, edge_feats, edge_ts):
        edge_time_feats = self.time_encoder(edge_ts)
        x = torch.cat([edge_feats, edge_time_feats], dim=1)
        return self.feat_encoder(x.to(torch.float))

class MLPMixer(nn.Module):
    """
    Input : [ batch_size, graph_size, edge_dims+time_dims]
    Output: [ batch_size, graph_size, output_dims]
    """
    def __init__(self, per_graph_size, time_channels,
                 input_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5,
                 token_expansion_factor=0.5, 
                 channel_expansion_factor=4, 
                 module_spec=None, use_single_layer=False
                ):
        super().__init__()
        self.per_graph_size = per_graph_size

        self.num_layers = num_layers
        
        # input & output classifer
        self.feat_encoder = FeatEncode(time_channels, input_channels, hidden_channels)
        self.layernorm = nn.LayerNorm(hidden_channels)
        self.mlp_head = nn.Linear(hidden_channels, out_channels)
        
        # inner layers
        self.mixer_blocks = torch.nn.ModuleList()
        for ell in range(num_layers):
            if module_spec is None:
                self.mixer_blocks.append(
                    MixerBlock(per_graph_size, hidden_channels, 
                               token_expansion_factor, 
                               channel_expansion_factor, 
                               dropout, module_spec=None, 
                               use_single_layer=use_single_layer)
                )
            else:
                self.mixer_blocks.append(
                    MixerBlock(per_graph_size, hidden_channels, 
                               token_expansion_factor, 
                               channel_expansion_factor, 
                               dropout, module_spec=module_spec[ell], 
                               use_single_layer=use_single_layer)
                )



        # init
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.mixer_blocks:
            layer.reset_parameters()
        self.feat_encoder.reset_parameters()
        self.layernorm.reset_parameters()
        self.mlp_head.reset_parameters()

    def forward(self, edge_feats, edge_ts, batch_size, inds,args):
        # x :     [ batch_size, graph_size, edge_dims+time_dims]
        edge_time_feats = self.feat_encoder(edge_feats, edge_ts)

        x = torch.zeros((batch_size * self.per_graph_size,
                         edge_time_feats.size(1))).to(edge_feats.device)
        x[inds] = x[inds] + edge_time_feats
        x = torch.split(x, self.per_graph_size)
        x = torch.stack(x)
        
        # apply to original feats
        for i in range(self.num_layers):
            # apply to channel + feat dim
            x = self.mixer_blocks[i](x)
        x = self.layernorm(x)

        #y = x.view(batch_size * self.per_graph_size, edge_time_feats.size(1))
        y = x

        x = torch.mean(x, dim=1)
        t = x
        x = self.mlp_head(x)
        return x,y,t
    
################################################################################################
################################################################################################
################################################################################################

"""
Edge predictor
"""

class EdgePredictor_per_node(torch.nn.Module):
    """
    out = linear(src_node_feats) + linear(dst_node_feats)
    out = ReLU(out)
    """
    def __init__(self, dim_in_time, dim_in_node,dim_in_conf_node):
        super().__init__()

        self.dim_in_time = dim_in_time
        self.dim_in_node = dim_in_node

        self.src_fc = torch.nn.Linear(dim_in_time + dim_in_node, 100)
        self.dst_fc = torch.nn.Linear(dim_in_time + dim_in_node, 100)
        self.out_fc = torch.nn.Linear(100, 1)
        self.reset_parameters()
        
    def reset_parameters(self, ):
        self.src_fc.reset_parameters()
        self.dst_fc.reset_parameters()
        self.out_fc.reset_parameters()

    def forward(self, h,neg_samples=1):
        num_edge = h.shape[0] // (neg_samples + 3)
        h_pos_src = self.src_fc(h[:num_edge])
        h_pos_dst = self.dst_fc(h[num_edge:num_edge*2])
        h_neg_src = self.src_fc(h[num_edge*2:num_edge*3])
        h_neg_dst = self.dst_fc(h[num_edge*3:])
        h_pos_edge = torch.nn.functional.relu(h_pos_src + h_pos_dst)
        h_neg_edge = torch.nn.functional.relu(h_neg_src.tile(neg_samples, 1) + h_neg_dst)
        # h_pos_edge = torch.nn.functional.relu(h_pos_dst)
        # h_neg_edge = torch.nn.functional.relu(h_neg_dst)
        return self.out_fc(h_pos_edge), self.out_fc(h_neg_edge)
    
class Mixer_per_node(nn.Module):
    """
    Wrapper of MLPMixer and EdgePredictor
    """
    def __init__(self, mlp_mixer_configs, edge_predictor_configs):
        super(Mixer_per_node, self).__init__()

        self.time_feats_dim = edge_predictor_configs['dim_in_time']
        self.node_feats_dim = edge_predictor_configs['dim_in_node']

        if self.time_feats_dim > 0:
            self.base_model = MLPMixer(**mlp_mixer_configs)

        self.edge_predictor = EdgePredictor_per_node(**edge_predictor_configs)
        
        self.creterion = nn.BCEWithLogitsLoss(reduction='none')
        self.reset_parameters()
        self.per_graph_size = mlp_mixer_configs['per_graph_size']
        self.num_hiddens = 100
        self.conf_node_feats_dim = edge_predictor_configs['dim_in_conf_node']
        self.linear = nn.Linear(self.conf_node_feats_dim, self.num_hiddens)
        self.input_dim = mlp_mixer_configs['hidden_channels']

        self.scale = torch.sqrt(torch.tensor(self.input_dim))
        self.Wq = nn.Linear(self.input_dim, self.num_hiddens, bias=True)
        self.Wk = nn.Linear(self.input_dim, self.num_hiddens, bias=True)
        self.linear_no_edge = nn.Linear(self.node_feats_dim, self.input_dim)

        self.NWFM_Wq = nn.Sequential(
            nn.Linear(self.input_dim, self.num_hiddens, bias=True),
            nn.ReLU())
        self.NWGM_Wc = nn.Sequential(
            nn.Linear(self.input_dim, self.num_hiddens, bias=True),
            nn.ReLU())
        self.NWGM_Wc_2 = nn.Sequential(
            nn.Linear(self.node_feats_dim, self.num_hiddens, bias=True),
            nn.ReLU())

        self.NWFM_W_node = nn.Sequential(
            nn.Linear(self.node_feats_dim, self.num_hiddens, bias=True),
            nn.ReLU())
        self.linear_x = nn.Sequential(
            nn.Linear(200, self.time_feats_dim + self.node_feats_dim, bias=True),
            nn.ReLU())
        self.mlp_head = nn.Linear(mlp_mixer_configs['hidden_channels'], mlp_mixer_configs['out_channels'])
    def reset_parameters(self):
        if self.time_feats_dim > 0:
            self.base_model.reset_parameters()
        self.edge_predictor.reset_parameters()
        
    def forward(self, subgraph_datas, neg_samples,edge_feats,args,subgraph_node_feats,confounder):

        num_edge = len(subgraph_datas) // (neg_samples + 2)
        confounder = confounder.to(args.device)

        subgraph_data = construct_mini_batch_giant_graph(subgraph_datas, args.max_edges)
        subgraph_edge_feats = edge_feats[subgraph_data['eid']]
        subgraph_edts = torch.from_numpy(subgraph_data['edts']).float()
        # get mini-batch inds
        all_inds, has_temporal_neighbors = [], []
        # ignore an edge pair if (src_node, dst_node) does not have temporal neighbors
        all_edge_indptr = subgraph_data['all_edge_indptr']
        for i in range(len(all_edge_indptr) - 1):
            num_edges = all_edge_indptr[i + 1] - all_edge_indptr[i]
            all_inds.extend([(args.max_edges * i + j) for j in range(num_edges)])
            has_temporal_neighbors.append(num_edges > 0)
        ###################################################
        model_inputs = [
            subgraph_edge_feats.to(args.device),
            subgraph_edts.to(args.device),
            len(has_temporal_neighbors),
            torch.tensor(all_inds).long()
        ]
        has_temporal_neighbors = [True for _ in range(len(has_temporal_neighbors))]  # not using it

        x, y, t = self.base_model(*model_inputs, args)
        src_H = y[:num_edge]
        dst_pos_H = y[num_edge:num_edge * 2]
        dst_neg_H = y[num_edge * 2:]

        src_M = t[:num_edge]
        dst_pos_M = t[num_edge:num_edge * 2]
        dst_neg_M = t[num_edge * 2:]

        src_pos_score, dst_pos_score, src_neg_score, dst_neg_score = self.get_subgraph_causal_conf_2(src_H, dst_pos_H,
                                                                                                     dst_neg_H, src_M,
                                                                                                     dst_pos_M,
                                                                                                     dst_neg_M)

        # pos_score = (torch.cat([src_pos_score, dst_pos_score, dst_neg_score],dim=0)).view(-1)
        # result_pos = torch.zeros_like(pos_score)
        # result_pos[all_inds] = pos_score[all_inds]
        # pos_edge = y.view(len(subgraph_datas) * self.per_graph_size,100) * result_pos.view(len(subgraph_datas) * self.per_graph_size, 1)
        # pos_edge = torch.split(pos_edge, args.max_edges)
        #
        # result_pos = result_pos.view(len(subgraph_datas),50)
        #
        # pos_edge_res = torch.zeros(num_edge*3,100)
        # num_edges = torch.sum(result_pos!=0, dim=1)
        # n_to_kepp = (num_edges * 0.4).to(torch.int)
        # threshold_values = [(torch.topk(result,k=n, largest=True)).indices for i,(n,result) in enumerate(zip(n_to_kepp,result_pos))]
        # for i, (thresh, pos_ed) in enumerate(zip(threshold_values,pos_edge)):
        #     edges = pos_ed[thresh]
        #     if len(thresh) == 0:
        #         edge = torch.sum(edges, dim=0)
        #
        #     else:
        #         edge = torch.sum(edges, dim=0) / len(thresh)
        #     pos_edge_res[i] = edge.unsqueeze(0)
        #
        # # neg samples
        # neg_score = (torch.cat([src_neg_score, dst_pos_score, dst_neg_score], dim=0)).view(-1)
        # result_neg = torch.zeros_like(neg_score)
        # result_neg[all_inds] = neg_score[all_inds]
        # neg_edge = y.view(len(subgraph_datas) * self.per_graph_size, 100) * result_neg.view(len(subgraph_datas) * self.per_graph_size, 1)
        # neg_edge = torch.split(neg_edge, args.max_edges)
        # result_neg = result_neg.view(len(subgraph_datas), 50)
        # neg_edge_res = torch.empty(0, 100).to(args.device)
        #
        # neg_edge_res = torch.zeros(num_edge * 3, 100)
        # num_edges = torch.sum(result_neg != 0, dim=1)
        # n_to_kepp = (num_edges * 0.4).to(torch.int)
        # threshold_values = [(torch.topk(result, k=n, largest=True)).indices for i, (n, result) in
        #                     enumerate(zip(n_to_kepp, result_neg))]
        # for i, (thresh, neg_ed) in enumerate(zip(threshold_values, neg_edge)):
        #     edges = neg_ed[thresh]
        #     if len(thresh) == 0:
        #         edge = torch.sum(edges, dim=0)
        #
        #     else:
        #         edge = torch.sum(edges, dim=0) / len(thresh)
        #     neg_edge_res[i] = edge.unsqueeze(0)
        #
        #
        #
        # pos_edge_ress = self.mlp_head(pos_edge_res.to('cuda:0'))
        # neg_edge_ress = self.mlp_head(neg_edge_res.to('cuda:0'))
        # pos_edge_res = torch.cat([pos_edge_ress, subgraph_node_feats], dim=1)
        # neg_edge_res = torch.cat([neg_edge_ress, subgraph_node_feats], dim=1)


        # bias
        pos_score = (torch.cat([src_pos_score, dst_pos_score, dst_neg_score], dim=0)).view(-1)
        result_pos = torch.zeros_like(pos_score)
        result_pos[all_inds] = pos_score[all_inds]
        pos_edge = y.view(len(subgraph_datas) * self.per_graph_size, 100) * result_pos.view(
            len(subgraph_datas) * self.per_graph_size, 1)
        pos_edge = torch.split(pos_edge, args.max_edges)
        result_pos = result_pos.view(len(subgraph_datas), 50)

        pos_edge_res = torch.empty(0, 100).to(args.device)
        for result_poses, pos_edgess, subgraph_data in zip(result_pos, pos_edge, subgraph_datas):


            num_edges = subgraph_data['num_edges']
            num_edges_to_select = num_edges-int(num_edges * args.top_percentage)
            indices = torch.where(result_poses != 0)[0]
            values = result_poses[indices]
            sorted_indices = torch.argsort(values, descending=False)
            selected_edge_indices = sorted_indices[:num_edges_to_select]
            edges = pos_edgess[selected_edge_indices]
            if len(selected_edge_indices) == 0:
                edge = torch.sum(edges, dim=0)

            else:
                edge = torch.sum(edges, dim=0) / num_edges_to_select
            pos_edge_res = torch.cat([pos_edge_res, edge.unsqueeze(0)], dim=0)

        neg_score = (torch.cat([src_neg_score, dst_pos_score, dst_neg_score], dim=0)).view(-1)
        result_neg = torch.zeros_like(neg_score)
        result_neg[all_inds] = neg_score[all_inds]
        neg_edge = y.view(len(subgraph_datas) * self.per_graph_size, 100) * result_neg.view(
            len(subgraph_datas) * self.per_graph_size, 1)
        neg_edge = torch.split(neg_edge, args.max_edges)
        result_neg = result_neg.view(len(subgraph_datas), 50)
        neg_edge_res = torch.empty(0, 100).to(args.device)
        for result_neges, neg_edgess, subgraph_data in zip(result_neg, neg_edge, subgraph_datas):


            num_edges = subgraph_data['num_edges']
            num_edges_to_select = num_edges-int(num_edges * args.top_percentage)
            indices = torch.where(result_poses != 0)[0]
            values = result_poses[indices]
            sorted_indices = torch.argsort(values, descending=False)


            selected_edge_indices = sorted_indices[:num_edges_to_select]
            edges = neg_edgess[selected_edge_indices]
            if len(selected_edge_indices) == 0:
                edge = torch.sum(edges, dim=0)
            else:
                edge = torch.sum(edges, dim=0) / num_edges_to_select
            neg_edge_res = torch.cat([neg_edge_res, edge.unsqueeze(0)], dim=0)


        pos_edge_ress = self.mlp_head(pos_edge_res)
        neg_edge_ress = self.mlp_head(neg_edge_res)
        pos_edge_res = torch.cat([pos_edge_ress, subgraph_node_feats], dim=1)
        neg_edge_res = torch.cat([neg_edge_ress, subgraph_node_feats], dim=1)



        x = torch.cat([pos_edge_res[:num_edge*2] , neg_edge_res[:num_edge] , neg_edge_res[num_edge*2:]],dim=0)
        pred_pos, pred_neg = self.edge_predictor(x, neg_samples=neg_samples)


        x = torch.cat([pos_edge_ress[:num_edge], neg_edge_ress[:num_edge],pos_edge_ress[num_edge:2*num_edge], neg_edge_ress[num_edge * 2:]], dim=0)
        subgraph_node_feats = torch.cat(
            [subgraph_node_feats[:num_edge], subgraph_node_feats[:num_edge], subgraph_node_feats[num_edge:2 * num_edge],
             subgraph_node_feats[2 * num_edge:]], dim=0)
        pred_pos_inv1, pred_neg_inv1 = self.predict_inv1(x, neg_samples, subgraph_node_feats,confounder)
        pred_pos_inv2, pred_neg_inv2 = self.predict_inv2(x, neg_samples, subgraph_node_feats,confounder)


        acc, auc = compute_ap_score(pred_pos, pred_neg, neg_samples)
        acc_inv1, auc_inv1 = compute_ap_score(pred_pos_inv1, pred_neg_inv1, neg_samples)
        acc_inv2, auc_inv2 = compute_ap_score(pred_pos_inv2, pred_neg_inv2, neg_samples)


        pos_mask, neg_mask = self.pos_neg_mask(has_temporal_neighbors, neg_samples)
        loss_pos = self.creterion(pred_pos, torch.ones_like(pred_pos)).mean()
        loss_neg = self.creterion(pred_neg, torch.zeros_like(pred_neg)).mean()

        loss_pos_inv1 = self.creterion(pred_pos_inv1, torch.ones_like(pred_pos_inv1)).mean()
        loss_neg_inv1 = self.creterion(pred_neg_inv1, torch.zeros_like(pred_neg_inv1)).mean()

        loss_pos_inv2 = self.creterion(pred_pos_inv2, torch.ones_like(pred_pos_inv2)).mean()
        loss_neg_inv2 = self.creterion(pred_neg_inv2, torch.zeros_like(pred_neg_inv2)).mean()
        all_loss = 10*(loss_pos + loss_neg) + 5*(loss_pos_inv1 + loss_neg_inv1) + 5*(loss_pos_inv2 + loss_neg_inv2)
        
        # compute roc and precision score
        if args.mode == 'test_fid':
            return pred_pos, pred_neg, acc, auc

        return all_loss, acc, auc, acc_inv1, auc_inv1, acc_inv2, auc_inv2
    
    def predict(self, model_inputs, has_temporal_neighbors, neg_samples, node_feats,len_pos,len_neg,args):
        
        if self.time_feats_dim > 0 and self.node_feats_dim == 0:
            x = self.base_model(*model_inputs)
        elif self.time_feats_dim > 0 and self.node_feats_dim > 0:
            x = self.base_model(*model_inputs,args)
            x = torch.cat([x, node_feats], dim=1)
        elif self.time_feats_dim == 0 and self.node_feats_dim > 0:
            x = node_feats
        else:
            print('Either time_feats_dim or node_feats_dim must larger than 0!')
        
        pred_pos, pred_neg = self.edge_predictor(x, len_pos,len_neg,neg_samples=neg_samples)
        return pred_pos, pred_neg

    def predict_inv1(self, x, neg_samples, node_feats,confounder):

        conf_node_feats = self.linear(confounder)
        x_src = torch.cat([x[:1200], node_feats[:1200]], dim=1)
        q_dst, C_dst = self.NWFM_W_node(node_feats[1200:]), self.NWGM_Wc(conf_node_feats)
        attention_scores = torch.matmul(q_dst, C_dst.transpose(0, 1)) / self.scale
        dst_x = torch.matmul(attention_scores, conf_node_feats)
        x_dst = self.linear_x(torch.cat([x[1200:], dst_x], dim=1))
        x = torch.cat([x_src, x_dst], dim=0)

        pred_pos, pred_neg = self.edge_predictor(x, neg_samples=neg_samples)
        return pred_pos, pred_neg

    def predict_inv2(self, x, neg_samples, node_feats,confounder):

        conf_node_feats = self.linear(confounder)
        x_src = torch.cat([x[:1200], node_feats[:1200]], dim=1)
        q_dst, C_dst = self.NWFM_Wq(x[1200:]), self.NWGM_Wc(conf_node_feats)
        attention_scores = torch.matmul(q_dst, C_dst.transpose(0, 1)) / self.scale
        dst_x = torch.matmul(attention_scores, conf_node_feats)
        x_dst = torch.cat([dst_x, node_feats[1200:]], dim=1)
        x = torch.cat([x_src, x_dst], dim=0)

        pred_pos, pred_neg = self.edge_predictor(x, neg_samples=neg_samples)
        return pred_pos, pred_neg
    def pos_neg_mask(self, mask, neg_samples):
        num_edge = len(mask) // (neg_samples + 3)
        src_mask = mask[:num_edge]
        pos_dst_mask = mask[num_edge:2 * num_edge]
        neg_dst_mask = mask[2 * num_edge:]

        pos_mask = [(i and j) for i,j in zip(src_mask, pos_dst_mask)]
        neg_mask = [(i and j) for i,j in zip(src_mask * neg_samples, neg_dst_mask)]
        return pos_mask, neg_mask


    def scaled_dot_product_attention(self, Q, K, scale_factor=1.0):

        Q = Q.unsqueeze(1)
        attention_scores = torch.matmul(Q, K.permute(0, 2, 1)) / scale_factor

        return attention_scores

    def scaled_dot_product_attention_onlyoneedge(self, Q, K, scale_factor=1.0):

        attention_scores = torch.matmul(Q, K.transpose(0, 1)) / scale_factor
        max_value, max_index = torch.max(attention_scores, dim=1)

        return max_value

    def self_attention(self, Q, K):
        queries, keys = self.Wq(Q), self.Wk(K)
        score = self.scaled_dot_product_attention(queries, keys, self.scale)
        return score

    def self_attention_for_only_oneedge(self, src_graph_Hi, dst_graph_Hi):

        queries, keys = self.Wq(src_graph_Hi), self.Wk(dst_graph_Hi)
        score = self.scaled_dot_product_attention_onlyoneedge(queries, keys, self.scale)

        queries, keys = self.Wq(dst_graph_Hi), self.Wk(src_graph_Hi)
        dst_score = self.scaled_dot_product_attention_onlyoneedge(queries, keys, self.scale)

        return score, dst_score

    def get_subgraph_causal_conf_2(self, K_src, K_dst_pos, K_dst_neg, Q_src, Q_dst_pos, Q_dst_neg):

        src_pos_score = self.self_attention(Q_dst_pos, K_src)
        dst_pos_score = self.self_attention(Q_src, K_dst_pos)

        src_neg_score = self.self_attention(Q_dst_neg, K_src)
        dst_neg_score = self.self_attention(Q_src, K_dst_neg)

        return src_pos_score,  dst_pos_score, src_neg_score, dst_neg_score

################################################################################################
################################################################################################
################################################################################################

"""
Module: Node classifier
"""


class NodeClassificationModel(nn.Module):

    def __init__(self, dim_in, dim_hid, num_class):
        super(NodeClassificationModel, self).__init__()
        self.fc1 = torch.nn.Linear(dim_in, dim_hid)
        self.fc2 = torch.nn.Linear(dim_hid, num_class)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        return x