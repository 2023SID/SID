import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import auc
import torch
import argparse
from train import load_model,load_all_data,load_ori_model
from link_pred_train_utils import run_for_fid, run_for_ori
from data_process_utils import pre_compute_subgraphs


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='WIKI_T')
    parser.add_argument('--num_neighbors', type=int, default=50)  # hyper-parameters K
    parser.add_argument('--sampled_num_hops', type=int, default=5)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--structure_hops', type=int, default=2)

    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=600)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--max_edges', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=1e-6)

    parser.add_argument('--model', type=str, default='mlp_mixer')
    parser.add_argument('--neg_samples', type=int, default=1)
    parser.add_argument('--extra_neg_samples', type=int, default=5)


    parser.add_argument('--hidden_dims', type=int, default=100)

    parser.add_argument('--regen_models', action='store_true')
    parser.add_argument('--check_data_leakage', action='store_true')

    parser.add_argument('--ignore_node_feats', action='store_true')
    parser.add_argument('--node_feats_as_edge_feats', action='store_true')
    parser.add_argument('--ignore_edge_feats', action='store_true')
    parser.add_argument('--use_onehot_node_feats', action='store_true', default=True)

    parser.add_argument('--use_graph_structure', action='store_true', default=True)
    parser.add_argument('--structure_time_gap', type=int, default=2000)  # hyper-parameters T


    parser.add_argument('--use_node_cls', action='store_true')
    parser.add_argument('--use_cached_subgraph', action='store_true', default=True)

    # 后来自己加的
    parser.add_argument('--top_percentage', type=float, default=0.8)
    parser.add_argument('--mode', action='store')
    return parser.parse_args()

args = get_args()
args.mode = 'test_fid'
args.regen_models = True
args.use_graph_structure = True
args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
args.device = torch.device(args.device)
node_feats, edge_feats,conf_node_feats, g, df, args = load_all_data(args)



test_subgraphs  = pre_compute_subgraphs(args, g, df, mode='test' )



args.top_percentage = 0.
model_top1, args = load_model(args)
model_top1.load_state_dict(torch.load(args.model_fn))
model_top1 = model_top1.to(args.device)
all_pred_pos_oris = []
with torch.no_grad():
    all_pred_pos_ori,all_pred_neg_ori,ap_ori,auc_ori = run_for_fid(model_top1, args, test_subgraphs, df, node_feats, edge_feats,conf_node_feats,'test')

pred_ori_result = [torch.cat((a,b),dim=0) for a, b in zip(all_pred_pos_ori,all_pred_neg_ori)]


sparsity = [0,0.2,0.4,0.6,0.8,1.0]
# get the result of our model
pred_pos_result = []
pred_neg_result = []
ap_result = []
auc_result = []
for spar in sparsity:
    args.top_percentage = spar
    model_our, args = load_model(args)
    model_our.load_state_dict(torch.load(args.model_fn))
    model_our = model_our.to(args.device)
    n=500
    start_time = time.time()
    for i in range(n):
        with torch.no_grad():
            all_pred_pos_our,all_pred_neg_our,ap_our,auc_our = run_for_fid(model_our, args, test_subgraphs, df, node_feats, edge_feats, conf_node_feats, 'test')
    end_time = time.time()
    execution_time = end_time - start_time
    print(execution_time)
    sum = 0
    for sub in test_subgraphs:
        sum += len(sub) / 3 * 2
    print("sum:%f" % sum)
    print(execution_time/sum)
    print(60/(execution_time/n/sum))

    pred_pos_result.append(all_pred_pos_our)
    pred_neg_result.append(all_pred_neg_our)
    ap_result.append(ap_our)
    auc_result.append(auc_our)

fidelity_ap = []

for i in range(6):
     fid_ap = [(b-a) for a,b in zip(ap_result[i], ap_ori)]
     fidelity_ap.append(np.array(fid_ap).mean())


# Calculate the AUC value
auc_value = auc(sparsity, fidelity_ap)
plt.plot(sparsity, fidelity_ap, label='FID Inverse vs. Sparsity')
plt.xlabel('Sparsity')
plt.ylabel('Fidelity')
plt.show()
print(f"AUFSC: {auc_value}")

max_fid = max(fidelity_ap)
index = fidelity_ap.index(max_fid)
print(f"Best Fid: {max_fid}")
print(f"sparsity: {sparsity[index]}")