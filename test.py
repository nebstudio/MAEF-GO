from graph_data import GoTermDataset, collate_fn
from torch.utils.data import DataLoader
from network import CL_protNET
import torch
from sklearn import metrics
import argparse
import pickle as pkl
from config import get_config
import numpy as np
from joblib import Parallel, delayed
import os
from utils import load_GO_annot


def test(config, task, model_pt, test_type='test',shaixuan=None):
    print(config.device)
    test_set = GoTermDataset(test_type, task,shaixuan=shaixuan)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    output_dim = test_set.y_true.shape[-1]
    model = CL_protNET(
        output_dim,
        cross_att=config.cross_att,
        graph=config.graph,
        fcatNet=config.fcatNet
    ).to(config.device)
    model.load_state_dict(torch.load(model_pt,map_location=config.device))
    model.eval()
    bce_loss = torch.nn.BCELoss()
    
    y_pred_all = []

    y_true_all = test_set.y_true.float()
    with torch.no_grad():
        
        for idx_batch, batch in enumerate(test_loader):

            y_pred = model(batch[0].to(config.device))
            y_pred_all.append(y_pred)
        y_pred_all = torch.cat(y_pred_all, dim=0).cpu()

        # eval_loss = bce_loss(y_pred_all, y_true_all)
        
    
    if test_type == 'AF2test':
        result_name = config.test_result_path + 'AF2'+ model_pt[6:]
    else:
        result_name = config.test_result_path + model_pt[6:]
    result_name = "result_mf.pkl"
    with open(result_name, "wb") as f:
        pkl.dump([y_pred_all.numpy(), y_true_all.numpy()], f)

def str2bool(v):
    if isinstance(v,bool):
        return v
    if v == 'True' or v == 'true':
        return True
    if v == 'False' or v == 'false':
        return False

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--task', type=str, default='mf', choices=['bp','mf','cc'], help='')
    p.add_argument('--device', type=str, default='0', help='')
    p.add_argument('--model', type=str, default='./model/model_mf.pt', help='')
    p.add_argument('--AF2test', default=False, type=str2bool, help='')
    p.add_argument('--cross_att', type=str2bool, default=True, 
                  help='Whether to use cross attention module')
    p.add_argument('--fcatNet', type=str2bool, default=True, 
                  help='Whether to use FECAM module')
    p.add_argument('--graph', type=str, default='both', 
                  choices=['GraphCNN', 'GraphGAT', 'both'],
                  help='Which graph neural network to use')
    p.add_argument('--shaixuan', type=str, default='<95%', 
                  choices=['<30%', '<40%', '<50%', '<70%', '<95%'],
                  help='筛选条件')
    
    
    args = p.parse_args()
    
    print(args)
    config = get_config()
    config.batch_size = 32
    if args.device != '':
        config.device = "cuda:" + args.device
    config.esmembed = args.esmembed
    config.pooling = args.pooling
    config.cross_att = args.cross_att
    config.fcatNet = args.fcatNet
    config.graph = args.graph
    
    if not args.AF2test:
        test(config, args.task, args.model,shaixuan=args.shaixuan)
    else:
        test(config, args.task, args.model, 'AF2test',shaixuan=args.shaixuan)
