from graph_data import GoTermDataset, collate_fn
from torch.utils.data import DataLoader
from network import CL_protNET
import torch.nn.functional as F
import torch
from sklearn import metrics
from utils import log
import argparse
from config import get_config
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def train(config, task, suffix):
    train_set = GoTermDataset("train", task, config.AF2model)
    valid_set = GoTermDataset("val", task, config.AF2model)

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(valid_set, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

    output_dim = valid_set.y_true.shape[-1]
    model = CL_protNET(
        output_dim,
        cross_att=config.cross_att,
        fcatNet=config.fcatNet,
        fecam_open=config.fcatNet
    ).to(config.device)

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        **config.optimizer,
        weight_decay=1e-6
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3,
        min_lr=1e-6, 
        verbose=True
    )

    bce_loss = torch.nn.BCELoss(reduction='mean')

    train_loss = []
    val_loss = []
    val_aupr = []
    y_true_all = valid_set.y_true.float().reshape(-1)

    for ith_epoch in range(config.max_epochs):
        epoch_train_loss = 0.0
        num_batches = 0

        for idx_batch, batch in enumerate(train_loader):
            model.train()

            y_pred = model(batch[0].to(config.device))
            y_true = batch[1].to(config.device)
            loss = bce_loss(y_pred, y_true)
            
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            num_batches += 1
            
        avg_train_loss = epoch_train_loss / num_batches

        eval_loss = 0
        model.eval()

        with torch.no_grad():
            y_pred_all = torch.cat([model(batch[0].to(config.device)) for _, batch in enumerate(val_loader)], dim=0).cpu().reshape(-1)

            eval_loss = bce_loss(y_pred_all, y_true_all).item()
            aupr = metrics.average_precision_score(y_true_all.numpy(), y_pred_all.numpy(), average="samples")
            val_aupr.append(aupr)
            log(f"{ith_epoch} VAL_epoch ||| loss: {round(float(eval_loss),4)} ||| aupr: {round(float(aupr),4)}")

            if ith_epoch == 0 or eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                torch.save(model.state_dict(), config.model_save_path + task + f"{suffix}.pt")
                es = 0
            else:
                es += 1
            patience = 4

            if es > patience - 1:
                print(f"Early stopping triggered after {ith_epoch} epochs")
                break

        scheduler.step(eval_loss)
        torch.cuda.empty_cache()

def str2bool(v):
    if isinstance(v,bool):
        return v
    if v == 'True' or v == 'true':
        return True
    if v == 'False' or v == 'false':
        return False


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--task', type=str, default='mf', choices=['bp', 'mf', 'cc'])
    p.add_argument('--device', type=str, default='1')
    p.add_argument('--AF2model', default=True, type=str2bool)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--cross_att', type=str2bool, default=True)
    p.add_argument('--fcatNet', type=str2bool, default=True)
    p.add_argument('--graph', type=str, default='both', choices=['GraphCNN', 'GraphGAT', 'both'])
    args = p.parse_args()
    config = get_config()
    config.optimizer['lr'] = 1e-4
    config.batch_size = args.batch_size
    config.max_epochs = 100
    config.device = f"cuda:{args.device}"
    config.AF2model = args.AF2model
    config.cross_att = args.cross_att
    config.fecam_open = args.fecam_open
    config.fcatNet = args.fcatNet
    
    train(config, args.task, args.suffix)
