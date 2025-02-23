import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
import pickle as pkl
from utils import load_GO_annot
import numpy as np
import os
from utils import aa2idx
import sys
import esm
import pandas as pd

def collate_fn(batch):
    graphs, y_trues = map(list, zip(*batch))
    return Batch.from_data_list(graphs), torch.stack(y_trues).float()

class GoTermDataset(Dataset):

    def __init__(self, set_type, task, AF2model=False,return_pdbch=False,shaixuan=None):
        # 这次将pdbch_sign = "train" if "train" in set_type else "val" if "test" not in set_type else set_type 改成了：
        #       pdbch_sign = "train" if "train" in set_type else "val" if "test" not in set_type else "test"
        # pdbch_sign = "train" if "train" in set_type else "val" if "test" not in set_type else "test"  
        if "train" in set_type:
            pdbch_sign = "train"
        else:
            if "test" not in set_type:
                pdbch_sign = "val"
            else:
                pdbch_sign = "test"

        # task can be among ['bp','mf','cc']
        self.task = task
        self.return_pdbch = return_pdbch
        if set_type != 'AF2test':
            prot2annot, goterms, gonames, counts = load_GO_annot("data/nrPDB-GO_2019.06.18_annot.tsv")
        else:
            prot2annot, goterms, gonames, counts = load_GO_annot('data/nrSwiss-Model-GO_annot.tsv')
        # prot2annot:三个GO类别的蛋白质标签
        # goterms：三个GO类别中的所有GO术语。
        # gonames：三个GO类别中所有GO术语的名称。
        # counts：三个GO类别相应GO术语在所有蛋白质中出现的次数。
        goterms = goterms[self.task]
        gonames = gonames[self.task]
        output_dim = len(goterms)
        class_sizes = counts[self.task]
        mean_class_size = np.mean(class_sizes)
        pos_weights = mean_class_size / class_sizes#计算每个GO术语的权重
        pos_weights = np.maximum(1.0, np.minimum(10.0, pos_weights)) #所有的权重都被限制在[1,10]
        # pos_weights = np.concatenate([pos_weights.reshape((len(pos_weights), 1)), pos_weights.reshape((len(pos_weights), 1))], axis=-1)
        # give weight for the 0/1 classification
        # pos_weights = {i: {0: pos_weights[i, 0], 1: pos_weights[i, 1]} for i in range(output_dim)}

        self.pos_weights = torch.tensor(pos_weights).float()


        self.processed_dir = 'data/processed'

        self.graph_list = torch.load(os.path.join(self.processed_dir, f"{set_type}_graph.pt")) 
        if set_type == 'AF2test':
            self.pdbch_list = torch.load(os.path.join(self.processed_dir, f"{set_type}_pdbch.pt"))["test_pdbch"]
        else:
            # 3414
            self.pdbch_list = torch.load(os.path.join(self.processed_dir, f"{set_type}_pdbch.pt"))[f"{pdbch_sign}_pdbch"]
        # print(torch.load(os.path.join(self.processed_dir, f"val_pdbch.pt"))["val_pdbch"])
        # print(len(torch.load(os.path.join(self.processed_dir, f"val_pdbch.pt"))["val_pdbch"]))
        # exit(0)
        self.y_true = np.stack([prot2annot[pdb_c][self.task] for pdb_c in self.pdbch_list])
        self.y_true = torch.tensor(self.y_true)
        if AF2model:
            prot2annot, goterms, gonames, counts = load_GO_annot("data/nrSwiss-Model-GO_annot.tsv")
            
            graph_list_af = torch.load(os.path.join(self.processed_dir, f"AF2{set_type}_graph.pt"))
            self.graph_list += graph_list_af
            # 3414
            self.pdbch_list_af = torch.load(os.path.join(self.processed_dir, f"AF2{set_type}_pdbch.pt"))[f"{set_type}_pdbch"]
            y_true_af = np.stack([prot2annot[pdb_c][self.task] for pdb_c in self.pdbch_list_af])
            
            self.y_true = np.concatenate([self.y_true, y_true_af],0)
            # torch.Size([3414, 320])
            self.y_true = torch.tensor(self.y_true)
        
        # shaixuan = "<30%"  # 可选"<30%","<40%","<50%","<70%","<95%"
        if shaixuan is not None:
            assert shaixuan in ["<30%","<40%","<50%","<70%","<95%"], "筛选条件错误,请输入<30%、<40%、<50%、<70%、<95%中的一个"
            csv_path = "data/nrPDB-GO_2019.06.18_test.csv"
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                PDB_chain_from_csv_0 = df[df[shaixuan] == 0]['PDB-chain'].tolist()
                
                index_list_from_pdbch_0 = []
                for i, pdb_chain in enumerate(self.pdbch_list):
                    if pdb_chain in PDB_chain_from_csv_0:
                        index_list_from_pdbch_0.append(i)
                
                for index in sorted(index_list_from_pdbch_0, reverse=True):
                    self.pdbch_list.pop(index)
                    self.graph_list.pop(index)
                    self.y_true = torch.cat([self.y_true[:index], self.y_true[index+1:]], dim=0)
                
                print(f"根据`{shaixuan}`筛选条件删除了{len(index_list_from_pdbch_0)}个样本")
            else:
                print(f"筛选条件`{shaixuan}`对应的CSV文件不存在，将不进行筛选！")
        else:
            print("未指定筛选条件，将不进行筛选！")

    def __getitem__(self, idx):
        if self.return_pdbch:
            return self.graph_list[idx], self.y_true[idx],self.pdbch_list[idx]
        else:
            return self.graph_list[idx], self.y_true[idx]

    def __len__(self):
        return len(self.graph_list)
