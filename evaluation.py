import csv
import pickle
import obonet
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import torch
from sklearn.metrics import average_precision_score as aupr
import pickle as pkl
from utils import load_GO_annot

import seaborn as sns
from matplotlib import pyplot as plt
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.rc('font', family='arial')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# go.obo
go_graph = obonet.read_obo(open("data/go-basic.obo", 'r'))

def bootstrap(Y_true, Y_pred):
    n = Y_true.shape[0]
    idx = np.random.choice(n, n)
    return Y_true[idx], Y_pred[idx]

def load_test_prots(fn):
    proteins = []
    seqid_mtrx = []
    with open(fn, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader, None)
        for row in csv_reader:
            inds = row[1:]
            inds = np.asarray([int(i) for i in inds]).reshape(1, len(inds))
            proteins.append(row[0])
            seqid_mtrx.append(inds)
    return np.asarray(proteins), np.concatenate(seqid_mtrx, axis=0)

def load_go2ic_mapping(fn):
    goterm2ic = {}
    fRead = open(fn, 'r')
    for line in fRead:
        goterm, ic = line.strip().split()
        goterm2ic[goterm] = float(ic)
    fRead.close()
    return goterm2ic

def propagate_go_preds(Y_hat, goterms):
    go2id = {go: ii for ii, go in enumerate(goterms)}
    for goterm in goterms:
        if goterm in go_graph:
            parents = set(goterms).intersection(nx.descendants(go_graph, goterm))
            for parent in parents:
                Y_hat[:, go2id[parent]] = torch.max(Y_hat[:, go2id[goterm]], Y_hat[:, go2id[parent]])
    return Y_hat

def propagate_ec_preds(Y_hat, goterms):
    go2id = {go: ii for ii, go in enumerate(goterms)}
    for goterm in goterms:
        if goterm.find('-') == -1:
            parent = goterm.split('.')
            parent[-1] = '-'
            parent = ".".join(parent)
            if parent in go2id:
                Y_hat[:, go2id[parent]] = torch.max(Y_hat[:, go2id[goterm]], Y_hat[:, go2id[parent]])
    return Y_hat

def normalizedSemanticDistance(Ytrue, Ypred, termIC, avg=False, returnRuMi=False):
    ru = normalizedRemainingUncertainty(Ytrue, Ypred, termIC, False)
    mi = normalizedMisInformation(Ytrue, Ypred, termIC, False)
    sd = torch.sqrt(ru ** 2 + mi ** 2)
    if avg:
        ru = torch.mean(ru)
        mi = torch.mean(mi)
        sd = torch.sqrt(ru ** 2 + mi ** 2)
    if not returnRuMi:
        return sd
    return [ru, mi, sd]

def normalizedRemainingUncertainty(Ytrue, Ypred, termIC, avg=False):
    num = torch.logical_and(Ytrue == 1, Ypred == 0).double().matmul(termIC)
    denom = torch.logical_or(Ytrue == 1, Ypred == 1).double().matmul(termIC)
    nru = num / denom
    if avg:
        nru = torch.mean(nru)
    return nru

def normalizedMisInformation(Ytrue, Ypred, termIC, avg=False):
    num = torch.logical_and(Ytrue == 0, Ypred == 1).double().matmul(termIC)
    denom = torch.logical_or(Ytrue == 1, Ypred == 1).double().matmul(termIC)
    nmi = num / denom
    if avg:
        nmi = torch.mean(nmi)
    return nmi

with open("data/ic_count.pkl", 'rb') as f:
    ic_count = pkl.load(f)
ic_count['bp'] = np.where(ic_count['bp'] == 0, 1, ic_count['bp'])
ic_count['mf'] = np.where(ic_count['mf'] == 0, 1, ic_count['mf'])
ic_count['cc'] = np.where(ic_count['cc'] == 0, 1, ic_count['cc'])
train_ic = {}
train_ic['bp'] = -np.log2(ic_count['bp'] / 69709)
train_ic['mf'] = -np.log2(ic_count['mf'] / 69709)
train_ic['cc'] = -np.log2(ic_count['cc'] / 69709)

class Method(object):
    def __init__(self, method_name, pckl_fn, ont):
        annot = pickle.load(open(pckl_fn, 'rb'))
        _, goterms, gonames, _ = load_GO_annot("data/nrPDB-GO_2019.06.18_annot.tsv")
        self.Y_true = torch.tensor(annot[1], dtype=torch.float64).to(device)
        self.Y_pred = torch.tensor(annot[0], dtype=torch.float64).to(device)
        self.goterms = goterms[ont]
        self.gonames = gonames[ont]
        self.ont = ont
        self.method_name = method_name
        self._propagate_preds()
        if self.ont == 'ec':
            goidx = [i for i, goterm in enumerate(self.goterms) if goterm.find('-') == -1]
            self.Y_true = self.Y_true[:, goidx]
            self.Y_pred = self.Y_pred[:, goidx]
            self.goterms = [self.goterms[idx] for idx in goidx]
            self.gonames = [self.gonames[idx] for idx in goidx]
        self.termIC = torch.tensor(train_ic[self.ont], dtype=torch.float64).to(device)

    def _propagate_preds(self):
        if self.ont == 'ec':
            self.Y_pred = propagate_ec_preds(self.Y_pred, self.goterms)
        else:
            self.Y_pred = propagate_go_preds(self.Y_pred, self.goterms)

    def _cafa_ec_aupr(self, labels, preds):
        n = labels.shape[0]
        goterms = np.asarray(self.goterms)
        prot2goterms = {}
        for i in range(0, n):
            prot2goterms[i] = set(goterms[np.where(labels[i].cpu().numpy() == 1)[0]])
        F_list, AvgPr_list, AvgRc_list, thresh_list = [], [], [], []
        for t in range(1, 100):
            threshold = t / 100.0
            predictions = (preds > threshold).int()
            m, precision, recall = 0, 0.0, 0.0
            for i in range(0, n):
                pred_gos = set(goterms[np.where(predictions[i].cpu().numpy() == 1)[0]])
                num_pred = len(pred_gos)
                num_true = len(prot2goterms[i])
                num_overlap = len(prot2goterms[i].intersection(pred_gos))
                if num_pred > 0:
                    m += 1
                    precision += float(num_overlap) / num_pred
                if num_true > 0:
                    recall += float(num_overlap) / num_true
            if m > 0:
                AvgPr = precision / m
                AvgRc = recall / n
                if AvgPr + AvgRc > 0:
                    F_score = 2 * (AvgPr * AvgRc) / (AvgPr + AvgRc)
                    F_list.append(F_score)
                    AvgPr_list.append(AvgPr)
                    AvgRc_list.append(AvgRc)
                    thresh_list.append(threshold)
        F_list, AvgPr_list, AvgRc_list, thresh_list = map(np.asarray, (F_list, AvgPr_list, AvgRc_list, thresh_list))
        return AvgRc_list, AvgPr_list, F_list, thresh_list

    def _cafa_go_aupr(self, labels, preds):
        n = labels.shape[0]
        goterms = np.asarray(self.goterms)
        ont2root = {'bp': 'GO:0008150', 'mf': 'GO:0003674', 'cc': 'GO:0005575'}
        prot2goterms = {}
        for i in range(0, n):
            all_gos = set()
            for goterm in goterms[np.where(labels[i].cpu().numpy() == 1)[0]]:
                all_gos = all_gos.union(nx.descendants(go_graph, goterm))
                all_gos.add(goterm)
            all_gos.discard(ont2root[self.ont])
            prot2goterms[i] = all_gos
        F_list, AvgPr_list, AvgRc_list, thresh_list = [], [], [], []
        for t in range(1, 100):
            threshold = t / 100.0
            predictions = (preds > threshold).int()
            m, precision, recall = 0, 0.0, 0.0
            for i in range(0, n):
                pred_gos = set()
                for goterm in goterms[np.where(predictions[i].cpu().numpy() == 1)[0]]:
                    pred_gos = pred_gos.union(nx.descendants(go_graph, goterm))
                    pred_gos.add(goterm)
                pred_gos.discard(ont2root[self.ont])
                num_pred = len(pred_gos)
                num_true = len(prot2goterms[i])
                num_overlap = len(prot2goterms[i].intersection(pred_gos))
                if num_pred > 0 and num_true > 0:
                    m += 1
                    precision += float(num_overlap) / num_pred
                    recall += float(num_overlap) / num_true
            if m > 0:
                AvgPr = precision / m
                AvgRc = recall / n
                if AvgPr + AvgRc > 0:
                    F_score = 2 * (AvgPr * AvgRc) / (AvgPr + AvgRc)
                    F_list.append(F_score)
                    AvgPr_list.append(AvgPr)
                    AvgRc_list.append(AvgRc)
                    thresh_list.append(threshold)
        F_list, AvgPr_list, AvgRc_list, thresh_list = map(np.asarray, (F_list, AvgPr_list, AvgRc_list, thresh_list))
        return AvgRc_list, AvgPr_list, F_list, thresh_list

    def _function_centric_aupr(self, keep_pidx=None, keep_goidx=None):
        if keep_pidx is not None:
            Y_true = self.Y_true[keep_pidx]
            Y_pred = self.Y_pred[keep_pidx]
        else:
            Y_true = self.Y_true
            Y_pred = self.Y_pred
        if keep_goidx is not None:
            tmp = []
            for goidx in keep_goidx:
                if Y_true[:, goidx].sum() > 0:
                    tmp.append(goidx)
            keep_goidx = tmp
        else:
            keep_goidx = torch.where(Y_true.sum(axis=0) > 0)[0]
        print(f"### Number of functions = {len(keep_goidx)}")
        Y_true = Y_true[:, keep_goidx]
        Y_pred = Y_pred[:, keep_goidx]
        micro_aupr = aupr(Y_true.cpu(), Y_pred.cpu(), average='micro')
        macro_aupr = aupr(Y_true.cpu(), Y_pred.cpu(), average='macro')
        aupr_goterms = aupr(Y_true.cpu(), Y_pred.cpu(), average=None)
        return micro_aupr, macro_aupr, aupr_goterms

    def _protein_centric_fmax(self, keep_pidx=None):
        if keep_pidx is not None:
            Y_true = self.Y_true[keep_pidx]
            Y_pred = self.Y_pred[keep_pidx]
        else:
            Y_true = self.Y_true
            Y_pred = self.Y_pred
        if self.ont in {'mf', 'bp', 'cc'}:
            Recall, Precision, Fscore, thresholds = self._cafa_go_aupr(Y_true, Y_pred)
        else:
            Recall, Precision, Fscore, thresholds = self._cafa_ec_aupr(Y_true, Y_pred)
        return Fscore, Recall, Precision, thresholds

    def fmax(self, keep_pidx):
        fscore, _, _, _ = self._protein_centric_fmax(keep_pidx=keep_pidx)
        return max(fscore)

    def macro_aupr(self, keep_pidx=None, keep_goidx=None):
        _, macro_aupr, _ = self._function_centric_aupr(keep_pidx=keep_pidx, keep_goidx=keep_goidx)
        return macro_aupr

    def smin(self, keep_pidx=None):
        if keep_pidx is not None:
            Y_true = self.Y_true[keep_pidx]
            Y_pred = self.Y_pred[keep_pidx]
        else:
            Y_true = self.Y_true
            Y_pred = self.Y_pred
        nrThresholds = 100
        thresholds = torch.linspace(0.0, 1.0, nrThresholds, dtype=torch.float64)
        ss = torch.zeros(thresholds.shape, device=device)
        for i, t in enumerate(thresholds):
            ss[i] = normalizedSemanticDistance(Y_true, (Y_pred >= t).int(), self.termIC, avg=True, returnRuMi=False)
        return torch.min(ss).item()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate protein function prediction methods")
    parser.add_argument("--method_name", type=str, default="test", required=False, help="")
    parser.add_argument("--pckl_fn", type=str, default="result_mf.pkl", required=False, help="")
    parser.add_argument("--ont", type=str, default="mf", required=False, help="")
    args = parser.parse_args()
    method = Method(method_name=args.method_name, pckl_fn=args.pckl_fn, ont=args.ont)
    all_proteins_idx = np.arange(method.Y_true.shape[0])
    fmax = method.fmax(keep_pidx=all_proteins_idx)
    print(f"F-max: {fmax:.4f}")
    macro_aupr = method.macro_aupr(keep_pidx=all_proteins_idx)
    print(f"Macro AUPR: {macro_aupr:.4f}")
    smin = method.smin(keep_pidx=all_proteins_idx)
    print(f"S-min: {smin:.4f}")
