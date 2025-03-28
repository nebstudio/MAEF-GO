import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import to_dense_batch, subgraph, k_hop_subgraph
from torch_geometric.nn import global_max_pool as gmp, global_mean_pool as gap
import torch_dct as dct
class CrossAttention(nn.Module):
    def __init__(self, in_dim1, in_dim2, k_dim=32, v_dim=32, num_heads=8):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim

        self.proj_q1 = nn.Linear(in_dim1, k_dim * num_heads, bias=False)
        self.proj_k2 = nn.Linear(in_dim2, k_dim * num_heads, bias=False)
        self.proj_v2 = nn.Linear(in_dim2, v_dim * num_heads, bias=False)
        self.proj_o = nn.Linear(v_dim * num_heads, in_dim1)

    def forward(self, x1, x2, mask=None):
        batch_size, seq_len1, in_dim1 = x1.size()
        seq_len2 = x2.size(1)

        q1 = self.proj_q1(x1).view(batch_size, seq_len1, self.num_heads, self.k_dim).permute(0, 2, 1, 3)
        k2 = self.proj_k2(x2).view(batch_size, seq_len2, self.num_heads, self.k_dim).permute(0, 2, 3, 1)
        v2 = self.proj_v2(x2).view(batch_size, seq_len2, self.num_heads, self.v_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q1, k2) / self.k_dim ** 0.5  # [batch_size, num_heads, seq_len1, seq_len2]
        
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1)  # [batch_size, num_heads, seq_len2]
            mask = mask.unsqueeze(2)  # [batch_size, num_heads, 1, seq_len2]
            attn = attn.masked_fill(mask == 0, -1e9)
            
        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v2).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len1, -1)
        output = self.proj_o(output)
        return output

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.ReLU(inplace=True)
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = nn.ModuleList([CrossAttention(
            in_dim1=hidden_size,
            in_dim2=hidden_size,
            k_dim=hidden_size // num_heads,
            v_dim=hidden_size // num_heads,
            num_heads=num_heads
        ) for _ in range(2)])
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, kv, attn_bias=None, mask=None):
        y = self.self_attention_norm(x)
        kv = self.self_attention_norm(kv)
        for i in range(2):
            y = self.self_attention[i](y, kv, mask=attn_bias) + y
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim, bias=True), nn.ReLU(inplace=True),
                                 nn.Linear(hidden_dim, output_dim, bias=True), nn.ReLU(inplace=True))
        
    def forward(self, x):
        return self.mlp(x)
class GraphGAT(nn.Module):
    def __init__(self, channel_dims=[256, 256, 256], fc_dim=512, num_classes=256):
        super(GraphGAT, self).__init__()
        gcn_dims = [512] + channel_dims

        gcn_layers = [GATConv(gcn_dims[i-1], gcn_dims[i], bias=True) for i in range(1, len(gcn_dims))]

        self.gcn = nn.ModuleList(gcn_layers)
        self.drop1 = nn.Dropout(p=0.2)
    

    def forward(self, x, data, pertubed=False):
        x = self.drop1(x)
        for idx, gcn_layer in enumerate(self.gcn):
            if idx == 0:
                x = F.relu(gcn_layer(x, data.edge_index.long()))
            else:
                x = x + F.relu(gcn_layer(x, data.edge_index.long())) #16906, 512
        return  x
    
class GraphCNN(nn.Module):
    def __init__(self, channel_dims=[256, 256, 256], fc_dim=512, num_classes=256):
        super(GraphCNN, self).__init__()

        gcn_dims = [512] + channel_dims

        gcn_layers = [GCNConv(gcn_dims[i-1], gcn_dims[i], bias=True) for i in range(1, len(gcn_dims))]

        self.gcn = nn.ModuleList(gcn_layers)
        self.drop1 = nn.Dropout(p=0.2)
    

    def forward(self, x, data, pertubed=False):
        x = self.drop1(x) #[16906, 512]
        for idx, gcn_layer in enumerate(self.gcn):
            if idx == 0:
                x = F.relu(gcn_layer(x, data.edge_index.long()))
            else:
                x = x + F.relu(gcn_layer(x, data.edge_index.long())) #16906, 512
        return  x
class dct_channel_block(nn.Module):
    def __init__(self, channel,is_begin_downsampling=False):
        super(dct_channel_block, self).__init__()
        self.is_begin_downsampling = is_begin_downsampling
        self.fc = nn.Sequential(
            nn.Linear(channel, channel * 4),
            nn.LayerNorm(channel * 4),
            nn.Dropout(p=0.1),
            nn.GELU(),
            nn.Linear(channel * 4, channel * 2),
            nn.LayerNorm(channel * 2),
            nn.Dropout(p=0.1),  
            nn.GELU(),
            nn.Linear(channel * 2, channel),
            nn.Dropout(p=0.1),  
            nn.Sigmoid()
        )

        if self.is_begin_downsampling:
            self.begin_downsampling = nn.Sequential(
                nn.Conv1d(channel * 3, channel, 1),
                nn.BatchNorm1d(channel),
                nn.Dropout(p=0.1),  
                nn.GELU()
            )

        self.dct_norm = nn.LayerNorm(512, eps=1e-6)
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)
        self.weights = nn.Parameter(torch.tensor([5.0, 1.0]))
        
    def forward(self, x):
        if self.is_begin_downsampling:
            x = self.begin_downsampling(x)

        b, c, l = x.size()  # [B, C, L]
    
        freq = dct.dct_2d(x).permute(0, 2, 1)  # torch.Size([64, 512, 745])  
        lr_weight = self.dct_norm(freq)  # [B, L, C]
        lr_weight = lr_weight.permute(0, 2, 1)  # [B, C, L]
        lr_weight = self.fc(lr_weight.permute(0, 2, 1)).permute(0, 2, 1)  # [B, C, L]
        w1, w2 = torch.softmax(self.weights, dim=0)
        enhanced = w1 * x + w2 * x * (1.0 + torch.sigmoid(self.alpha) * lr_weight)
        
        return enhanced
class GatedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(GatedConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.gate_conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        conv_out = self.conv(x)
        gate_out = self.sigmoid(self.gate_conv(x))

        return conv_out * gate_out

class CL_protNET(torch.nn.Module):
    def __init__(self, out_dim, cross_att=True, graph_mode="both",fcatNet=True):
        super(CL_protNET,self).__init__()
        self.out_dim = out_dim
        self.pool1=gmp
        
        if graph_mode in ["GraphCNN"]:
            self.gcn1 = GraphCNN([512, 512, 512])
        if graph_mode in ["GraphGAT"]:
            self.gcn2 = GraphGAT([512, 512, 512]) 
        if graph_mode in ["both"]:
            self.gcn1 = GraphCNN()
            self.gcn2 = GraphGAT()  
        self.seqconv = nn.Conv1d(1280, 512, kernel_size=3, padding=1)
        self.seqbn = nn.BatchNorm1d(512)
        self.seqact = nn.GELU()
        

        self.cross_att = cross_att
        self.graph_mode = graph_mode
        self.fcatNet = fcatNet  
        

        self.gatedConv1 = GatedConv1d(1280, 512, kernel_size=3, padding=1)
        self.gatedConv2 = GatedConv1d(1280, 512, kernel_size=5, padding=2)
        self.gatedConv3 = GatedConv1d(1280, 512, kernel_size=7, padding=3)
        

        if not self.fcatNet:
            self.fusion = nn.Sequential(
                nn.Linear(512 * 3, 512),
                nn.GELU()
            )
        
        self.bnProtein = nn.BatchNorm1d(512 * 3 if self.fcatNet else 512)
        self.act = nn.GELU()
        if self.fcatNet:
            self.fecam = dct_channel_block(512,True if self.fcatNet else False)
        
        
        self.smi_attention_poc= EncoderLayer(512, 512, 0.1, 0.1, 2)
        self.w1 = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        self.global_maxpool1d = nn.AdaptiveMaxPool1d((1))
        self.layer_norm = nn.LayerNorm(512)
        self.dropout = nn.Dropout(0.1)
        self.readout = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, out_dim),
            nn.Sigmoid()
        )

     
    def forward(self, data):
        seqGraph = data.x.float()
        seqGraph, mask1 = to_dense_batch(seqGraph, data.batch)
        seqGraph = seqGraph.permute(0, 2, 1)
        seqGraph = self.seqconv(seqGraph)
        seqGraph = self.seqbn(seqGraph)
        seqGraph = self.seqact(seqGraph)
        seqGraph = seqGraph.permute(0, 2, 1)
        struInput = seqGraph[mask1.bool()]
        if self.graph_mode == "GraphCNN":
            struGCN = self.gcn1(struInput, data)
            struGCN, mask2 = to_dense_batch(struGCN, data.batch)
            graph = struGCN
        elif self.graph_mode == "GraphGAT":
            struGAT = self.gcn2(struInput, data)
            struGAT, mask2 = to_dense_batch(struGAT, data.batch)
            graph = struGAT
        else:  # "both"
            struGCN = self.gcn1(struInput, data)
            struGAT = self.gcn2(struInput, data)
            struGCN, mask2 = to_dense_batch(struGCN, data.batch)
            struGAT, mask2 = to_dense_batch(struGAT, data.batch)
            graph = torch.cat([struGCN, struGAT], dim=2)


        seq = data.x.float()
        seq,mask3=to_dense_batch(seq,data.batch)
        seq = seq.permute(0, 2, 1)
        seq1 = self.gatedConv1(seq)
        seq2 = self.gatedConv2(seq)
        seq3 = self.gatedConv3(seq)
        
        seq = torch.cat([seq1, seq2, seq3], dim=1)  

        if not self.fcatNet:
            seq = seq.permute(0, 2, 1)
            seq = self.fusion(seq)  
            seq = seq.permute(0, 2, 1)
        if self.fcatNet:
            seq = self.fecam(seq) 
        seq=seq.permute(0, 2, 1)
        if self.cross_att:
            mask2 = (~mask2).to(dtype=seq.dtype)
            mask3 = (~mask3).to(dtype=seq.dtype)
            struCross = self.smi_attention_poc(graph, seq, attn_bias=mask2)
            seqCross = self.smi_attention_poc(seq, graph, attn_bias=mask3)
        else:
            struCross = graph
            seqCross = seq

        struCross = struCross.permute(0, 2, 1)
        seqCross = seqCross.permute(0, 2, 1)
        
        struCross = self.global_maxpool1d(struCross).squeeze(2)  
        seqCross = self.global_maxpool1d(seqCross).squeeze(2)   
    
        w1 = F.sigmoid(self.w1)
        output = torch.add((1 - w1) * struCross, w1 * seqCross)  

        output = self.readout(output)
        return output 

