"""Graph encoder and prototype-conditioned classifier."""
import torch
import torch.nn as nn

class DGG(torch.nn.Module):

    def __init__(self, GNN, Transformer, transformer_d, max_hop=1, max_dist=2, max_nodes=41, embedding_dim=128):
        super(DGG, self).__init__()
        self.GNN = GNN
        self.Transformer = Transformer
        self.Transformer_d = transformer_d
        self.act = nn.PReLU()
        self.Ehe = nn.Parameter(torch.randn(max_hop + 1, embedding_dim))
        self.Ede = nn.Parameter(torch.randn(max_dist + 1, embedding_dim))
        self.Eft = nn.Parameter(torch.randn(max_nodes, embedding_dim))
        self.Ece = nn.Parameter(torch.randn(4, embedding_dim))

    def forward(self, H_v, H_e, H_e_pad, adj_e, adj_v, T, mask, ra, normal_prompt_raw, abnormal_prompt_raw, normal_mean, normal_cov, abnormal_cov, abnormal_mean):
        normal_prompt = normal_prompt_raw
        abnormal_prompt = abnormal_prompt_raw
        output = self.GNN(H_v, H_e, adj_e, adj_v, T)
        device = next(self.parameters()).device
        pad_output = torch.zeros((H_e_pad.shape[0], H_e_pad.shape[1], H_e_pad.shape[2])).to(device)
        for i in range(H_e_pad.shape[0]):
            length = output[i].shape[0]
            pad_output[i, :length, :] = output[i]
        (logits, output, normal_mean, normal_cov, abnormal_cov, abnormal_mean) = self.Transformer(H_e_pad, pad_output, mask, normal_prompt, abnormal_prompt, normal_mean, normal_cov, abnormal_cov, abnormal_mean)
        return (logits, output, normal_prompt, abnormal_prompt, normal_mean, normal_cov, abnormal_cov, abnormal_mean)
