"""Shared backbone construction, checkpoint loading and online OTTA execution."""
import os
import random
from types import SimpleNamespace

import numpy as np
import torch

from model.CensNet import CensNet
from model.DGG import DGG
from model.Transformer import PrototypeAttention, TransformerBinaryClassifier
from otta_modules import (CrossWindowContextRecovery, EntropyPrototypeAnchor,
                          TemporalStructuralReliableMemoryUpdate,
                          compute_ego_graph_neighbor_probs)

MODEL_DEFAULTS = dict(input_dim=128, hidden_dim=258, n_heads=4, drop_out=0.4,
                      n_layer=6, momentum=0.9)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def resolve_device(name):
    device = torch.device(name)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; select a CUDA-enabled runtime or use --device cpu.')
        torch.cuda.set_device(device)
    return device


def build_model(config, device):
    return DGG(CensNet(config.input_dim, config.drop_out),
               TransformerBinaryClassifier(config, device, hidden_size=config.hidden_dim),
               PrototypeAttention()).to(device)


def empty_statistics(dim=128):
    return dict(normal_mean=torch.zeros(dim), normal_cov=torch.eye(dim),
                abnormal_mean=torch.zeros(dim), abnormal_cov=torch.eye(dim))


def load_model(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = SimpleNamespace(**(MODEL_DEFAULTS | checkpoint.get('model_config', {})))
    model = build_model(config, device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    statistics = empty_statistics(config.input_dim)
    statistics.update({k: checkpoint[k] for k in statistics if k in checkpoint})
    return model, config, checkpoint, statistics


def forward_batch(model, batch, device, prompts, statistics):
    return model(
        [x.to(device) for x in batch['input_nodes_feature']],
        [x.to(device) for x in batch['input_edges_feature']],
        batch['input_edges_pad'].to(device),
        [x.to(device) for x in batch['eadjs']],
        [x.to(device) for x in batch['adjs']],
        [x.to(device) for x in batch['Tmats']], batch['mask_edge'].to(device),
        batch['ra'], prompts[0].to(device), prompts[1].to(device),
        statistics['normal_mean'].to(device), statistics['normal_cov'].to(device),
        statistics['abnormal_cov'].to(device), statistics['abnormal_mean'].to(device))


class OnlineOTTA:
    """Score chunk t before adapting its state for chunk t+1; labels are unused."""
    def __init__(self, model, checkpoint, statistics, device, *, buffer_size=100,
                 delta=2.0, lambda_fusion=0.3, top_k=5, tau=0.5, top_n_ratio=0.1,
                 otta_score_blend=0.1, min_effective_blend=0.005,
                 rank_penalty_strong=0.2, min_structural_neighbors=2,
                 anomaly_min_support=0.4, anomaly_min_reliability=0.75,
                 abnormal_momentum=0.985, min_abnormal_update_count=3):
        self.model = model.eval().requires_grad_(False)
        self.device, self.statistics = device, statistics
        self.context = CrossWindowContextRecovery(buffer_size, delta, lambda_fusion, top_k)
        self.reliability = TemporalStructuralReliableMemoryUpdate(
            tau, top_n_ratio, otta_score_blend, min_effective_blend,
            rank_penalty_strong, min_structural_neighbors, anomaly_min_support,
            anomaly_min_reliability, abnormal_momentum, min_abnormal_update_count)
        self.anchor = EntropyPrototypeAnchor(top_ratio=top_n_ratio)
        bank = checkpoint.get('prototype_buffer', [])
        if not bank:
            raise ValueError('Checkpoint has no prototype_buffer; first run train.py.')
        index = int(checkpoint.get('best_prototype_idx', 0))
        if not 0 <= index < len(bank):
            raise ValueError('Invalid best_prototype_idx in checkpoint')
        self.prompts = tuple(x.detach() for x in bank[index][:2])
        for obj in (self.reliability, self.anchor):
            obj.prev_normal_prototype = self.prompts[0].detach().cpu().clone()
            obj.prev_abnormal_prototype = self.prompts[1].detach().cpu().clone()

    @torch.no_grad()
    def step(self, batch):
        with torch.amp.autocast('cuda', enabled=self.device.type == 'cuda'):
            raw, embeddings, *_ = forward_batch(self.model, batch, self.device,
                                                self.prompts, self.statistics)
        raw = raw.reshape(-1).float()
        probabilities = raw.clamp(1e-6, 1-1e-6) if torch.all((raw >= 0) & (raw <= 1)) else raw.sigmoid().clamp(1e-6, 1-1e-6)
        if embeddings.ndim == 3:
            embeddings = embeddings.reshape(-1, embeddings.shape[-1])
        base, _ = self.anchor.score_representations(embeddings, probabilities, None, None, self.device)
        enhanced = self.context.retrieve_and_fuse(embeddings, self.device)
        scores, _ = self.reliability.score_representations(enhanced, base, self.device)
        # Every update happens after current scores have been produced.
        self.anchor.adapt_chunk(embeddings.detach(), base.detach(), (base.detach() > 0.5).float(), self.device)
        neighbors, novelty = compute_ego_graph_neighbor_probs(batch, scores.detach(), self.device)
        reliability, pseudo = self.reliability.compute_reliability_scores(
            enhanced.detach(), scores.detach(), neighbors, self.device, endpoint_novelty=novelty)
        mask = self.reliability.select_reliable_samples(reliability, pseudo)
        prototypes = self.reliability.build_target_prototypes(enhanced.detach(), pseudo, reliability, mask)
        self.reliability.update_prototypes(*prototypes)
        self.context.update_buffer(enhanced.detach(), pseudo, reliability)
        return scores.detach().cpu().numpy()
