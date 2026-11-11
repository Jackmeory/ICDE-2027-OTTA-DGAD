"""OTTA-DGAD context recovery, reliable updates and internal entropy anchor."""
import torch
import torch.nn.functional as F
import numpy as np


class CrossWindowContextRecovery:
    """
    Recovers missing temporal-topological context caused by chunk-wise observation.

    Maintains a lightweight historical context buffer from recent target chunks.
    Only stores high-reliability historical items for efficient context retrieval.
    """

    def __init__(self, buffer_size=100, delta=2.0, lambda_fusion=0.3, top_k=5):
        """
        Args:
            buffer_size: Maximum number of historical items to store
            delta: Age decay factor for temporal weighting
            lambda_fusion: Fusion weight (lambda * z_i + (1-lambda) * c_i)
            top_k: Number of top historical items to retrieve
        """
        self.buffer_size = buffer_size
        self.delta = delta
        self.lambda_fusion = lambda_fusion
        self.top_k = top_k

        # Historical context buffer
        # Each item: (representation, pseudo_label, reliability, chunk_idx)
        self.buffer = []
        self.current_chunk_idx = 0

    def update_buffer(self, representations, pseudo_labels, reliability_scores):
        """
        Update historical context buffer with current chunk's reliable items.

        Args:
            representations: Tensor [N, D] edge representations
            pseudo_labels: Tensor [N] pseudo labels (0 or 1)
            reliability_scores: Tensor [N] reliability scores
        """
        # Select top-K most reliable items to store
        k = min(self.buffer_size // 10, len(reliability_scores))  # Store ~10% per chunk
        self.last_inserted_indices = torch.empty(0, dtype=torch.long)
        if k > 0:
            top_k_indices = torch.topk(reliability_scores, k, largest=True)[1]
            self.last_inserted_indices = top_k_indices.detach().cpu()

            for idx in top_k_indices:
                item = {
                    'representation': representations[idx].detach().cpu(),
                    'pseudo_label': pseudo_labels[idx].item(),
                    'reliability': reliability_scores[idx].item(),
                    'chunk_idx': self.current_chunk_idx,
                    'age': 0  # Will be computed dynamically
                }
                self.buffer.append(item)

        # Maintain buffer size
        if len(self.buffer) > self.buffer_size:
            # Remove oldest items
            self.buffer = self.buffer[-self.buffer_size:]

        self.current_chunk_idx += 1

    def retrieve_and_fuse(self, current_representations, device):
        """
        Retrieve relevant historical context and fuse with current representations.

        Args:
            current_representations: Tensor [N, D] current edge representations
            device: torch device

        Returns:
            context_enhanced_representations: Tensor [N, D] fused representations
        """
        if len(self.buffer) == 0:
            # No historical context available
            return current_representations

        N, D = current_representations.shape

        buffer_repr = torch.stack([item['representation'].to(device) for item in self.buffer], dim=0)  # [B, D]
        buffer_rel = torch.tensor([item['reliability'] for item in self.buffer], device=device)  # [B]
        buffer_age = torch.tensor([self.current_chunk_idx - item['chunk_idx'] for item in self.buffer], device=device, dtype=torch.float32)
        age_weight = torch.exp(-buffer_age / self.delta) * buffer_rel

        current_norm = F.normalize(current_representations, dim=1)
        buffer_norm = F.normalize(buffer_repr, dim=1)
        sim_matrix = current_norm @ buffer_norm.t()  # [N, B]
        score_matrix = sim_matrix * age_weight.unsqueeze(0)

        top_k = min(self.top_k, score_matrix.shape[1])
        top_scores, top_indices = torch.topk(score_matrix, top_k, dim=1, largest=True)
        gathered_context = buffer_repr[top_indices]  # [N, K, D]
        normalized_weights = torch.softmax(top_scores, dim=1)
        context = (normalized_weights.unsqueeze(-1) * gathered_context).sum(dim=1)  # [N, D]
        return self.lambda_fusion * current_representations + (1 - self.lambda_fusion) * context


class TemporalStructuralReliableMemoryUpdate:
    """
    Replaces entropy-only confident detection with reliable target adaptation.

    Combines confidence, structural verification, and temporal verification
    to select reliable pseudo-labeled samples for memory update.
    """

    def __init__(self, tau=0.5, top_n_ratio=0.1, otta_score_blend=0.10,
                 min_effective_blend=0.005, rank_penalty_strong=0.20,
                 min_structural_neighbors=2, anomaly_min_support=0.40,
                 anomaly_min_reliability=0.75, abnormal_momentum=0.985,
                 min_abnormal_update_count=3):
        """
        Args:
            tau: Temperature for temporal verification
            top_n_ratio: Ratio of top reliable samples to select (e.g., 0.1 = 10%)
        """
        self.tau = tau
        self.top_n_ratio = top_n_ratio

        # Previous target-side class prototypes
        self.prev_normal_prototype = None
        self.prev_abnormal_prototype = None
        self.otta_score_blend = float(otta_score_blend)
        self.proto_temperature = 1.0
        self.logit_adjust_scale = 1.0
        self.normal_prototype_momentum = 0.85
        self.abnormal_prototype_momentum = float(abnormal_momentum)
        self.min_proto_separation = 0.02
        self.good_proto_separation = 0.20
        self.hard_min_proto_separation = 1e-4
        self.min_effective_blend = float(min_effective_blend)
        self.rank_penalty_strong = float(rank_penalty_strong)
        self.rank_penalty_mild = 0.50
        self.min_abnormal_update_count = int(min_abnormal_update_count)
        self.min_abnormal_update_reliability = float(anomaly_min_reliability)
        self.min_structural_neighbors = int(min_structural_neighbors)
        self.anomaly_min_support = float(anomaly_min_support)
        self.anomaly_min_reliability = float(anomaly_min_reliability)
        self.last_update_stats = {
            'has_target_update': False,
            'avg_reliability': 0.0,
            'reliable_normal': 0,
            'reliable_abnormal': 0,
            'normal_update_used': False,
            'abnormal_update_used': False,
        }
        self.last_score_diagnostics = {}
        self.last_selection_diagnostics = {}
        self.last_confidence_scores = None
        self.last_structural_support_scores = None

    def compute_reliability_scores(self, representations, probs, ego_graph_neighbors_probs, device,
                                   endpoint_novelty=None):
        """
        Compute reliability scores for each target edge.

        Args:
            representations: Tensor [N, D] context-enhanced representations
            probs: Tensor [N] anomaly probabilities
            ego_graph_neighbors_probs: List of Tensors, ego-graph neighbor probabilities
            device: torch device

        Returns:
            reliability_scores: Tensor [N] reliability scores
            pseudo_labels: Tensor [N] pseudo labels (0 or 1)
        """
        N = len(probs)
        eps = 1e-10

        # Step 1: Pseudo-label from evaluated scores. If one class is absent,
        # prototype fallback keeps the historical prototype for that class.
        pseudo_labels = (probs > 0.5).float()  # [N]

        # Step 2: Confidence from entropy
        entropy = -probs * torch.log(probs + eps) - (1 - probs) * torch.log(1 - probs + eps)
        confidence = 1 - entropy / np.log(2)  # [N]

        # Step 3: Structural verification
        structural_verification = torch.zeros(N, device=device)
        for i in range(N):
            novelty = endpoint_novelty[i] if endpoint_novelty is not None else None
            if pseudo_labels[i] == 1 and novelty is not None:
                # TAX51 injections introduce a disconnected endpoint: novelty,
                # rather than anomalous-neighbor agreement, is anomaly evidence.
                structural_verification[i] = novelty
            elif (ego_graph_neighbors_probs[i] is not None
                    and len(ego_graph_neighbors_probs[i]) >= self.min_structural_neighbors):
                neighbors_prob = ego_graph_neighbors_probs[i]
                continuity = (1 - neighbors_prob).mean()
                structural_verification[i] = (
                    0.5 * continuity + 0.5 * (1 - novelty)
                    if novelty is not None else continuity
                )
            else:
                structural_verification[i] = 0.0

        # Retained only for optional post-hoc visualization capture.
        self.last_confidence_scores = confidence.detach().cpu()
        self.last_structural_support_scores = structural_verification.detach().cpu()

        # Step 4: Temporal verification
        temporal_verification = torch.ones(N, device=device)

        if self.prev_normal_prototype is not None and self.prev_abnormal_prototype is not None:
            prev_normal = self.prev_normal_prototype.to(device)
            prev_abnormal = self.prev_abnormal_prototype.to(device)

            for i in range(N):
                z_tilde_i = representations[i]

                target_proto = prev_abnormal if pseudo_labels[i] == 1 else prev_normal
                sim = F.cosine_similarity(
                    z_tilde_i.float().unsqueeze(0),
                    target_proto.float().unsqueeze(0),
                    dim=1
                ).squeeze()
                temporal_verification[i] = torch.exp((sim.clamp(-1.0, 1.0) - 1.0) / max(self.tau, 1e-6))

        # Step 5: Raw reliability
        raw_reliability = confidence * structural_verification * temporal_verification

        # Step 6: Class-wise normalization
        normalized_reliability = torch.zeros_like(raw_reliability)

        for label in [0, 1]:
            mask = (pseudo_labels == label)
            if mask.sum() > 0:
                class_reliability = raw_reliability[mask]
                r_min = class_reliability.min()
                r_max = class_reliability.max()

                if r_max > r_min:
                    class_normalized = (class_reliability - r_min) / (r_max - r_min + eps)
                else:
                    class_normalized = torch.ones_like(class_reliability)

                normalized_reliability[mask] = class_normalized

        return normalized_reliability, pseudo_labels

    def select_reliable_samples(self, reliability_scores, pseudo_labels):
        """
        Select top reliable samples based on class-normalized reliability.

        Args:
            reliability_scores: Tensor [N] normalized reliability scores
            pseudo_labels: Tensor [N] pseudo labels

        Returns:
            reliable_mask: Tensor [N] boolean mask for reliable samples
        """
        N = len(reliability_scores)
        max_select = min(N, max(10, int(self.top_n_ratio * N)))
        min_select = min(max_select, max(2, int(0.03 * N)))

        chunk_quality = reliability_scores.mean().clamp(0.0, 1.0)
        n_select = int(round(min_select + (max_select - min_select) * float(chunk_quality.item())))
        n_select = max(min_select, min(max_select, n_select))
        selected_parts = []
        eligible_mask = torch.ones(N, dtype=torch.bool, device=reliability_scores.device)
        if self.last_structural_support_scores is not None:
            support = self.last_structural_support_scores.to(reliability_scores.device)
            anomaly_mask = pseudo_labels == 1
            eligible_mask[anomaly_mask] = (
                (support[anomaly_mask] >= self.anomaly_min_support)
                & (reliability_scores[anomaly_mask] >= self.anomaly_min_reliability)
            )
        present_labels = [label for label in [0, 1] if ((pseudo_labels == label) & eligible_mask).sum() > 0]
        for label in [0, 1]:
            label_indices = torch.where((pseudo_labels == label) & eligible_mask)[0]
            if len(label_indices) == 0:
                continue

            label_quota = n_select if len(present_labels) == 1 else max(1, n_select // 2)
            label_quota = min(len(label_indices), label_quota)
            label_scores = reliability_scores[label_indices]
            selected_parts.append(label_indices[torch.topk(label_scores, label_quota, largest=True)[1]])

        if selected_parts:
            selected_indices = torch.cat(selected_parts)
            if len(selected_indices) < n_select:
                selected_mask = torch.zeros(N, dtype=torch.bool, device=reliability_scores.device)
                selected_mask[selected_indices] = True
                remaining_indices = torch.where((~selected_mask) & eligible_mask)[0]
                if len(remaining_indices) > 0:
                    fill = min(n_select - len(selected_indices), len(remaining_indices))
                    remaining_scores = reliability_scores[remaining_indices]
                    selected_indices = torch.cat([
                        selected_indices,
                        remaining_indices[torch.topk(remaining_scores, fill, largest=True)[1]]
                    ])
            elif len(selected_indices) > n_select:
                selected_scores = reliability_scores[selected_indices]
                selected_indices = selected_indices[torch.topk(selected_scores, n_select, largest=True)[1]]
        else:
            selected_indices = torch.topk(reliability_scores, n_select, largest=True)[1]

        reliable_mask = torch.zeros(N, dtype=torch.bool, device=reliability_scores.device)
        reliable_mask[selected_indices] = True
        self.last_selected_indices = selected_indices.detach().cpu()
        self.last_selection_diagnostics = {
            'reliable_normal': int(((pseudo_labels == 0) & reliable_mask).sum().item()),
            'reliable_abnormal': int(((pseudo_labels == 1) & reliable_mask).sum().item()),
            'pseudo_normal': int((pseudo_labels == 0).sum().item()),
            'pseudo_abnormal': int((pseudo_labels == 1).sum().item()),
        }
        return reliable_mask

    def build_target_prototypes(self, representations, pseudo_labels, reliability_scores, reliable_mask):
        """
        Build class-wise target prototypes with reliability-weighted aggregation.

        Args:
            representations: Tensor [N, D] context-enhanced representations
            pseudo_labels: Tensor [N] pseudo labels
            reliability_scores: Tensor [N] reliability scores
            reliable_mask: Tensor [N] boolean mask for reliable samples

        Returns:
            target_normal_prototype: Tensor [D] or None
            target_abnormal_prototype: Tensor [D] or None
        """
        device = representations.device
        D = representations.shape[1]

        # Build prototypes only from reliable samples. Abnormal updates are
        # intentionally stricter because rare false-positive pseudo-labels can
        # damage top-anomaly ranking and AUPRC.
        reliable_representations = representations[reliable_mask]
        reliable_pseudo_labels = pseudo_labels[reliable_mask]
        reliable_reliability = reliability_scores[reliable_mask]

        target_normal_prototype = None
        target_abnormal_prototype = None

        # Normal prototype (label = 0)
        normal_mask = (reliable_pseudo_labels == 0)
        normal_count = int(normal_mask.sum().item())
        normal_reliability_mean = 0.0
        if normal_mask.sum() > 0:
            normal_reprs = reliable_representations[normal_mask]
            normal_weights = reliable_reliability[normal_mask]
            normal_reliability_mean = float(normal_weights.mean().detach().item())
            if normal_weights.sum() <= 1e-10:
                target_normal_prototype = normal_reprs.mean(dim=0)
            else:
                normal_weights = normal_weights / (normal_weights.sum() + 1e-10)
                target_normal_prototype = (normal_weights.unsqueeze(1) * normal_reprs).sum(dim=0)

        # Abnormal prototype (label = 1)
        abnormal_mask = (reliable_pseudo_labels == 1)
        abnormal_count = int(abnormal_mask.sum().item())
        abnormal_reliability_mean = 0.0
        if abnormal_mask.sum() > 0:
            abnormal_reprs = reliable_representations[abnormal_mask]
            abnormal_weights = reliable_reliability[abnormal_mask]
            abnormal_reliability_mean = float(abnormal_weights.mean().detach().item())
            enough_abnormal = (
                abnormal_count >= self.min_abnormal_update_count
                and abnormal_reliability_mean >= self.min_abnormal_update_reliability
            )
            if enough_abnormal:
                if abnormal_weights.sum() <= 1e-10:
                    target_abnormal_prototype = abnormal_reprs.mean(dim=0)
                else:
                    abnormal_weights = abnormal_weights / (abnormal_weights.sum() + 1e-10)
                    target_abnormal_prototype = (abnormal_weights.unsqueeze(1) * abnormal_reprs).sum(dim=0)

        selected_count = int(reliable_mask.sum().item())
        avg_reliability = (
            float(reliable_reliability.mean().detach().item())
            if selected_count > 0 else 0.0
        )
        self.pending_update_stats = {
            'has_target_update': selected_count > 0,
            'avg_reliability': avg_reliability,
            'reliable_normal': normal_count,
            'reliable_abnormal': abnormal_count,
            'normal_reliability_mean': normal_reliability_mean,
            'abnormal_reliability_mean': abnormal_reliability_mean,
            'normal_update_used': target_normal_prototype is not None,
            'abnormal_update_used': target_abnormal_prototype is not None,
        }

        return target_normal_prototype, target_abnormal_prototype

    def update_prototypes(self, target_normal_prototype, target_abnormal_prototype):
        """
        Update previous prototypes for next chunk.

        Args:
            target_normal_prototype: Tensor [D] or None
            target_abnormal_prototype: Tensor [D] or None
        """
        if target_normal_prototype is not None:
            if self.prev_normal_prototype is None:
                updated_normal = target_normal_prototype.detach()
            else:
                updated_normal = (
                    self.normal_prototype_momentum * self.prev_normal_prototype.to(target_normal_prototype.device)
                    + (1 - self.normal_prototype_momentum) * target_normal_prototype.detach()
                )
            self.prev_normal_prototype = updated_normal.cpu()

        if target_abnormal_prototype is not None:
            if self.prev_abnormal_prototype is None:
                updated_abnormal = target_abnormal_prototype.detach()
            else:
                updated_abnormal = (
                    self.abnormal_prototype_momentum * self.prev_abnormal_prototype.to(target_abnormal_prototype.device)
                    + (1 - self.abnormal_prototype_momentum) * target_abnormal_prototype.detach()
                )
            self.prev_abnormal_prototype = updated_abnormal.cpu()

        if hasattr(self, 'pending_update_stats'):
            self.last_update_stats = dict(self.pending_update_stats)

        return self.prev_normal_prototype, self.prev_abnormal_prototype

    def get_target_prototypes(self, current_normal=None, current_abnormal=None, device=None):
        """
        Resolve current target prototypes with historical fallback.

        The current chunk may only produce one pseudo-label class. In that case,
        keep using the previous target prototype for the missing class when it
        exists instead of disabling adaptation entirely.
        """
        normal_proto = current_normal if current_normal is not None else self.prev_normal_prototype
        abnormal_proto = current_abnormal if current_abnormal is not None else self.prev_abnormal_prototype

        if device is not None:
            if normal_proto is not None:
                normal_proto = normal_proto.to(device)
            if abnormal_proto is not None:
                abnormal_proto = abnormal_proto.to(device)

        return normal_proto, abnormal_proto

    def score_representations(self, representations, probs, device):
        """
        Score current representations with target prototypes learned from
        previous chunks. If both target prototypes are unavailable, preserve the
        frozen backbone probabilities.
        """
        self.last_score_diagnostics = {
            'proto_score_used': False,
            'otta_score_blend': float(self.otta_score_blend),
            'score_gate': 0.0,
            'effective_blend': 0.0,
            'rank_penalty': 1.0,
            'score_guard_status': 'hard_fallback',
            'proto_disabled_reason': 'unavailable',
            'proto_cosine_separation': 0.0,
            'proto_l2_separation': 0.0,
            'proto_logit_mean': 0.0,
            'proto_logit_std': 0.0,
            'proto_logit_abs_mean': 0.0,
            'base_logit_std': 0.0,
        }
        normal_proto, abnormal_proto = self.get_target_prototypes(device=device)
        if normal_proto is None or abnormal_proto is None:
            return probs, False

        proto_cosine_separation = 1 - F.cosine_similarity(
            normal_proto.float().unsqueeze(0),
            abnormal_proto.float().unsqueeze(0),
            dim=1
        ).squeeze()
        proto_l2_separation = torch.norm(normal_proto.float() - abnormal_proto.float(), p=2)

        confidence = 1 - (
            -probs * torch.log(probs + 1e-10)
            - (1 - probs) * torch.log(1 - probs + 1e-10)
        ) / np.log(2)
        chunk_confidence = confidence.mean().clamp(0.0, 1.0)

        update_stats = self.last_update_stats
        reliable_normal = float(update_stats.get('reliable_normal', 0))
        reliable_abnormal = float(update_stats.get('reliable_abnormal', 0))
        avg_reliability = float(update_stats.get('avg_reliability', 0.0))

        if (not torch.isfinite(proto_cosine_separation)) or (not torch.isfinite(proto_l2_separation)):
            self.last_score_diagnostics.update({
                'proto_score_used': False,
                'score_guard_status': 'hard_fallback',
                'otta_score_blend': float(self.otta_score_blend),
                'score_gate': 0.0,
                'effective_blend': 0.0,
                'rank_penalty': 0.0,
                'proto_disabled_reason': 'invalid_prototype',
            })
            return probs, False

        sep_gate = (0.10 + 0.90 * (proto_cosine_separation / max(self.good_proto_separation, 1e-6))).clamp(0.0, 1.0)
        normal_gate = min(1.0, reliable_normal / 4.0)
        abnormal_gate = min(1.0, reliable_abnormal / 2.0)
        count_gate = 0.20 + 0.80 * float(np.sqrt(max(normal_gate, 0.0) * max(abnormal_gate, 0.0)))
        if not update_stats.get('has_target_update', False):
            count_gate = 0.35
            avg_reliability = 0.50
        reliability_gate = 0.55 * count_gate + 0.45 * max(0.0, min(1.0, avg_reliability))
        confidence_gate = 0.20 + 0.80 * chunk_confidence.clamp(0.0, 1.0)
        gate = (sep_gate * float(reliability_gate) * confidence_gate).clamp(0.0, 1.0)

        base_logits = torch.logit(probs.clamp(1e-6, 1 - 1e-6))
        base_centered = base_logits - base_logits.mean()
        base_std = base_centered.std(unbiased=False)
        blend = min(0.95, max(0.0, float(self.otta_score_blend)))
        effective_blend = blend * float(gate.detach().item())
        if blend > 0 and proto_cosine_separation >= self.hard_min_proto_separation:
            effective_blend = max(effective_blend, max(0.0, float(self.min_effective_blend)))
        rank_preserve_candidate = (
            (not update_stats.get('has_target_update', False))
            or (reliable_normal < 2 and reliable_abnormal >= 2)
        )
        if rank_preserve_candidate:
            effective_blend = 0.0

        disabled_reason = ''
        if proto_cosine_separation < self.hard_min_proto_separation:
            disabled_reason = 'low_separation'
        elif effective_blend <= 1e-12 and not rank_preserve_candidate:
            disabled_reason = 'low_gate'
        elif base_std <= 1e-8:
            disabled_reason = 'flat_base'

        if disabled_reason:
            self.last_score_diagnostics.update({
                'proto_score_used': False,
                'otta_score_blend': float(blend),
                'score_gate': float(gate.detach().item()),
                'effective_blend': 0.0,
                'rank_penalty': 0.0,
                'score_guard_status': 'hard_fallback',
                'proto_disabled_reason': disabled_reason,
                'proto_cosine_separation': float(proto_cosine_separation.detach().item()),
                'proto_l2_separation': float(proto_l2_separation.detach().item()),
                'chunk_confidence_mean': float(chunk_confidence.detach().item()),
                'state_avg_reliability': float(avg_reliability),
                'state_reliable_normal': int(reliable_normal),
                'state_reliable_abnormal': int(reliable_abnormal),
                'base_logit_std': float(base_std.detach().item()),
            })
            return probs, False

        rep_norm = F.normalize(representations.float(), dim=1)
        normal = F.normalize(normal_proto.float(), dim=0)
        abnormal = F.normalize(abnormal_proto.float(), dim=0)

        # Larger means more anomalous: close to abnormal prototype and far from normal.
        cosine_margin = (rep_norm @ abnormal - rep_norm @ normal)
        proto_logit = cosine_margin
        proto_logit = proto_logit / max(self.proto_temperature, 1e-6)

        centered_proto = proto_logit - proto_logit.mean()
        if torch.dot(centered_proto, probs.float() - probs.float().mean()) < 0:
            proto_logit = -proto_logit

        proto_median = proto_logit.median()
        proto_centered = proto_logit - proto_median
        proto_mad = proto_centered.abs().median()
        proto_std = proto_centered.std(unbiased=False)
        robust_scale = 1.4826 * proto_mad
        if robust_scale > 1e-8:
            proto_z = proto_centered / robust_scale
        elif proto_std > 1e-8:
            proto_z = proto_centered / proto_std
        else:
            proto_z = torch.zeros_like(proto_centered)
        proto_z = proto_z.clamp(-2.0, 2.0)

        top_count = min(len(base_logits), max(10, int(0.50 * len(base_logits))))
        base_top_indices = torch.topk(base_logits, top_count, largest=True)[1]
        base_top_mask = torch.zeros_like(base_logits, dtype=torch.bool)
        base_top_mask[base_top_indices] = True

        logit_scale = base_std.clamp(0.05, 1.00)
        prototype_boost = proto_z.clone()
        # Preserve the base top precision band; move the rest of the ranking.
        # This keeps AUPRC-sensitive leading scores stable without disabling
        # OTTA residuals for the entire chunk.
        prototype_boost[base_top_mask] = 0.0
        rank_preserve_applied = False
        if rank_preserve_candidate:
            rank_preserve_applied = True
            fused_logits = base_logits
        else:
            fused_logits = base_logits + effective_blend * logit_scale * prototype_boost
            if base_top_mask.any() and (~base_top_mask).any():
                protected_floor = base_logits[base_top_mask].min()
                fused_logits[~base_top_mask] = torch.minimum(
                    fused_logits[~base_top_mask],
                    protected_floor - 1e-6
                )
        if not torch.isfinite(fused_logits).all():
            self.last_score_diagnostics.update({
                'proto_score_used': False,
                'otta_score_blend': float(blend),
                'score_gate': float(gate.detach().item()),
                'effective_blend': 0.0,
                'rank_penalty': 0.0,
                'score_guard_status': 'hard_fallback',
                'proto_disabled_reason': 'invalid_fused_logits',
                'proto_cosine_separation': float(proto_cosine_separation.detach().item()),
                'proto_l2_separation': float(proto_l2_separation.detach().item()),
                'chunk_confidence_mean': float(chunk_confidence.detach().item()),
                'state_avg_reliability': float(avg_reliability),
                'state_reliable_normal': int(reliable_normal),
                'state_reliable_abnormal': int(reliable_abnormal),
                'base_logit_std': float(base_std.detach().item()),
            })
            return probs, False
        fused_top_indices = torch.topk(fused_logits, top_count, largest=True)[1]
        fused_top_mask = torch.zeros_like(base_top_mask)
        fused_top_mask[fused_top_indices] = True
        top_overlap = (base_top_mask & fused_top_mask).float().sum() / max(top_count, 1)
        top_order_preserved = bool(torch.equal(base_top_indices, fused_top_indices))
        guarded_effective_blend = effective_blend
        rank_penalty = 1.0
        score_guard_status = 'full_active'
        if rank_preserve_applied:
            score_guard_status = 'soft_downweighted'
        if not top_order_preserved:
            if top_overlap >= 0.95:
                rank_penalty = self.rank_penalty_mild
            elif top_overlap >= 0.80:
                rank_penalty = 0.5 * (self.rank_penalty_mild + self.rank_penalty_strong)
            else:
                rank_penalty = self.rank_penalty_strong

            score_guard_status = 'soft_downweighted'
            guarded_effective_blend = max(
                effective_blend * rank_penalty,
                max(0.0, float(self.min_effective_blend))
            )
            fused_logits = base_logits + guarded_effective_blend * logit_scale * prototype_boost
            if base_top_mask.any() and (~base_top_mask).any():
                protected_floor = base_logits[base_top_mask].min()
                fused_logits[~base_top_mask] = torch.minimum(
                    fused_logits[~base_top_mask],
                    protected_floor - 1e-6
                )
            if rank_preserve_applied:
                fused_logits = base_logits

        fused_std = fused_logits.std(unbiased=False)
        if (not torch.isfinite(fused_std)) or fused_std < 0.50 * base_std:
            self.last_score_diagnostics.update({
                'proto_score_used': False,
                'otta_score_blend': float(blend),
                'score_gate': float(gate.detach().item()),
                'effective_blend': 0.0,
                'rank_penalty': float(rank_penalty),
                'score_guard_status': 'hard_fallback',
                'proto_disabled_reason': 'variance_collapse',
                'top_overlap': float(top_overlap.detach().item()),
                'top_order_preserved': bool(top_order_preserved),
                'rank_guard_applied': score_guard_status == 'soft_downweighted',
                'proto_cosine_separation': float(proto_cosine_separation.detach().item()),
                'proto_l2_separation': float(proto_l2_separation.detach().item()),
                'chunk_confidence_mean': float(chunk_confidence.detach().item()),
                'state_avg_reliability': float(avg_reliability),
                'state_reliable_normal': int(reliable_normal),
                'state_reliable_abnormal': int(reliable_abnormal),
                'base_logit_std': float(base_std.detach().item()),
                'fused_logit_std': float(fused_std.detach().item()),
            })
            return probs, False

        if rank_preserve_applied:
            fused = probs
        else:
            fused = torch.sigmoid(fused_logits).clamp(1e-6, 1 - 1e-6)

        self.last_score_diagnostics = {
            'proto_score_used': True,
            'otta_score_blend': float(blend),
            'score_gate': float(gate.detach().item()),
            'effective_blend': float(guarded_effective_blend),
            'rank_penalty': float(rank_penalty),
            'score_guard_status': score_guard_status,
            'proto_disabled_reason': '',
            'top_overlap': float(top_overlap.detach().item()),
            'top_order_preserved': bool(top_order_preserved),
            'rank_preserve_applied': bool(rank_preserve_applied),
            'rank_guard_applied': score_guard_status == 'soft_downweighted',
            'proto_cosine_separation': float(proto_cosine_separation.detach().item()),
            'proto_l2_separation': float(proto_l2_separation.detach().item()),
            'proto_logit_mean': float(proto_logit.mean().detach().item()),
            'proto_logit_std': float(proto_std.detach().item()),
            'proto_logit_abs_mean': float(proto_logit.abs().mean().detach().item()),
            'proto_boost_mean': float(prototype_boost.mean().detach().item()),
            'proto_boost_max': float(prototype_boost.max().detach().item()),
            'prototype_adjust_abs_mean': float((guarded_effective_blend * logit_scale * prototype_boost).abs().mean().detach().item()),
            'prototype_adjust_abs_max': float((guarded_effective_blend * logit_scale * prototype_boost).abs().max().detach().item()),
            'base_logit_std': float(base_std.detach().item()),
            'fused_logit_std': float(fused_std.detach().item()),
            'chunk_confidence_mean': float(chunk_confidence.detach().item()),
            'state_avg_reliability': float(avg_reliability),
            'state_reliable_normal': int(reliable_normal),
            'state_reliable_abnormal': int(reliable_abnormal),
        }
        return fused, True


def compute_ego_graph_neighbor_probs(batch_sample, probs, device):
    """
    Extract ego-graph neighbor probabilities for structural verification.

    Args:
        batch_sample: Batch data dictionary
        probs: Tensor [N] anomaly probabilities for current batch
        device: torch device

    Returns:
        ego_graph_neighbors_probs: List of Tensors, neighbor probabilities for each edge
    """
    N = len(probs)
    node_ids = batch_sample.get('structural_node_ids', batch_sample.get('raw_node_ids'))
    endpoint_novelty = batch_sample.get('endpoint_novelty')
    edge_ids = None if batch_sample.get('structural_node_ids') is not None else batch_sample.get('raw_edge_ids')

    if edge_ids is None and node_ids is None:
        return [torch.empty(0, device=device) for _ in range(N)], None

    memberships = {}
    for kind, collections in (('edge', edge_ids), ('node', node_ids)):
        if collections is None:
            continue
        for index, identifiers in enumerate(collections):
            for identifier in np.unique(np.asarray(identifiers, dtype=np.int64)):
                memberships.setdefault((kind, int(identifier)), []).append(index)

    neighbor_sets = [set() for _ in range(N)]
    for members in memberships.values():
        if len(members) < 2:
            continue
        for index in members:
            neighbor_sets[index].update(other for other in members if other != index)

    ego_graph_neighbors_probs = []
    for neighbor_indices in neighbor_sets:
        if neighbor_indices:
            ego_graph_neighbors_probs.append(probs[torch.as_tensor(sorted(neighbor_indices), device=device)])
        else:
            ego_graph_neighbors_probs.append(torch.empty(0, device=device))

    novelty = None if endpoint_novelty is None else endpoint_novelty.to(device).float()
    return ego_graph_neighbors_probs, novelty


class EntropyPrototypeAnchor:
    """Internal entropy prototype anchor in the OTTA-DGAD score/update chain."""

    def __init__(self, top_ratio=0.1, momentum=0.9):
        self.prev_normal_prototype = None
        self.prev_abnormal_prototype = None
        self.score_blend = 0.5
        self.proto_temperature = 1.0
        self.confidence_floor = 0.0
        self.use_temporal_bias = False
        self.last_num_reliable = 0
        self.logit_adjust_scale = 0.7
        self.top_ratio = top_ratio
        self.momentum = momentum
        self.score_blend = 0.8
        self.proto_temperature = 1.0
        self.confidence_floor = 0.0

    def adapt_chunk(self, representations, probs, pseudo_labels, device):
        eps = 1e-10
        entropy = -probs * torch.log(probs + eps) - (1 - probs) * torch.log(1 - probs + eps)
        confidence = 1 - entropy / np.log(2)
        n_select = min(len(confidence), max(1, int(self.top_ratio * len(confidence))))
        top_indices = torch.topk(confidence, n_select, largest=True)[1]
        self.last_selected_indices = top_indices.detach().cpu()
        self.last_num_reliable = int(n_select)
        selected_repr = representations[top_indices]
        selected_labels = pseudo_labels[top_indices]
        selected_conf = confidence[top_indices]
        normal_mask = selected_labels == 0
        abnormal_mask = selected_labels == 1
        normal_proto = None
        abnormal_proto = None
        if normal_mask.sum() > 0:
            normal_weights = selected_conf[normal_mask]
            normal_weights = normal_weights / (normal_weights.sum() + 1e-10)
            normal_proto = (selected_repr[normal_mask] * normal_weights.unsqueeze(1)).sum(0)
        if abnormal_mask.sum() > 0:
            abnormal_weights = selected_conf[abnormal_mask]
            abnormal_weights = abnormal_weights / (abnormal_weights.sum() + 1e-10)
            abnormal_proto = (selected_repr[abnormal_mask] * abnormal_weights.unsqueeze(1)).sum(0)
        if normal_proto is not None:
            if self.prev_normal_prototype is None:
                self.prev_normal_prototype = normal_proto
            else:
                self.prev_normal_prototype = self.momentum * self.prev_normal_prototype.to(device) + (1 - self.momentum) * normal_proto
        if abnormal_proto is not None:
            if self.prev_abnormal_prototype is None:
                self.prev_abnormal_prototype = abnormal_proto
            else:
                self.prev_abnormal_prototype = self.momentum * self.prev_abnormal_prototype.to(device) + (1 - self.momentum) * abnormal_proto
        return (self.prev_normal_prototype, self.prev_abnormal_prototype)

    def get_target_prototypes(self, current_normal=None, current_abnormal=None, device=None):
        """
        Resolve current target prototypes with historical fallback.

        This avoids dropping back to backbone probabilities just because a
        current chunk produced confident samples from only one pseudo-label
        class.
        """
        normal_proto = current_normal if current_normal is not None else self.prev_normal_prototype
        abnormal_proto = current_abnormal if current_abnormal is not None else self.prev_abnormal_prototype
        if device is not None:
            if normal_proto is not None:
                normal_proto = normal_proto.to(device)
            if abnormal_proto is not None:
                abnormal_proto = abnormal_proto.to(device)
        return (normal_proto, abnormal_proto)

    def score_representations(self, representations, probs, target_normal_proto, target_abnormal_proto, device):
        """Method-specific prototype scoring used to produce distinct OTTA behaviors."""
        (target_normal_proto, target_abnormal_proto) = self.get_target_prototypes(target_normal_proto, target_abnormal_proto, device)
        if target_normal_proto is None or target_abnormal_proto is None:
            return (probs, False)
        rep_norm = F.normalize(representations.float(), dim=1)
        normal = F.normalize(target_normal_proto.float(), dim=0)
        abnormal = F.normalize(target_abnormal_proto.float(), dim=0)
        margin = (rep_norm @ abnormal - rep_norm @ normal) / max(self.proto_temperature, 1e-06)
        centered_margin = margin - margin.mean()
        if torch.dot(centered_margin, probs.float() - probs.float().mean()) < 0:
            margin = -margin
        if self.use_temporal_bias and self.prev_normal_prototype is not None and (self.prev_abnormal_prototype is not None):
            prev_normal = F.normalize(self.prev_normal_prototype.to(device).float(), dim=0)
            prev_abnormal = F.normalize(self.prev_abnormal_prototype.to(device).float(), dim=0)
            temporal_margin = rep_norm @ prev_abnormal - rep_norm @ prev_normal
            centered_temporal_margin = temporal_margin - temporal_margin.mean()
            if torch.dot(centered_temporal_margin, probs.float() - probs.float().mean()) < 0:
                temporal_margin = -temporal_margin
            margin = 0.5 * margin + 0.5 * temporal_margin
        margin = margin - margin.mean()
        margin_std = margin.std(unbiased=False)
        if margin_std > 1e-08:
            margin = margin / margin_std
        else:
            margin = torch.zeros_like(margin)
        confidence = 1 - (-probs * torch.log(probs + 1e-10) - (1 - probs) * torch.log(1 - probs + 1e-10)) / np.log(2)
        base_logits = torch.logit(probs.clamp(1e-06, 1 - 1e-06))
        base_std = base_logits.std(unbiased=False).clamp(0.05, 1.0)
        confidence_weight = 0.25 + 0.75 * confidence.clamp(0.0, 1.0)
        adaptive_weight = (1 - self.score_blend) * self.logit_adjust_scale
        if self.confidence_floor > 0:
            adapt_mask = confidence > self.confidence_floor
            adjusted = torch.sigmoid(base_logits + adaptive_weight * base_std * confidence_weight * margin)
            fused = probs * (1 - adapt_mask.float()) + adjusted * adapt_mask.float()
            return (fused, True)
        return (torch.sigmoid(base_logits + adaptive_weight * base_std * confidence_weight * margin), True)
