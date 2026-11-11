"""Sequential source pretraining for the OTTA-DGAD backbone."""
import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from data import ChunkedDygDataset
from runtime import (MODEL_DEFAULTS, build_model, empty_statistics, forward_batch,
                     resolve_device, set_seed)


@torch.no_grad()
def prototype_relevance(model, stream, device, bank, statistics, limit):
    """Rank source-memory pairs using labeled source embeddings only."""
    pairs = [torch.stack([entry[label] for entry in bank]).to(device) for label in (0, 1)]
    sums = [torch.zeros(len(bank), device=device) for _ in range(2)]
    counts = [0, 0]
    model.eval()
    for i in range(limit):
        batch = stream[i]
        _, embeddings, *_ = forward_batch(model, batch, device, bank[0][:2], statistics)
        labels = batch['labels'].to(device).reshape(-1)
        for label in (0, 1):
            mask = labels == label
            if mask.any():
                sums[label] += torch.cdist(pairs[label], embeddings[mask]).sum(1)
            counts[label] += int(mask.sum())
    if not all(counts):
        return [0.5]*len(bank)
    relevance = -0.5*(sums[0]/counts[0] + sums[1]/counts[1])
    span = relevance.max()-relevance.min()
    return ((relevance-relevance.min())/span).cpu().tolist() if span > 0 else [0.5]*len(bank)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--source-datasets', nargs='+', required=True)
    parser.add_argument('--output', type=Path, default=Path('checkpoints/source.pt'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=95540)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--train-fraction', type=float, default=0.5)
    parser.add_argument('--prototype-bank-fraction', type=float, default=0.1)
    parser.add_argument('--classification-weight', type=float, default=0.5)
    parser.add_argument('--relevance-weight', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=4)
    parser.add_argument('--max-chunks', type=int)
    parser.add_argument('--anomaly-ratio', type=float, default=0.1)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.learning_rate <= 0 or args.patience < 1:
        parser.error('epochs/patience must be positive, batch-size >= 2, learning-rate > 0')
    if not 0 < args.train_fraction <= 1 or not 0 < args.prototype_bank_fraction <= 1:
        parser.error('Training and prototype-bank fractions must be in (0, 1]')
    if not 0 <= args.classification_weight <= 1 or not 0 <= args.relevance_weight <= 1:
        parser.error('Loss/relevance weights must be in [0, 1]')
    if args.max_chunks is not None and args.max_chunks < 1:
        parser.error('--max-chunks must be positive')
    set_seed(args.seed)
    device = resolve_device(args.device)
    config = SimpleNamespace(**MODEL_DEFAULTS)
    model = build_model(config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    bank, relevance, history, trained = [], [], [], []
    best_index = 0
    for source_index, dataset in enumerate(args.source_datasets):
        data_config = SimpleNamespace(**vars(config), dir_data=str(args.data_dir),
                                      data_set=dataset, stream_split='train',
                                      train_fraction=args.train_fraction,
                                      anomaly_ratio=args.anomaly_ratio)
        stream = ChunkedDygDataset(data_config, args.batch_size)
        limit = min(len(stream), args.max_chunks) if args.max_chunks else len(stream)
        if not limit:
            raise ValueError(f'Empty source training stream: {dataset}')
        events = stream.sharded_stream.count if stream.sharded_stream else sum(len(c['labels']) for c in stream.chunks)
        capacity = max(1, int(events*args.prototype_bank_fraction))
        statistics = {k: v.to(device) for k, v in empty_statistics(config.input_dim).items()}
        if bank:
            relevance = prototype_relevance(model, stream, device, bank, statistics, limit)
        else:
            relevance = []
        combined = [args.relevance_weight*r+(1-args.relevance_weight)*p[2] for r, p in zip(relevance, bank)]
        best_index = int(np.argmax(combined)) if combined else 0
        best_loss, stale = math.inf, 0
        for epoch in range(args.epochs):
            model.train()
            loss_sum, samples = 0.0, 0
            for i in range(limit):
                batch = stream[i]
                labels = batch['labels'].to(device).float().reshape(-1)
                if not torch.all((labels == 0) | (labels == 1)):
                    raise ValueError('Source labels must be 0 (normal) or 1 (anomalous)')
                # BatchNorm cannot estimate variance from a singleton tail.
                model.Transformer.bn.train(len(labels) > 1)
                prompts = (bank[best_index][:2] if bank and (epoch > 0 or source_index > 0)
                           else (torch.randn(config.input_dim, device=device), torch.randn(config.input_dim, device=device)))
                optimizer.zero_grad()
                scores, embeddings, normal, abnormal, nm, nc, ac, am = forward_batch(
                    model, batch, device, prompts, statistics)
                statistics = dict(normal_mean=nm.detach(), normal_cov=nc.detach(),
                                  abnormal_mean=am.detach(), abnormal_cov=ac.detach())
                # The classifier returns probabilities, not unnormalized logits.
                classify = F.binary_cross_entropy(scores.reshape(-1).clamp(1e-6, 1-1e-6), labels)
                normal_mask, abnormal_mask = labels == 0, labels == 1
                alignment = embeddings.sum()*0
                has_pair = bool(normal_mask.any() and abnormal_mask.any())
                if has_pair:
                    alignment = (torch.linalg.vector_norm(embeddings[abnormal_mask]-abnormal, dim=1).mean() +
                                 0.1*torch.linalg.vector_norm(embeddings[normal_mask]-normal, dim=1).mean())
                loss = args.classification_weight*classify + (1-args.classification_weight)*alignment
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite source training loss')
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise RuntimeError('Non-finite source gradients')
                optimizer.step()
                if has_pair:
                    separation = float(torch.sigmoid(0.1*torch.linalg.vector_norm(normal-abnormal)))
                    candidate = (normal.detach().cpu().clone(), abnormal.detach().cpu().clone(), separation)
                    candidate_score = args.relevance_weight + (1-args.relevance_weight)*separation
                    if len(bank) < capacity:
                        bank.append(candidate); relevance.append(1.0); combined.append(candidate_score)
                    else:
                        worst = int(np.argmin(combined))
                        if candidate_score > combined[worst]:
                            if source_index > 0:
                                candidate = (embeddings[normal_mask].mean(0).detach().cpu().clone(),
                                             embeddings[abnormal_mask].mean(0).detach().cpu().clone(), separation)
                            bank[worst] = candidate; relevance[worst] = 1.0; combined[worst] = candidate_score
                    best_index = int(np.argmax(combined))
                loss_sum += float(loss.detach())*len(labels)
                samples += len(labels)
            mean_loss = loss_sum/samples
            row = dict(dataset=dataset, epoch=epoch+1, loss=mean_loss, events=samples,
                       prototype_pairs=len(bank))
            history.append(row); print(json.dumps(row), flush=True)
            if mean_loss < best_loss:
                best_loss, stale = mean_loss, 0
            else:
                stale += 1
                if stale >= args.patience:
                    break
        trained.append(dataset)
    if not bank:
        raise RuntimeError('No source prototype pair was built; source batches must contain both classes')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                      model_config=vars(config), prototype_buffer=bank, best_prototype_idx=best_index,
                      prototype_relevance_scores=relevance, source_datasets=trained,
                      training_args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                      **{k: v.detach().cpu() for k, v in statistics.items()})
    torch.save(checkpoint, args.output)
    args.output.with_suffix('.training.json').write_text(json.dumps(history, indent=2)+'\n')
    print(f'Saved checkpoint: {args.output}', flush=True)


if __name__ == '__main__':
    main()
