"""OTTA-DGAD online test entry point."""
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from data import ChunkedDygDataset
from runtime import OnlineOTTA, load_model, resolve_device, set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('outputs/test'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=95540)
    parser.add_argument('--stream-split', choices=['test', 'full'], default='test')
    parser.add_argument('--chunk-size', type=int, default=128)
    parser.add_argument('--n-chunks', type=int)
    parser.add_argument('--anomaly-ratio', type=float, default=0.1)
    parser.add_argument('--buffer-size', type=int, default=100)
    parser.add_argument('--delta', type=float, default=2.0)
    parser.add_argument('--lambda-fusion', type=float, default=0.3)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--top-n-ratio', type=float, default=0.1)
    parser.add_argument('--otta-score-blend', type=float, default=0.1)
    args = parser.parse_args()
    if args.chunk_size < 1 or args.buffer_size < 1 or args.top_k < 1 or args.delta <= 0 or args.tau <= 0:
        parser.error('Sizes, delta and tau must be positive')
    if args.n_chunks is not None and args.n_chunks < 1:
        parser.error('--n-chunks must be positive')
    if not 0 <= args.lambda_fusion <= 1 or not 0 < args.top_n_ratio <= 1 or not 0 <= args.otta_score_blend <= 1:
        parser.error('Invalid fusion, selection or blend ratio')
    set_seed(args.seed)
    device = resolve_device(args.device)
    model, config, checkpoint, statistics = load_model(args.checkpoint, device)
    data_config = SimpleNamespace(**vars(config), dir_data=str(args.data_dir),
                                  data_set=args.dataset, stream_split=args.stream_split,
                                  anomaly_ratio=args.anomaly_ratio)
    stream = ChunkedDygDataset(data_config, args.chunk_size)
    count = min(len(stream), args.n_chunks) if args.n_chunks else len(stream)
    if count == 0:
        parser.error('The requested stream is empty')
    adapter = OnlineOTTA(model, checkpoint, statistics, device,
                         buffer_size=args.buffer_size, delta=args.delta,
                         lambda_fusion=args.lambda_fusion, top_k=args.top_k,
                         tau=args.tau, top_n_ratio=args.top_n_ratio,
                         otta_score_blend=args.otta_score_blend)
    args.output.mkdir(parents=True, exist_ok=True)
    labels, predictions = [], []
    start = stream.sharded_stream.start if stream.sharded_stream is not None else stream.event_start
    with (args.output/'predictions.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['event_index', 'score', 'label'])
        for i in range(count):
            batch = stream[i]
            scores = adapter.step(batch)
            y = batch['labels'].reshape(-1).cpu().numpy()
            if not np.isfinite(scores).all():
                raise RuntimeError('Non-finite anomaly scores')
            writer.writerows((start+j, float(score), int(label)) for j, (score, label) in enumerate(zip(scores, y)))
            start += len(scores)
            labels.extend(y.tolist()); predictions.extend(scores.tolist())
            if (i+1) % 10 == 0 or i+1 == count:
                print(f'Chunk {i+1}/{count}, events={len(labels)}', flush=True)
    metrics = dict(method='OTTA-DGAD', events=len(labels), chunks=count,
                   auroc=None, auprc=None, context_items=len(adapter.context.buffer))
    if len(set(labels)) == 2:
        metrics.update(auroc=float(roc_auc_score(labels, predictions)),
                       auprc=float(average_precision_score(labels, predictions)))
    (args.output/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    print(json.dumps(metrics))


if __name__ == '__main__':
    main()
