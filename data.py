"""Load external processed graph datasets; no dataset is bundled."""
from pathlib import Path
import json,pickle
import numpy as np
import torch


class ShardedTax51Stream:
    """Lazily materialize full TAX51 chunks from temporal graph shards."""
    def __init__(self, root, chunk_size, anomaly_ratio, stream_split, train_fraction=0.5):
        self.root = Path(root)
        with (self.root / 'manifest.json').open() as handle:
            self.manifest = json.load(handle)
        self.format = self.manifest.get('format')
        if self.format not in {'tax51_temporal_shards_v1', 'tax51_temporal_shards_v2'}:
            raise ValueError(f"Unsupported TAX51 stream format in {self.root}")
        self.chunk_size = chunk_size
        self.shard_size = int(self.manifest['shard_size'])
        self.total_samples = int(self.manifest['num_events'])
        self.start = 0 if stream_split == 'full' else self.total_samples // 2
        self.count = self.total_samples - self.start
        if stream_split in {"train", "validation"}:
            boundary = int(self.total_samples * train_fraction)
            self.start = 0 if stream_split == "train" else boundary
            self.count = boundary if stream_split == "train" else self.total_samples - boundary
        ratio_key = f'{float(anomaly_ratio):.2f}'
        if ratio_key not in self.manifest['ratio_keys']:
            available = ', '.join(self.manifest['ratio_keys'])
            raise ValueError(f"Anomaly ratio {anomaly_ratio} is unavailable; choose one of {available}")
        self.ratio_key = ratio_key
        self._cache = {}

    @staticmethod
    def _features(ids, feature_dim, salt):
        ids = np.asarray(ids, dtype=np.float32).reshape(-1, 1)
        dimensions = np.arange(feature_dim, dtype=np.float32).reshape(1, -1)
        values = np.sin((ids + 1.0) * 12.9898 + (dimensions + 1.0) * 78.233 + salt)
        return (values * 43758.5453 - np.floor(values * 43758.5453)).astype(np.float32)

    @staticmethod
    def _graph_tensors(node_count, edge_pairs):
        edge_count = len(edge_pairs)
        transition = np.zeros((node_count, edge_count), dtype=np.float32)
        adjacency = np.eye(node_count, dtype=np.float32)
        for edge_index, (left, right) in enumerate(edge_pairs):
            transition[left, edge_index] = 1.0
            transition[right, edge_index] = 1.0
            adjacency[left, right] = adjacency[right, left] = 1.0
        adjacency /= adjacency.sum(axis=1, keepdims=True).clip(min=1.0)
        edge_adjacency = np.eye(edge_count, dtype=np.float32)
        for left in range(edge_count):
            for right in range(left + 1, edge_count):
                if len(set(edge_pairs[left]) & set(edge_pairs[right])):
                    edge_adjacency[left, right] = edge_adjacency[right, left] = 1.0
        edge_adjacency /= edge_adjacency.sum(axis=1, keepdims=True).clip(min=1.0)
        return torch.from_numpy(transition), torch.from_numpy(adjacency), torch.from_numpy(edge_adjacency)

    def _shard(self, shard_index):
        if shard_index not in self._cache:
            with (self.root / self.manifest['shards'][shard_index]).open('rb') as handle:
                self._cache[shard_index] = pickle.load(handle)
            if len(self._cache) > 2:
                self._cache.pop(next(iter(self._cache)))
        return self._cache[shard_index]

    def _record(self, index):
        shard_index, local_index = divmod(index, self.shard_size)
        shard = self._shard(shard_index)
        graphs = (shard['graphs_by_ratio'][self.ratio_key]
                  if self.format == 'tax51_temporal_shards_v2' else shard)
        return (
            graphs['nodefeatures'][local_index],
            graphs['edgefeatures'][local_index],
            graphs['edge_pairs'][local_index],
            shard['labels_by_ratio'][self.ratio_key][local_index],
        )

    def __len__(self):
        return (self.count + self.chunk_size - 1) // self.chunk_size

    def labels_for_ratio(self, anomaly_ratio):
        ratio_key = f'{float(anomaly_ratio):.2f}'
        if ratio_key not in self.manifest['ratio_keys']:
            raise ValueError(f"Anomaly ratio {anomaly_ratio} is unavailable")
        labels = []
        start, end = self.start, self.start + self.count
        for index in range(start, end):
            shard_index, local_index = divmod(index, self.shard_size)
            labels.append(self._shard(shard_index)['labels_by_ratio'][ratio_key][local_index])
        return np.asarray(labels, dtype=np.int64)

    def chunk(self, chunk_index, feature_dim):
        if chunk_index < 0 or chunk_index >= len(self):
            raise IndexError(chunk_index)
        start = self.start + chunk_index * self.chunk_size
        end = min(start + self.chunk_size, self.start + self.count)
        records = [self._record(index) for index in range(start, end)]
        max_edges = max(len(record[1]) for record in records)
        edge_pad = np.zeros((len(records), max_edges, feature_dim), dtype=np.float32)
        mask = np.ones((len(records), max_edges), dtype=np.float32)
        node_features, edge_features, raw_node_ids, raw_edge_ids, structural_node_ids, endpoint_novelty, transitions, adjs, eadjs, ranks, labels = [], [], [], [], [], [], [], [], [], [], []
        for index, (nodes, edges, pairs, label) in enumerate(records):
            raw_node_ids.append(np.asarray(nodes, dtype=np.int64))
            raw_edge_ids.append(np.asarray(edges, dtype=np.int64))
            structural_node_ids.append(np.asarray(nodes[:2], dtype=np.int64))
            prior_pairs = [pair for pair in pairs if tuple(pair[:2]) != (0, 1)]
            endpoint_degrees = [
                sum(int(endpoint in pair[:2]) for pair in prior_pairs)
                for endpoint in (0, 1)
            ]
            endpoint_novelty.append(1.0 / (1.0 + min(endpoint_degrees)))
            node_features.append(torch.from_numpy(self._features(nodes, feature_dim, 0.0)))
            edge_feature = self._features(edges, feature_dim, 19.19)
            edge_features.append(torch.from_numpy(edge_feature))
            edge_pad[index, :len(edges)] = edge_feature
            mask[index, :len(edges)] = 0.0
            transition, adjacency, edge_adjacency = self._graph_tensors(len(nodes), pairs)
            transitions.append(transition)
            adjs.append(adjacency)
            eadjs.append(edge_adjacency)
            ranks.append(torch.arange(len(nodes), dtype=torch.float32))
            labels.append(int(label))
        return {
            'input_nodes_feature': node_features,
            'input_edges_feature': edge_features,
            'input_edges_pad': torch.from_numpy(edge_pad),
            'raw_node_ids': raw_node_ids,
            'raw_edge_ids': raw_edge_ids,
            'structural_node_ids': structural_node_ids,
            'endpoint_novelty': torch.tensor(endpoint_novelty, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'Tmats': transitions,
            'adjs': adjs,
            'eadjs': eadjs,
            'mask_edge': torch.from_numpy(mask),
            'ra': ranks,
            'chunk_idx': chunk_index,
        }


class ChunkedDygDataset:
    """
    Dataset class that loads data and prepares it for chunk-wise streaming.
    Sorts all interactions by timestamp and splits into consecutive chunks.
    """
    def __init__(self, config, chunk_size=128):
        self.config = config
        self.chunk_size = chunk_size
        stream_root = Path(config.dir_data) / config.data_set
        if (stream_root / 'manifest.json').exists():
            self.sharded_stream = ShardedTax51Stream(
                stream_root, chunk_size, config.anomaly_ratio, config.stream_split,
                getattr(config, "train_fraction", 0.5)
            )
            self.chunks = None
            print(f"Prepared {len(self.sharded_stream)} lazy chunks of size {chunk_size}")
            return
        self.sharded_stream = None
        dataset_path = f'{config.dir_data}/{config.data_set}.pkl'

        with open(dataset_path, 'rb') as file:
            data = pickle.load(file)

        # Get test split data
        self.chunks = self.prepare_chunks(data)
        if not self.chunks:
            raise ValueError("The requested dataset split is empty")
        print(f"Prepared {len(self.chunks)} chunks of size {chunk_size}")

    def prepare_chunks(self, data):
        """
        Prepare chunk-wise streaming data.

        Steps:
        1. Load all data
        2. Sort by timestamp (already sorted in temporal order in the dataset)
        3. Split into consecutive non-overlapping chunks
        """
        node_features = data['nodefeatures']
        edge_features = data['edgefeatures']
        labels = data['labels']
        Tmats = data['Tmats']
        adjs = data['adjs']
        eadjs = data['eadjs']
        ra = data['ra']

        # Generate random features
        flattened_node = np.concatenate([arr for arr in node_features])
        unique_node = np.unique(flattened_node)
        num_nodes = int(max(unique_node.size, np.max(unique_node) + 1))

        flattened_edge = np.concatenate([arr for arr in edge_features])
        unique_edge = np.unique(flattened_edge)
        num_edges = int(max(unique_edge.size, np.max(unique_edge) + 1))

        Nfeatures = np.random.uniform(low=0.0, high=1.0, size=(num_nodes, self.config.input_dim))
        Efeatures = np.random.uniform(low=0.0, high=1.0, size=(num_edges, self.config.input_dim))

        # Determine split index (use test split only)
        if self.config.data_set == 'btc_alpha':
            split_indices = 7000
        elif self.config.data_set == 'btc_otc':
            split_indices = 10000
        else:
            split_indices = int(num_edges / 2)

        # Full-stream experiments opt out of the legacy half-stream target split.
        if self.config.stream_split in {'train', 'validation'}:
            split_indices = int(len(labels) * self.config.train_fraction)
        if self.config.stream_split == 'train':
            node_features, edge_features, labels, Tmats, adjs, eadjs, ra = (
                x[:split_indices] for x in (node_features, edge_features, labels, Tmats, adjs, eadjs, ra)
            )
        elif self.config.stream_split != 'full':
            node_features = node_features[split_indices:]
            edge_features = edge_features[split_indices:]
            labels = labels[split_indices:]
            Tmats = Tmats[split_indices:]
            adjs = adjs[split_indices:]
            eadjs = eadjs[split_indices:]
            ra = ra[split_indices:]

        self.event_start = 0 if self.config.stream_split in {'train', 'full'} else split_indices
        # Prepare features
        NS = len(edge_features)
        max_mask_edge = max(len(arr) for arr in edge_features)
        masks_edge = [len(edge) for edge in edge_features]

        mask_edge = np.ones((NS, max_mask_edge))
        for i in range(NS):
            mask_edge[i, :masks_edge[i]] = 0

        # Pad edge features
        hidden = Efeatures.shape[1]
        input_edges_pad = np.zeros((NS, max_mask_edge, hidden))
        for i, indices in enumerate(edge_features):
            indices = indices.astype(int)
            input_edges_pad[i, :len(indices), :] = Efeatures[indices]

        input_edges_feature = [torch.tensor(input_edges_pad[i, :masks_edge[i], :]) for i in range(NS)]

        # Pad node features
        max_mask_node = max(len(arr) for arr in node_features)
        masks_node = [len(node) for node in node_features]

        hidden = Nfeatures.shape[1]
        input_nodes_pad = np.zeros((NS, max_mask_node, hidden))
        for i, indices in enumerate(node_features):
            indices = indices.astype(int)
            input_nodes_pad[i, :len(indices), :] = Nfeatures[indices]

        input_nodes_feature = [torch.tensor(input_nodes_pad[i, :masks_node[i], :]) for i in range(NS)]

        input_edges_pad = torch.tensor(input_edges_pad)
        labels = torch.tensor(labels)
        mask_edge = torch.tensor(mask_edge)

        # Split into chunks
        chunks = []
        total_samples = len(labels)

        for start_idx in range(0, total_samples, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, total_samples)

            chunk_data = {
                'input_nodes_feature': input_nodes_feature[start_idx:end_idx],
                'input_edges_feature': input_edges_feature[start_idx:end_idx],
                'input_edges_pad': input_edges_pad[start_idx:end_idx],
                'raw_node_ids': node_features[start_idx:end_idx],
                'raw_edge_ids': edge_features[start_idx:end_idx],
                'labels': labels[start_idx:end_idx],
                'Tmats': Tmats[start_idx:end_idx],
                'adjs': adjs[start_idx:end_idx],
                'eadjs': eadjs[start_idx:end_idx],
                'mask_edge': mask_edge[start_idx:end_idx],
                'ra': ra[start_idx:end_idx],
                'chunk_idx': len(chunks)
            }
            chunks.append(chunk_data)

        return chunks

    def __len__(self):
        if self.sharded_stream is not None:
            return len(self.sharded_stream)
        return len(self.chunks)

    def __getitem__(self, idx):
        if self.sharded_stream is not None:
            return self.sharded_stream.chunk(idx, self.config.input_dim)
        return self.chunks[idx]

    def labels_for_ratio(self, anomaly_ratio):
        if self.sharded_stream is None:
            raise RuntimeError('Stored ratio labels are only available for a sharded TAX51 stream')
        return self.sharded_stream.labels_for_ratio(anomaly_ratio)
