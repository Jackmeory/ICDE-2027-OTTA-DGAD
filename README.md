## Packages

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

## Training

```bash
python train.py \
  --data-dir /path/to/data \
  --source-datasets source_dataset_1 source_dataset_2 \
  --train-fraction 0.5 \
  --epochs 50 --batch-size 128 --learning-rate 0.001 \
  --prototype-bank-fraction 0.1 --classification-weight 0.5 \
  --output checkpoints/source.pt \
  --device cuda:0 --seed 95540
```

## Testing

```bash
python test.py \
  --data-dir /path/to/data \
  --dataset tax51_full \
  --checkpoint checkpoints/source.pt \
  --stream-split full \
  --chunk-size 128 --buffer-size 100 \
  --output outputs/test \
  --device cuda:0 --seed 95540
```
