# ICDE-2027-OTTA-DGAD

This repository is the official implementation of paper "Online Test-Time Adaptation for Generalizable
Dynamic Graph Anomaly Detection"
	

## Requirements
To install requirements:
```Python
pip install -r requirements.txt 
```
## Pretrain on source datasets
To train OTTA-DGAD on source datasests, please run the following code
```Python
python train_source.py 
```
## Update and test on target datasets
To update OTTA-DGAD on target datasests, please run the following code
```Python
python infer_target.py 
```
## Additional Notes
This code take reference from [GeneralDyG](https://github.com/YXNTU/GeneralDyG?tab=readme-ov-file), AAAI 2025; [ARC](https://github.com/yixinliu233/ARC?tab=readme-ov-file), NeurIPS 2024; [AnomalyGFM](https://github.com/mala-lab/anomalygfm), KDD 2025. We want to express our sincere thanks to above authors for their well structured open-source code!
