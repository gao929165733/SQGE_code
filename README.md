This code is for paper "Support-Query Prototype Guidance and Enhanced Few-Shot Relational Triple Extraction"

Requirements
---

```
python=3.7
pytorch=1.9.1
cuda=10.2
transformers=2.8.0
```

Usage
---


1. Training

For example, if you want to train our SQGE in 5-way-5-shot setting, it should be like:

```shell
python main.py --model=proto_dot --trainN=5 --evalN=5 --K=5 --Q=1 --O=0 --distance_metric="conv" --contrastive="Normal" --temperature_alpha=0.5 --temperature_beta=0.1 --alpha=0.5 --beta=0.5 --threshold="None" --result_dir="result/proto_hcl"
```


2. Testing

For example, if you want to test our SQGE in 5-way-5-shot setting,  it should be like:

```shell
python main.py --model=proto_dot --trainN=5 --evalN=5 --K=5 --Q=1 --O=0 --distance_metric="conv" --contrastive="Normal" --temperature_alpha=0.5 --temperature_beta=0.1 --alpha=0.5 --beta=0.5 --threshold="None" --load_ckpt="proto_dot_fewrel_xxxx" --result_dir="result/proto_hcl"
```

