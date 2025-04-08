# CODI: Continuous Chain-of-Thought via Self-Distillation

This repository is a PyTorch implementation of the paper [CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation](https://arxiv.org/abs/2502.21074).

## Overview

CODI is a novel framework that compresses Chain-of-Thought (CoT) reasoning into a continuous space through self-distillation. The model learns to reason in a compact continuous space, achieving performance comparable to explicit CoT while providing significant speed improvements.

Key features of CODI:
- Enables LLMs to reason in a compressed continuous space
- Achieves performance on par with explicit CoT with 3.1x compression
- Maintains interpretability by decoding continuous thoughts
- Works for both compact and verbose CoTs

## Project Structure

```
project/
│
├── main.py                  # Main training script
├── inference.py             # Inference and visualization script
│
├── models/
│   ├── __init__.py
│   └── codi_model.py        # Implementation of CODI model
│
├── data/
│   ├── __init__.py
│   └── dataset.py           # GSM8k dataset processing
│
└── utils/
    ├── __init__.py
    ├── trainer.py           # Training and evaluation functions
    └── visualization.py     # Visualization utilities for continuous thoughts
```

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/codi-implementation.git
cd codi-implementation

# Install dependencies
pip install torch transformers matplotlib numpy tqdm
```

## Data Preparation

CODI is trained on the GSM8k-Aug dataset, which is an augmented version of the GSM8k mathematical reasoning dataset. The data should be structured in JSONL format with fields for 'question', 'cot', and 'answer'.

Place your data in the `data/GSM8k-Aug/` directory with files:
- `train.jsonl`
- `validation.jsonl`
- `test.jsonl`

## Training

To train CODI, run the main script:

```bash
python main.py \
    --model gpt2 \
    --data_path data/GSM8k-Aug \
    --output_dir outputs \
    --num_latent 6 \
    --batch_size 8 \
    --effective_batch_size 128 \
    --learning_rate 3e-4 \
    --num_epochs 40 \
    --alpha 1.0 \
    --beta 1.0 \
    --gamma 1.0
```

For LLaMA models, use:

```bash
python main.py \
    --model meta-llama/Llama-2-1b \
    --data_path data/GSM8k-Aug \
    --output_dir outputs \
    --learning_rate 8e-5 \
    --gamma 20.0 \
    --num_epochs 10
```

## Inference

For inference and visualization of continuous thoughts:

```bash
python inference.py \
    --model_path outputs/best_model.pt \
    --model_name gpt2 \
    --question "Jenny buys 1 bag of cookies a week. The bag has 36 cookies and she puts 4 cookies in her son's lunch box 5 days a week. Her husband eats 1 cookie a day for 7 days. Jenny eats the rest of the cookies. How many cookies does Jenny eat?" \
    --visualize
```

## How CODI Works

CODI uses a self-distillation framework where the same model serves as both teacher and student:

1. **Teacher Task**: Learns to generate explicit CoT and the final answer using a language modeling objective.
2. **Student Task**: Generates continuous thoughts before predicting the final answer.
3. **Knowledge Distillation**: Aligns the hidden activation of the token preceding the final answer between teacher and student.

The key innovation is the single-step distillation in feature space, which mitigates forgetting issues inherent in curriculum learning approaches used in previous methods.

## Interpretability

CODI maintains interpretability by projecting continuous thoughts back into vocabulary space. The `inference.py` script can visualize these continuous thoughts, helping to understand the reasoning process.

## Citation

```
@article{shen2025codi,
  title={CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation},
  author={Shen, Zhenyi and Yan, Hanqi and Zhang, Linhai and Hu, Zhanghao and Du, Yali and He, Yulan},
  journal={arXiv preprint arXiv:2502.21074},
  year={2025}
}
```