# VDCNN — character-level text classification

[![Tests](https://github.com/mukesh-mehta/VDCNN/actions/workflows/ci.yml/badge.svg)](https://github.com/mukesh-mehta/VDCNN/actions/workflows/ci.yml)

A runnable PyTorch implementation inspired by
[Very Deep Convolutional Networks for Text Classification](https://arxiv.org/abs/1606.01781),
with CPU training, prediction, and checkpoint round trips. Historical TensorFlow
and toxic-comment experiments remain available for reference.

## Run a small experiment

Python 3.12 is tested. The sample data is synthetic and demonstrates the workflow;
it is not an accuracy benchmark.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt
python -m pytest -q
python vdcnn.py train examples/demo.csv artifacts/demo.pt --tiny --epochs 2 --length 128
python vdcnn.py predict artifacts/demo.pt "A team won the match" "A new software release"
```

Training prints JSON with per-epoch training losses. Prediction returns a class
ID and a probability vector for each text. The example uses `0=sport`, `1=tech`.
Choose a new checkpoint filename for each run; existing files are never replaced.
On Windows, activate with `.venv\Scripts\activate`.

## Data and model

Provide a UTF-8 CSV with `text,label` columns. Labels must be contiguous integer
IDs starting at zero and include at least two classes. Keep your own mapping from
IDs to human-readable names. Text is lowercased, truncated on the right, and padded.
The vocabulary uses lowercase ASCII letters, digits, punctuation, and space;
other characters use a distinct unknown ID. Padding and unknown tokens never
share an ID.

The model has a character embedding, an initial convolution, four convolutional
stages with batch normalization/ReLU, three downsampling operations, temporal
k-max pooling, and three dense layers. The configurable convolution counts are
9, 17, 29, and 49. K-max selection preserves the selected positions' temporal order.
The output is **logits**; the CLI uses cross-entropy for single-label classes.

```python
from vdcnn import Config, VDCNN, encode
model = VDCNN(Config(classes=6, depth=17))
logits = model(encode(["An example document"], length=256))
```

Default width is 64 and dense width is 2048. `--tiny` changes them to 8 and 64 for
quick CPU experiments. Default k is 8; sequence length must be at least 57 for
three pooling stages. The encoder itself accepts lengths of at least 8, which is
useful with smaller k. `--threads` is a global option, e.g.
`python vdcnn.py --threads 1 train ...`.

## Features and remaining work

| Capability | Status |
| --- | --- |
| Four configurable VDCNN depths | Implemented |
| Separate padding/unknown characters | Implemented |
| Temporal k-max pooling | Implemented |
| CPU CSV training and batched prediction | Implemented |
| Small model preset | Implemented |
| Seeded shuffling and gradient clipping | Implemented |
| Checkpoint metadata and weights-only loading | Implemented |
| Forward/backward and reload regression tests | Implemented |
| Held-out validation and early stopping | Planned |
| Multilabel toxic-comment CLI | Planned |
| GPU training and resume optimizer state | Planned |
| Paper-dataset benchmark reproduction | Planned |

The training CLI currently reports **training loss only**. Create a separate
held-out evaluation before comparing models; the tiny demo is not evidence of
useful generalization. This implementation does not claim to reproduce the paper:
it uses AdamW in the CLI, a customizable alphabet/head, and no residual shortcuts.

## Repository guide

- `vdcnn.py`: maintained model, tokenizer, training, and prediction.
- `examples/demo.csv`: small original synthetic dataset.
- `tests/`: all four depths, gradient flow, encoding, pooling, and checkpoint tests.
- `VDCNN_tf.py`, `toxic.ipynb`: historical experiments; they depend on old Keras,
  TensorFlow, and external data and are not run by current CI.
- `docs/ORIGINAL_README.md`: original reported Kaggle results, not revalidated.

The paper PDF is third-party material and remains subject to its own terms.
New maintained code and documentation use [MIT](LICENSE); read [NOTICE.md](NOTICE.md)
for exclusions. See [CONTRIBUTING.md](CONTRIBUTING.md) for development.
