## Trajectory Transformer

### Installation

1. `conda env create -f environment.yml`
2. `conda activate trajectory`
2. `pip install -e .`

### Training

`python scripts/train.py --dataset halfcheetah-medium-expert-v2`

### Acknowledgements

The GPT implementation is from Andrej Karpathy's [minGPT](https://github.com/karpathy/minGPT) repo.
