# Trajectory Transformer

Code release for [Reinforcement Learning as One Big Sequence Modeling Problem](https://arxiv.org/abs/2106.02039).

## Installation

```
conda env create -f environment.yml
conda activate trajectory
pip install -e .
```

## Usage

Train a transformer with:
```
python scripts/train.py --dataset halfcheetah-medium-expert-v2
```

To reproduce the offline RL results:
```
python scripts/plan.py --dataset halfcheetah-medium-expert-v2
```

By default, this will use the hyperparameters in [config/offline.py](config/offline.py). You can override any hyperparameter with a runtime flag, _e.g._:
```
python scripts/plan.py --dataset halfcheetah-medium-expert-v2 \
	--horizon 4 --beam_width 256
```

## Docker

Copy your MuJoCo key to the Docker build context and build the container:
```
cp ~/.mujoco/mjkey.txt azure/files/
docker build -f azure/Dockerfile . -t trajectory
```

Test the container:
```
docker run -it --rm --gpus all \
	--mount type=bind,source=$PWD,target=/home/code \
	--mount type=bind,source=$HOME/.d4rl,target=/root/.d4rl \
	trajectory \
	bash -c \
	"export PYTHONPATH=$PYTHONPATH:/home/code && \
	python /home/code/scripts/train.py --dataset hopper-medium-expert-v2 --exp_name docker/"
```

## Running on Azure

Launching jobs on Azure requires one more dependency:
```
pip install git+https://github.com/JannerM/doodad.git@janner
```

Tag the image and push it to dockerhub:
```
export DOCKER_USERNAME=$(docker info | sed '/Username:/!d;s/.* //')
docker tag trajectory ${DOCKER_USERNAME}/trajectory:latest
docker image push ${DOCKER_USERNAME}/trajectory
```

Update [azure/config.py](azure/config.py), either by modifying the file directly or setting the relevant [environment variables](azure/config.py#L35-L40).

To set the `AZURE_STORAGE_CONNECTION` variable, navigate to the `Access keys` section of your storage account. Click `Show keys` and copy the `Connection string`.

Finally, launch with:
```
python azure/launch_train.py
```

## Reference
```
@article{janner2020sequence,
  title={Reinforcement Learning as One Big Sequence Modeling Problem},
  author={Michael Janner and Qiyang Li and Sergey Levine},
  journal={arXiv preprint arXiv:2106.02039},
  year={2021},
}
```

## Acknowledgements

The GPT implementation is from Andrej Karpathy's [minGPT](https://github.com/karpathy/minGPT) repo.
