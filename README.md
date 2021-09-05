## Trajectory Transformer

### Installation

1. `conda env create -f environment.yml`
2. `conda activate trajectory`
2. `pip install -e .`

### Usage

Train a transformer with:
```
python scripts/train.py --dataset halfcheetah-medium-expert-v2
```

To reproduce the offline RL results:
```
python scripts/plan.py --dataset halfcheetah-medium-expert-v2
```

By default, this will use the hyperparameters in [config/offline.py](config/offline.py). You can override any hyperparameter with a runtime flag:
```
python scripts/plan.py --dataset halfcheetah-medium-expert-v2 --horizon 4 --beam_width 256
```

### Docker

```
cp ~/.mujoco/mjkey.txt azure/files/ && docker build -f azure/Dockerfile . -t trajectory
```

Test the container with:
```
docker run -it --rm --gpus all \
	--mount type=bind,source=/home/janner/mount/trajectory-transformer,target=/home/code \
	trajectory \
	bash -c \
	"export PYTHONPATH=$PYTHONPATH:/home/code && \
	python /home/code/scripts/train.py --dataset hopper-medium-expert-v2 --exp_name docker/"
```

### Running on Azure

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

Update [azure/config.py](azure/config.py), either by modifying the file directly or setting the relevant [environment variables]((azure/config.py#L35-L40).

Finally, launch with:
```
python azure/launch_train.py
```

### Acknowledgements

The GPT implementation is from Andrej Karpathy's [minGPT](https://github.com/karpathy/minGPT) repo.
