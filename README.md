# Trajectory Transformer

Code release for [Reinforcement Learning as One Big Sequence Modeling Problem](https://arxiv.org/abs/2106.02039).

## Installation

All python dependencies are in [`environment.yml`](environment.yml). Install with:

```
conda env create -f environment.yml
conda activate trajectory
pip install -e .
```

For reproducibility, we have also included system requirements in a [`Dockerfile`](azure/Dockerfile) (see [installation instructions](#Docker)), but the conda installation should work on most standard Linux machines.

## Usage

Train a transformer with:
```
python scripts/train.py --dataset halfcheetah-medium-expert-v2
```

To reproduce the offline RL results:
```
python scripts/plan.py --dataset halfcheetah-medium-expert-v2
```

By default, this will use the hyperparameters in [`config/offline.py`](config/offline.py). You can override any hyperparameter with a runtime flag, _e.g._:
```
python scripts/plan.py --dataset halfcheetah-medium-expert-v2 \
	--horizon 5 --beam_width 32
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

#### Setup

1. Launching jobs on Azure requires one more python dependency:
```
pip install git+https://github.com/JannerM/doodad.git@janner
```

2. Tag the image built in [the previous section](#Docker) and push it to dockerhub:
```
export DOCKER_USERNAME=$(docker info | sed '/Username:/!d;s/.* //')
docker tag trajectory ${DOCKER_USERNAME}/trajectory:latest
docker image push ${DOCKER_USERNAME}/trajectory
```

3. Update [`azure/config.py`](azure/config.py), either by modifying the file directly or setting the relevant [environment variables](azure/config.py#L47-L52). To set the `AZURE_STORAGE_CONNECTION` variable, navigate to the `Access keys` section of your storage account. Click `Show keys` and copy the `Connection string`.

4. Download [`azcopy`](https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10) to `bin`:
```
./azure/download.sh
```

#### Usage

Launch training jobs with
```
python azure/launch_train.py
```
and planning jobs with
```
python azure/launch_plan.py
```

These scripts do not take runtime arguments. Instead, they run corresponding scripts (`scripts/train.py` and `scripts/plan.py`, respectively) using the Cartesian product of the parameters in [`params_to_sweep`](azure/launch_train.py#L36-48).

#### Viewing results

To rsync the results from the Azure storage container, run
```
./azure/sync.sh
```

To mount the storage container, first create a blobfuse config with
```
./azure/make_fuse_config.sh
```
and then mount with
```
./azure/mount.sh
```
This will mount the storage container to a new folder called `mount/`. To unmount and remove the folder, run
```
./azure/umount.sh
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
