import os
import glob
import numpy as np
import json
import pdb

import trajectory.utils as utils

DATASETS = [
	f'{env}-{buffer}'
	for env in ['hopper', 'walker2d', 'halfcheetah', 'ant']
	for buffer in ['medium-expert-v2', 'medium-v2', 'medium-replay-v2']
]

LOGBASE = 'logs'
TRIAL = '*'
EXP_NAME = 'plans/pretrained'

def load_results(paths):
	'''
		paths : path to directory containing experiment trials
	'''
	scores = []
	for i, path in enumerate(sorted(paths)):
		score = load_result(path)
		if score is None:
			print(f'Skipping {path}')
			continue
		scores.append(score)

		suffix = path.split('/')[-1]

	mean = np.mean(scores)
	err = np.std(scores) / np.sqrt(len(scores))
	return mean, err, scores

def load_result(path):
	'''
		path : path to experiment directory; expects `rollout.json` to be in directory
	'''
	fullpath = os.path.join(path, 'rollout.json')
	suffix = path.split('/')[-1]

	if not os.path.exists(fullpath):
		return None

	results = json.load(open(fullpath, 'rb'))
	score = results['score']
	return score * 100

#######################
######## setup ########
#######################

if __name__ == '__main__':

	class Parser(utils.Parser):
	    dataset: str = None

	args = Parser().parse_args()

	for dataset in ([args.dataset] if args.dataset else DATASETS):
		subdirs = glob.glob(os.path.join(LOGBASE, dataset, EXP_NAME))

		for subdir in subdirs:
			reldir = subdir.split('/')[-1]
			paths = glob.glob(os.path.join(subdir, TRIAL))

			mean, err, scores = load_results(paths)
			print(f'{dataset.ljust(30)} | {subdir.ljust(50)} | {len(scores)} scores \n    {mean:.2f} +/- {err:.2f}\n')
