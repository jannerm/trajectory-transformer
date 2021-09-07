import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pdb

from plotting.scores import means

class Colors:
	grey = '#B4B4B4'
	gold = '#F6C781'
	red = '#EC7C7D'
	blue = '#70ABCC'

def get_mean(results, exclude=None):
	'''
		results : { environment: score, ... }
	'''
	filtered = {
		k: v for k, v in results.items()
		if (not exclude) or (exclude and exclude not in k)
	}
	return np.mean(list(filtered.values()))

if __name__ == '__main__':

	#################
	## latex
	#################
	matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
	matplotlib.rc('text', usetex=True)
	matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
	#################

	fig = plt.gcf()
	ax = plt.gca()
	fig.set_size_inches(7.5, 3)

	means = {k: get_mean(v, exclude='ant') for k, v in means.items()}
	print(means)

	algs = ['BC', 'MBOP', 'BRAC', 'CQL', 'Decision\nTransformer', 'Trajectory\nTransformer']
	vals = [means[alg] for alg in algs]

	colors = [
		Colors.grey, Colors.gold,
		Colors.red, Colors.red, Colors.blue, Colors.blue
	]

	plt.bar(algs, vals, color=colors, edgecolor=Colors.gold, lw=0)
	plt.ylabel('Average normalized return', labelpad=15)
	plt.title('Offline RL Results')

	matplotlib.rcParams['hatch.linewidth'] = 7.5
	ax.patches[-1].set_hatch('/')

	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)

	plt.savefig('plotting/bar.pdf', bbox_inches='tight')
