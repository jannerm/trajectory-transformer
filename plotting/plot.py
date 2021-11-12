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

LABELS = {
	# 'BC': 'Behavior\nCloning',
	# 'MBOP': 'Model-Based\nOffline Planning',
	# 'BRAC': 'Behavior-Reg.\nActor-Critic',
	# 'CQL': 'Conservative\nQ-Learning',
}

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
	fig.set_size_inches(7.5, 2.5)

	means = {k: get_mean(v, exclude='ant') for k, v in means.items()}
	print(means)

	algs = ['BC', 'MBOP', 'BRAC', 'CQL', 'Decision\nTransformer', 'Trajectory\nTransformer']
	vals = [means[alg] for alg in algs]

	colors = [
		Colors.grey, Colors.gold,
		Colors.red, Colors.red, Colors.blue, Colors.blue
	]

	labels = [LABELS.get(alg, alg) for alg in algs]
	plt.bar(labels, vals, color=colors, edgecolor=Colors.gold, lw=0)
	plt.ylabel('Average normalized return', labelpad=15)
	# plt.title('Offline RL Results')

	legend_labels = ['Behavior Cloning', 'Trajectory Optimization', 'Temporal Difference', 'Sequence Modeling']
	colors = [Colors.grey, Colors.gold, Colors.red, Colors.blue]
	handles = [plt.Rectangle((0,0),1,1, color=color) for label, color in zip(legend_labels, colors)]
	plt.legend(handles, legend_labels, ncol=4,
		bbox_to_anchor=(1.07, -.18), fancybox=False, framealpha=0, shadow=False, columnspacing=1.5, handlelength=1.5)

	matplotlib.rcParams['hatch.linewidth'] = 7.5
	# ax.patches[-1].set_hatch('/')

	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)

	# plt.savefig('plotting/bar.pdf', bbox_inches='tight')
	plt.savefig('plotting/bar.png', bbox_inches='tight', dpi=500)
