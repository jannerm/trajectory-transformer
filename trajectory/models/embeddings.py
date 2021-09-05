import numpy as np
import torch
import torch.nn as nn
import pdb

def make_weights(N, weights):
	assert len(weights) % 2 == 1, f'Expected odd number of weights, got: {weights}'
	center = int((len(weights) - 1) / 2)

	tokens = np.zeros((N, N))
	for i in range(N):
		token = np.zeros(N)
		for j, w in enumerate(weights):
			ind = i + j - center
			ind = np.clip(ind, 0, N-1)
			token[ind] += w
		tokens[i] = token
	assert np.allclose(tokens.sum(axis=-1), 1)
	return tokens

def add_stop_token(tokens):
	N = len(tokens)
	## regular tokens put 0 probability on stop token
	pad = np.zeros((N, 1))
	tokens = np.concatenate([tokens, pad], axis=1)
	## stop token puts 1 probability on itself
	stop_weight = np.zeros((1, N+1))
	stop_weight[0,-1] = 1
	tokens = np.concatenate([tokens, stop_weight], axis=0)

	assert tokens.shape[0] == tokens.shape[1]
	assert np.allclose(tokens.sum(axis=-1), 1)
	return tokens

class SmoothEmbedding(nn.Module):

	def __init__(self, num_embeddings, embedding_dim, weights, stop_token=False):
		super().__init__()
		self.weights = make_weights(num_embeddings, weights)
		if stop_token:
			self.weights = add_stop_token(self.weights)
			num_embeddings += 1
		self.weights = torch.tensor(self.weights, dtype=torch.float, device='cuda:0')
		self.inds = torch.arange(0, num_embeddings, device='cuda:0')
		self._embeddings = nn.Embedding(num_embeddings, embedding_dim)

	def forward(self, x):
		'''
			x : [ batch_size x context ]
		'''
		## [ num_embeddings x embedding_dim ]
		embed = self._embeddings(self.inds)
		## [ batch_size x context x num_embeddings ]
		weights = self.weights[x]
		assert torch.allclose(weights.sum(-1), torch.ones(1, device=weights.device))

		# [ batch_size x context x embedding_dim ]
		weighted_embed = torch.einsum('btn,nd->btd', weights, embed)
		return weighted_embed


if __name__ == '__main__':

	x = torch.randint(0, 100, size=(5, 10,)).cuda()

	## test with weights
	embed = SmoothEmbedding(100, 32, weights=[0.15, 0.2, 0.3, 0.2, 0.15], stop_token=True)
	embed.cuda()
	out = embed(x)

	## test limiting case of regular embedding module
	embed_1 = SmoothEmbedding(100, 32, weights=[1.0], stop_token=True)
	embed_1.cuda()
	out_1 = embed_1(x)

	## reference
	out_0 = embed_1._embeddings(x)

	print(f'Same: {(out_0 == out_1).all().item()}')
