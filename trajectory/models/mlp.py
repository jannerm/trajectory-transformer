import torch
import torch.nn as nn
import pdb


def get_activation(params):
	if type(params) == dict:
		name = params['type']
		kwargs = params['kwargs']
	else:
		name = str(params)
		kwargs = {}
	return lambda: getattr(nn, name)(**kwargs)

def flatten(condition_dict):
	keys = sorted(condition_dict)
	vals = [condition_dict[key] for key in keys]
	condition = torch.cat(vals, dim=-1)
	return condition

class MLP(nn.Module):

	def __init__(self, input_dim, hidden_dims, output_dim, activation='GELU', output_activation='Identity', name='mlp', model_class=None):
		"""
			@TODO: clean up model instantiation from config so we don't have to pass in `model_class` to the model itself
		"""
		super(MLP, self).__init__()
		self.input_dim = input_dim
		self.name = name
		activation = get_activation(activation)
		output_activation = get_activation(output_activation)

		layers = []
		current = input_dim
		for dim in hidden_dims:
			linear = nn.Linear(current, dim)
			layers.append(linear)
			layers.append(activation())
			current = dim

		layers.append(nn.Linear(current, output_dim))
		layers.append(output_activation())

		self._layers = nn.Sequential(*layers)

	def forward(self, x):
		return self._layers(x)

	@property
	def num_parameters(self):
		parameters = filter(lambda p: p.requires_grad, self.parameters())
		return sum([p.numel() for p in parameters])
	
	def __repr__(self):
		return  '[ {} : {} parameters ] {}'.format(
			self.name, self.num_parameters,
			super().__repr__())

class FlattenMLP(MLP):

	def forward(self, *args):
		x = torch.cat(args, dim=-1)
		return super().forward(x)
