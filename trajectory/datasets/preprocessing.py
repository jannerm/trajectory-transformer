import numpy as np

def kitchen_preprocess_fn(observations):
    ## keep first 30 dimensions of 60-dimension observations
    keep = observations[:, :30]
    remove = observations[:, 30:]
    assert (remove.max(0) == remove.min(0)).all(), 'removing important state information'
    return keep

def vmap(fn):

    def _fn(inputs):
        if inputs.ndim == 1:
            inputs = inputs[None]
            return_1d = True
        else:
            return_1d = False

        outputs = fn(inputs)

        if return_1d:
            return outputs.squeeze(0)
        else:
            return outputs

    return _fn

def preprocess_dataset(preprocess_fn):

    def _fn(dataset):
        for key in ['observations', 'next_observations']:
            dataset[key] = preprocess_fn(dataset[key])
        return dataset

    return _fn

preprocess_functions = {
    'kitchen-complete-v0': vmap(kitchen_preprocess_fn),
    'kitchen-mixed-v0': vmap(kitchen_preprocess_fn),
    'kitchen-partial-v0': vmap(kitchen_preprocess_fn),
}

dataset_preprocess_functions = {
    k: preprocess_dataset(fn) for k, fn in preprocess_functions.items()
}
