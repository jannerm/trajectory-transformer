import os
import pdb

from doodad.wrappers.easy_launch import sweep_function, save_doodad_config

codepath = '/home/code'
script = 'scripts/plan.py'

def remote_fn(doodad_config, variant):
    ## get suffix range to allow running multiple trials per job
    n_suffices = variant['n_suffices']
    suffix_start = variant['suffix_start']
    del variant['n_suffices']
    del variant['suffix_start']

    kwarg_string = ' '.join([
        f'--{k} {v}' for k, v in variant.items()
    ])
    print(kwarg_string)

    d4rl_path = os.path.join(doodad_config.output_directory, 'datasets/')
    os.system(f'ls -a {codepath}')
    os.system(f'mv {codepath}/git {codepath}/.git')

    for suffix in range(suffix_start, suffix_start + n_suffices):
        os.system(
            f'''export PYTHONPATH=$PYTHONPATH:{codepath} && '''
            f'''export CUDA_VISIBLE_DEVICES=0 && '''
            f'''export D4RL_DATASET_DIR={d4rl_path} && '''
            f'''python {os.path.join(codepath, script)} '''
            f'''--suffix {suffix} '''
            f'''{kwarg_string}'''

        )

    save_doodad_config(doodad_config)

if __name__ == "__main__":

    environments = ['ant']
    buffers = ['medium-expert-v2', 'medium-v2', 'medium-replay-v2', 'random-v2']
    datasets = [f'{env}-{buf}' for env in environments for buf in buffers]

    azure_logpath = 'defaults/'

    params_to_sweep = {
        'dataset': datasets,
        'horizon': [15],
    }

    default_params = {
        'logbase': os.path.join('/doodad_tmp', azure_logpath, 'logs'),
        'prefix': 'plans/azure/',
        'verbose': False,
        'suffix_start': 0,
        'n_suffices': 3,
    }

    print(params_to_sweep)
    print(default_params)

    sweep_function(
        remote_fn,
        params_to_sweep,
        default_params=default_params,
        config_path=os.path.abspath('azure/config.py'),
        log_path=azure_logpath,
        azure_region='westus2',
        # gpu_model='nvidia-tesla-v100',
        gpu_model='nvidia-tesla-t4',
        filter_dir=['logs', 'bin', 'mount'],
        use_gpu=True,
    )
