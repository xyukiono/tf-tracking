#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import time
import hashlib
import numpy as np
import glob
from hyper_params import HyperParamsBase, ParamGenerator

def is_cc(): # check server is compute canada
    hostname = os.uname()[1]
    is_cc = 'gra' in hostname or 'cedar' in hostname
    
    return is_cc 

class HyperParams(HyperParamsBase):
    def __init__(self):
        super(HyperParams, self).__init__()
        self.script_name = 'imagenet_train.py'
        self.sub_dir = 'test' # results are dumped in res_root_dir/sub_dir/...
        if is_cc():
            res_dir = '/scratch/trulls/yuki.ono/results'
        else:
            res_dir = os.getenv("PROJ_RES_DIR")
            if res_dir is None or res_dir == "":
                res_dir = os.path.join(os.getcwd(), 'results')
        self.res_root_dir = os.path.join(res_dir, 'practice/CFNet')

        ######################
        # Parameter catalogs
        ######################

        # Train
        self.max_itr = 1000000
        self.batch_size = 32
        self.optim_method = 'adam'
        self.lr = '1e-3'
        self.lr_decay = False
        self.weight_decay = 0
        # Dataset
        if is_cc():
            dataset_dir = '/scratch/trulls/yuki.ono/datasets'
        else:
            dataset_dir = '/cvlabdata1/home/ono/datasets'
        self.imagenet_dir = os.path.join(dataset_dir, 'imagenet')

        # Network
        self.model = 'custom_vgg' # custom_vgg | custom_alexnet

ROOT_JOB = 'jobs'
TODO_DIR = '{}/todo'.format(ROOT_JOB)
QUEUE_DIR = '{}/queue'.format(ROOT_JOB)
DONE_DIR = '{}/done'.format(ROOT_JOB)
FAIL_DIR = '{}/fail'.format(ROOT_JOB)

def check_job_pool():
    if not os.path.exists(TODO_DIR):
        os.makedirs(TODO_DIR)
    if not os.path.exists(QUEUE_DIR):
        os.makedirs(QUEUE_DIR)
    if not os.path.exists(DONE_DIR):
        os.makedirs(DONE_DIR)
    if not os.path.exists(FAIL_DIR):
        os.makedirs(FAIL_DIR)

def get_command(params, log_dir):

    CMD = params.script_name
    if not is_cc():
        CMD += ' --clear_logs'
    # Train
    CMD += ' --log_dir={}'.format(log_dir)
    CMD += ' --max_itr={}'.format(params.max_itr)
    CMD += ' --batch_size={}'.format(params.batch_size)
    CMD += ' --optim_method={}'.format(params.optim_method)
    CMD += ' --lr={}'.format(params.lr)
    CMD += ' --lr_decay={}'.format(params.lr_decay)
    CMD += ' --weight_decay={}'.format(params.weight_decay)

    # Dataset
    CMD += ' --imagenet_dir={}'.format(params.imagenet_dir)

    # Network
    CMD += ' --model={}'.format(params.model)

    return CMD

def get_log_dir(params):

    lr_str = 'lr-dcy' if params.lr_decay else 'lr-{}'.format(params.lr)

    log_dir = os.path.join(params.res_root_dir,
                params.sub_dir,
                'bs-{}'.format(params.batch_size),
                '{}-{}'.format(params.optim_method, lr_str),
                'wd-{}'.format(params.weight_decay),
                '{}'.format(params.model),
        )

    return log_dir

def write_shell_script(command, memo=None, params=None, log_dir=None):
    hash_str = hashlib.sha256(command.encode('utf-8')).hexdigest()[:16]
    job_name = 'job-{}.sh'.format(hash_str)
    job_file = os.path.join(TODO_DIR, job_name)

    PRESET = 'OMP_NUM_THREAD=4 python'

    with open(job_file, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write('# [TIME] {}\n'.format(time.asctime()))
        if memo is not None and len(memo) > 0:
            f.write('# [MEMO] {}\n'.format(memo))
        if params is not None and len(params) > 0:
            f.write('# [PARAM] {}\n'.format(params))
        if log_dir is not None and len(log_dir) > 0:
            f.write('# [LOGDIR] {}\n'.format(log_dir))

        if is_cc():
            f.write("#SBATCH --account=def-kyi\n")
            f.write("#SBATCH --output=/scratch/trulls/yuki.ono/outputs/%x-%j.out\n")
            f.write("source /scratch/trulls/venvs/tflift/bin/activate\n")
            f.write("cd {}\n".format(os.getcwd()))
        f.write("CMD='{} {}'\n".format(PRESET, command))
        f.write('echo $CMD\n'.format(PRESET, command))
        f.write('eval $CMD\n'.format(PRESET, command))
    print('Create {}'.format(job_file))


if __name__ == '__main__':

    from utils.argparse_utils import *
    parser = get_parser()
    parser.add_argument('--dry_run', action='store_const',
                        const=True, default=False,
                        help="Just print the result")
    config, unparsed = get_config(parser)
    if len(unparsed) > 0:
        raise ValueError('Warning: miss identify argument ?? unparsed={}\n'.format(unparsed))

    exist_strategy = '' # skip, remove, recreate, overwrite
    memo = None
    pg = ParamGenerator()

    # Basic test
    memo = 'Imagenet training'
    pg.add_params('sub_dir', '180715-imagenet')
    # pg.add_params('optim_method', ['adam'])
    # pg.add_params('lr', ['1e-3'])
    # # pg.add_params('weight_decay', ['0', '5e-4'])
    # pg.add_params('model', ['custom_vgg', 'custom_alexnet'])

    pg.add_params('optim_method', ['momentum'])
    pg.add_params('lr_decay', [True])
    pg.add_params('weight_decay', ['5e-4'])
    pg.add_params('model', ['custom_vgg', 'custom_alexnet'])


    all_params = pg.generate(base_params=HyperParams())

    if not config.dry_run:
        check_job_pool()

    num_new_jobs = 0

    log_dir_set = set()

    for idx, params in enumerate(all_params):

        log_dir = get_log_dir(params)
        log_dir_set.add(log_dir)
        # print(os.path.join(params.res_root_dir, log_dir))
        if os.path.exists(os.path.join(params.res_root_dir, log_dir)):
            print('logdir {} has already existed ! strategy={}'.format(log_dir, exist_strategy))
            if exist_strategy == 'skip':
                continue
        cmd = get_command(params, log_dir)
        print(idx, log_dir)
        if not config.dry_run:
            write_shell_script(cmd, memo=memo, params=params.param_str, log_dir=log_dir)
        num_new_jobs += 1

    print('Add {} {}/{}'.format(num_new_jobs, ROOT_JOB, len(all_params)))
    total_num_jobs = len(glob.glob(os.path.join(TODO_DIR, 'job*sh')))
    if config.dry_run:
        total_num_jobs += num_new_jobs
    print('Total #jobs todo = {}'.format(total_num_jobs))    

    if len(log_dir_set) != len(all_params):
        print('')
        print('[WARNING] log_dir may be overlapped !! expected={} but actual={}'.format(len(all_params), len(log_dir_set)))

