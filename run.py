#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import glob
import time
import shutil
import subprocess

from flufl.lock import Lock
from datetime import datetime, timedelta
from inspect import currentframe, getframeinfo

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--logdir', type=str, default='',
                        help="where to save")
parser.add_argument('--mode', type=str, default='run',
                        help='run|monitor')
parser.add_argument('--gpu', type=int, default=-1,
                        help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--N', type=int, default=1,
                        help='the number of script (N<=0 means run-all)')
parser.add_argument('--time_from', type=str, default=None,
                        help='starting time of run.py')
parser.add_argument('--time_to', type=str, default=None,
                        help='ending time of run.py')
parser.add_argument('--die_if_fail', action='store_const',
                        const=True, default=False,
                        help="Stop all jobs if one of them is failed.")
parser.add_argument('--least_memory', type=int, default=0,
                        help='the least gpu memory size (MB) which your program requires ')
parser.add_argument('--jobdir', type=str, default='jobs',
                        help="where to place the jobs")


### GPU Checker (copy from pynvidia function)

def filter_alphanumeric(string):
    return ''.join([c for c in string if c.isnumeric()])

def read_csv(string, return_header=True, remove_nonalpha=True):
    if isinstance(string, bytes):
        string = string.decode('utf-8')
    rows = string.split('\n')
    rows = [row.split(',') for row in rows if row != '']
    if remove_nonalpha:
        rows = [[filter_alphanumeric(entry) for entry in row] for row in rows]
    else:
        rows = [[entry.strip() for entry in row] for row in rows]
    rows = list(map(tuple, rows))
    if not return_header:
        rows = rows[1:]
    return rows

def get_gpu_memory(gpu_id):
    output = subprocess.check_output(
            "nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total --format=csv", shell=True)
    csv = read_csv(output, return_header=False)

    if gpu_id >= len(csv) or gpu_id < 0:
        raise ValueError('Invalid GPU_ID, GPU_ID should be less than {}, but input is {}'.format(len(csv), gpu_id))

    _, used_mem, free_mem, total_mem = map(int, csv[gpu_id])
    return used_mem, free_mem, total_mem

def get_todo_script(todo_dir, latest_first=False):
    files = glob.glob(os.path.join(todo_dir, '*.sh'))
    files.sort(key=os.path.getmtime, reverse=latest_first)
    files = [os.path.basename(_file) for _file in files]
    return files

def get_now():
    return time.asctime()

def set_lock(check_lock_file):
    check_lock = None
    if os.name == "posix":
        check_lock = Lock(check_lock_file)
        check_lock.lifetime = timedelta(hours=1)
        frameinfo = getframeinfo(currentframe())
        print("-- {}/{}: waiting to obtain lock --".format(
            frameinfo.filename, frameinfo.lineno))
        check_lock.lock()
        print(">> obtained lock for posix system<<")
    elif os.name == "nt":
        import filelock
        check_lock = filelock.FileLock(check_lock_file)
        check_lock.timeout = 100 # 100s
        check_lock.acquire()
        if check_lock.is_locked:
            print(">> obtained lock for windows system <<")
    else:
        print("Unknown operating system, lock unavailable")
    return check_lock

def set_unlock(check_lock):
    if os.name == "posix":
        check_lock.unlock()
        print("-- free lock --")
    elif os.name == "nt":
        check_lock.release()
        print("-- free lock --")
    else:
        pass

def get_server_info():
    hostname = os.uname()[1]
    gpu_id = os.getenv('CUDA_VISIBLE_DEVICES')
    if gpu_id is None:
        gpu_id = -1
    return hostname, int(gpu_id)

def add_history(HISTORY, job, directory, curr_time, status):
    # Lock should be done outside this function
    hostname, gpu_id = get_server_info()

    with open(HISTORY, 'a') as f:
        f.write('{} {} {} {} {} {}\n'.format(curr_time, job, directory, status, hostname, gpu_id))

def update_history(HISTORY, HISTORY_LOCK, job, prev_dir, prev_time, next_dir, next_time, status):

    hostname, gpu_id = get_server_info()

    check_lock = set_lock(HISTORY_LOCK)
    history = open(HISTORY, 'r').readlines()
    history = [line.rstrip('\n') for line in history]

    find_idx = -1
    find_line = '{} {} {}'.format(prev_time, job, prev_dir)
    for i in range(len(history)):
        if find_line in history[i]:
            find_idx = i
            break
    if find_idx >= 0:
        # print('FIND IDX {}'.format(find_idx))
        del history[find_idx]
        history.append('{} {} {} {} {} {}'.format(next_time, job, next_dir, status, hostname, gpu_id))
    else:
        print('Not find {}'.format(find_line))

    with open(HISTORY, 'w') as f:
        for line in history:
            f.write('{}\n'.format(line))
    set_unlock(check_lock)

def run_jobs(config):

    # HISTORY = 'jobs/HISTORY'
    # HISTORY_LOCK = HISTORY + '.lock'
    # TODO_DIR = 'jobs/todo'
    # QUEUE_DIR = 'jobs/queue'
    # DONE_DIR = 'jobs/done'
    # FAIL_DIR = 'jobs/fail'
    HISTORY = os.path.join(config.jobdir, 'HISTORY')
    HISTORY_LOCK = HISTORY + '.lock'
    TODO_DIR = os.path.join(config.jobdir, 'todo')
    QUEUE_DIR = os.path.join(config.jobdir, 'queue')
    DONE_DIR = os.path.join(config.jobdir, 'done')
    FAIL_DIR = os.path.join(config.jobdir, 'fail')

    # memory check
    if config.least_memory > 0 and config.gpu >= 0:
        used_mem, free_mem, total_mem = get_gpu_memory(config.gpu)
        if config.least_memory > free_mem:
            print('There are not enough gpu memory, only {}/{}MB are available on GPU#{}. all jobs are canceled.'.format(free_mem, total_mem, config.gpu))
            return -1

    sh_files = get_todo_script(TODO_DIR)
    if len(sh_files) == 0:
        print('There remains no jobs...quit.')
        return -1
    if config.N > 0:
        N = min(config.N, len(sh_files))
    else:
        N = len(sh_files)

    if N == 0:
        print('There is no jobs at {}'.format(TODO_DIR))
        return -1

    jobfiles = sh_files[:N]
    times = [None] * N

    # move from 'todo' to 'queue', locking will be continued until all scripts move to queue directory.
    check_lock = set_lock(HISTORY_LOCK)
    for i, job in enumerate(jobfiles):
        print('#{} {}'.format(i, job))
        shutil.move(os.path.join(TODO_DIR, job), os.path.join(QUEUE_DIR, job))
        times[i] = get_now()
        add_history(HISTORY, job, QUEUE_DIR, times[i], 'READY')
    set_unlock(check_lock)

    print('Move {}/{} jobs to {}'.format(N, len(sh_files), QUEUE_DIR))

    success_jobs = []
    fail_jobs = []

    for i, job in enumerate(jobfiles):
        try:
            print('[{}] RUN {}/{} {}'.format(get_now(), i+1, N, job))
            update_history(HISTORY, HISTORY_LOCK, job, QUEUE_DIR, times[i], QUEUE_DIR, times[i], 'RUNNING')

            p = subprocess.Popen(['sh', os.path.join(QUEUE_DIR, job)],
                                    stdin=subprocess.PIPE,
                                    stdout=sys.stdout,
                                    stderr=subprocess.PIPE,
                                    shell=False)
            if p.wait() == 0:
                shutil.move(os.path.join(QUEUE_DIR, job), os.path.join(DONE_DIR, job))
                new_time = get_now()
                update_history(HISTORY, HISTORY_LOCK, job, QUEUE_DIR, times[i], DONE_DIR, new_time, 'DONE')
                print('[{}] job {} has done successfully.'.format(new_time, job))
                success_jobs.append(job)
            else:
                # Dump log file
                with open(os.path.join(FAIL_DIR, job+'.log'), 'w') as f:
                    print('write logs..')
                    for line in p.stderr.readlines():
                        line = line.decode('utf-8')
                        f.write(line)
                        print(line, end='')
                raise Exception('Error happens')
        except:
            shutil.move(os.path.join(QUEUE_DIR, job), os.path.join(FAIL_DIR, job))

            new_time = get_now()
            update_history(HISTORY, HISTORY_LOCK, job, QUEUE_DIR, times[i], FAIL_DIR, new_time, 'FAIL')
            print('[FAIL!! {}] job {} was failed.'.format(new_time, job))
            fail_jobs.append(job)

            if config.die_if_fail:
                raise RuntimeError('Die because a job failed.')

    print('[{}] {} jobs are finished.'.format(get_now(), N))
    print('Success: {} jobs, {}'.format(len(success_jobs), success_jobs))
    print('Fail: {} jobs, {}'.format(len(fail_jobs), fail_jobs))
    return 1

def get_scheduled_time(hhmm = None):
    if hhmm is None:
        return None
    times = hhmm.split(':')

    if len(times) == 1:
        hours = int(times[0])
        minutes = 0
    elif len(times) == 2:
        hours = int(times[0])
        minutes = int(times[1])
    else:
        raise ValueError('Invalid format of time hh:mm, {}\n'.format(config.time_from))

    now = datetime.now()
    schd_time = datetime(now.year, now.month, now.day, hours, minutes)

    if (schd_time - now).days < 0:
        schd_time = datetime(now.year, now.month, now.day+1, hours, minutes)

    return schd_time

def run_debug(config):
    print('run debug')
    try:
        # subprocess.check_call(['sh', 'job-sample_run.sh'])
        p = subprocess.Popen(['sh', 'job-sample_run.sh'],
                     stdin=subprocess.PIPE,
                     stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE,
                     shell=False)
        print(p.wait())

        if p.wait() == 0:
            print('run successfully')
        else:
            print('Error happens --> dump log.txt')
            with open('log.txt', 'w') as f:
                for line in p.stdout.readlines():
                    line = line.decode('utf-8')
                    f.write(line)
                for line in p.stderr.readlines():
                    line = line.decode('utf-8')
                    print(line, end='')
                    f.write(line)
    except:
        pass

if __name__ == '__main__':
    config, unparsed = parser.parse_known_args()
    if len(unparsed) > 0:
        print('[Error] unparsed args: {}'.format(unparsed))
        exit(1)
        
    if len(config.logdir) > 0:
        os.environ["PROJ_RES_DIR"] = config.logdir
        print('Change PROJ_RES_DIR to {}'.format(os.getenv('PROJ_RES_DIR')))
    if config.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
        print('Switch GPU device to {}'.format(os.getenv('CUDA_VISIBLE_DEVICES')))
    else:
        hostname, gpu_id = get_server_info()
        config.gpu = gpu_id

    time_from = get_scheduled_time(config.time_from)
    time_to = get_scheduled_time(config.time_to)

    if time_from is not None:
        while (True):
            now = datetime.now()
            if (time_from-now).days < 0:
                print("It's time to run jobs. Let's start !!!")
                break
            else:
                hostname, gpu_id = get_server_info()
                print('[{} GPU#{}]It has not been the specified running time ({}) so I will sleep 300s.'.format(hostname, gpu_id, time_from))
                time.sleep(300)

    stop_file = '{}_stop'.format(os.uname()[1])
    if os.path.exists(stop_file):
    	print('[Warning] Did you forget removing {}?'.format(stop_file))

    if config.mode == 'run':
        run_jobs(config)
    elif config.mode == 'monitor':
        while True:
            print('-----<<< Run as monitor-mode >>>-----')
            ret = run_jobs(config) # loop until no-jobs
            if ret < 0:
            	break
            if os.path.exists(stop_file):
            	print('Detect {} and running loop is finished'.format(stop_file))
            	break # hostname_stop
            if time_to is not None:
                now = datetime.now()
                if (time_to-now).days < 0:
                    print("Time's up for today. Quit all jobs.")
                    break

    elif config.mode == 'debug':
        run_debug(config)
    elif config.mode == 'clear':
        pass
