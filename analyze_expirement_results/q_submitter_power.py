# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 10:16:41 2017

@author: Oren
"""

import time
import os
import subprocess
import stat
from pathlib import Path
from time import time, sleep
import traceback
from sys import argv
from datetime import timedelta

from utils import get_logger, str_to_bool

JOB_EXTENSION = '.slurm'
JOB_SUBMITTER = 'sbatch'
MEMORY_SUFFIX = 'G'
CHECK_JOB_DETAILS_COMMAND = 'scontrol show job'

JOB_NAME_ENVIRONMENT_VARIABLE = 'SLURM_JOB_NAME'
JOB_ID_ENVIRONMENT_VARIABLE = 'SLURM_JOB_ID'
JOB_FILES_DEBUG_MODE = False
JOB_WALL_TIME_KEY = 'RunTime='
JOB_CPUS_KEY = 'NumCPUs='

Q_SUBMITTER_ADD_SSH_PREFIX = False
LOGIN_NODE = 'c-001'

CONDA_INSTALLATION_DIR = Path('/home/ai_center/ai_users/yairshimony/miniconda')
CONDA_ENVIRONMENT_DIR = Path('/home/ai_center/ai_users/yairshimony/miniconda/envs/ships')
SLURM_ACCOUNT = 'gpu-research'
SLURM_PARTITION = 'cpu-killable'

JOB_DONE_FILE_SUFFIX = '.done'
CHECK_JOB_DONE_INTERVAL_SECONDS = 10


def add_slurm_header(sbatch_file_content, queue_name, tmp_dir, job_name, CPUs, account_name, memory, time_in_hours,
                     node_name):
    sbatch_file_content += f'#SBATCH --job-name={job_name}\n'
    sbatch_file_content += f'#SBATCH --account={account_name}\n'
    sbatch_file_content += f'#SBATCH --partition={queue_name}\n'
    sbatch_file_content += f'#SBATCH --ntasks=1\n'
    sbatch_file_content += f'#SBATCH --cpus-per-task={CPUs}\n'
    sbatch_file_content += f'#SBATCH --mem={memory}{MEMORY_SUFFIX}\n'
    if time_in_hours:
        sbatch_file_content += f'#SBATCH --time={time_in_hours}:00:00\n'
    if node_name:
        sbatch_file_content += f'#SBATCH --nodelist={node_name}\n'
    sbatch_file_content += f'#SBATCH --output={tmp_dir}/{job_name}_%j.out\n'
    sbatch_file_content += f'#SBATCH --error={tmp_dir}/{job_name}_%j.err\n'

    sbatch_file_content += f'echo Job ID: ${JOB_ID_ENVIRONMENT_VARIABLE}\n'
    sbatch_file_content += f'echo Running on nodes: $SLURM_JOB_NODELIST\n'
    sbatch_file_content += f'echo Allocated CPUs: $SLURM_JOB_CPUS_PER_NODE\n'
    sbatch_file_content += f'echo Memory per node: $SLURM_MEM_PER_NODE MB\n'
    sbatch_file_content += f'echo Job name: $SLURM_JOB_NAME\n'

    return sbatch_file_content


def generate_job_file(logger, queue_name, tmp_dir, cmds_path, job_name, job_path, CPUs, account_name, memory,
                      time_in_hours, node_name):
    """compose the job file content and fetches it"""
    job_file_content = f'#!/bin/bash {"-x" if JOB_FILES_DEBUG_MODE else ""}\n'  # 'old bash: #!/bin/tcsh -x\n'
    job_file_content = add_slurm_header(job_file_content, queue_name, tmp_dir, job_name, CPUs, account_name, memory,
                                        time_in_hours, node_name)
    job_file_content += f'{cmds_path}\n'

    # log the runtime of the job
    job_log_file_path = tmp_dir / f'$(echo ${JOB_NAME_ENVIRONMENT_VARIABLE})_$(echo ${JOB_ID_ENVIRONMENT_VARIABLE})_log.txt'
    job_file_content += f'{CHECK_JOB_DETAILS_COMMAND} ${JOB_ID_ENVIRONMENT_VARIABLE} | grep -m 1 "{JOB_WALL_TIME_KEY}" >> {job_log_file_path}\n'
    job_file_content += f'{CHECK_JOB_DETAILS_COMMAND} ${JOB_ID_ENVIRONMENT_VARIABLE} | grep -m 1 "{JOB_CPUS_KEY}" >> {job_log_file_path}\n'

    with open(job_path, 'w') as job_fp:  # write the job
        job_fp.write(job_file_content)
    subprocess.call(['chmod', '+x', job_path])  # set execution permissions


def submit_cmds_from_file_to_q(logger, job_name, cmds_path, tmp_dir, queue_name, CPUs, account_name, memory,
                               time_in_hours, node_name, additional_params=''):
    job_path = tmp_dir / f'{job_name}{JOB_EXTENSION}'  # path to job
    generate_job_file(logger, queue_name, tmp_dir, cmds_path, job_name, job_path, CPUs, account_name, memory,
                      time_in_hours, node_name)

    # execute the job
    # queue_name may contain more arguments, thus the string of the cmd is generated and raw cmd is called

    if Q_SUBMITTER_ADD_SSH_PREFIX:
        terminal_cmd = f'ssh {LOGIN_NODE} "{JOB_SUBMITTER} {job_path} {additional_params}"'  # FIX by danny 5-1-2023
    else:
        terminal_cmd = f'{JOB_SUBMITTER} {job_path} {additional_params}'

    job_submitted_successfully = False
    try_index = 1
    while not job_submitted_successfully:
        try:
            logger.info(f'Submitting: {terminal_cmd}')
            subprocess.run(terminal_cmd, shell=True, check=True, capture_output=True, text=True)
            logger.info(f"Job {job_path} submitted successfully")
            job_submitted_successfully = True
        except subprocess.CalledProcessError as e:
            logger.error(f"Job submission of {job_path} failed (try {try_index}): {e.stderr}")
            try_index += 1
            if try_index >= 100:
                logger.error(f"Job submission of {job_path} failed too many times. Exiting")
                raise e
            sleep(1)


def submit_mini_batch(logger, script_path, mini_batch_parameters_list, logs_dir, job_name, error_file_path, num_of_cpus=1,
                      memory=None, time_in_hours=None, environment_variables_to_change_before_script: dict = None):
    """
    :param script_path:
    :param mini_batch_parameters_list: a list of lists. each sublist corresponds to a single command and contain its parameters
    :param logs_dir:
    :param job_name:
    :param num_of_cpus:
    :return: an example command to debug on the shell
    """

    shell_cmds_as_str = ''

    # shell_cmds_as_str += f'source ~/.bashrc{new_line_delimiter}'
    conda_sh_path = CONDA_INSTALLATION_DIR / 'etc' / 'profile.d' / 'conda.sh'
    shell_cmds_as_str += f'source {conda_sh_path}\n'
    shell_cmds_as_str += f'conda activate {CONDA_ENVIRONMENT_DIR}\n'
    shell_cmds_as_str += f'export PATH=$CONDA_PREFIX/bin:$PATH\n'

    # PREPARING RELEVANT COMMANDS
    if environment_variables_to_change_before_script:
        for key, value in environment_variables_to_change_before_script.items():
            shell_cmds_as_str += f'export {key}={value}\n'

    for params in mini_batch_parameters_list:
        shell_cmds_as_str += ' '.join(
            ['python', str(script_path), *[str(param) for param in params],
             f'--logs_dir {logs_dir}', f'--error_file_path {error_file_path}', f'--job_name {job_name}',
             f'--cpus {num_of_cpus}']) + '\n'

    # GENERATE DONE FILE
    shell_cmds_as_str += f'touch {logs_dir / (job_name + JOB_DONE_FILE_SUFFIX)}\n'

    if memory is None:
        memory = 16

    # WRITING CMDS FILE
    cmds_path = logs_dir / f'{job_name}.sh'
    with open(cmds_path, 'w') as f:
        f.write(shell_cmds_as_str)

    # Add execution permissions to cmds_path
    current_permissions = cmds_path.stat().st_mode
    cmds_path.chmod(current_permissions | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    submit_cmds_from_file_to_q(logger, job_name, cmds_path, logs_dir, SLURM_PARTITION, str(num_of_cpus),
                               SLURM_ACCOUNT, memory, time_in_hours, None)


def run_step(args, step_method, *step_args):
    logger = get_job_logger(args.logs_dir, args.job_name)
    logger.info(f'Starting command is: {" ".join(argv)}')

    try:
        step_method(logger, *step_args)
    except subprocess.CalledProcessError as e:
        error_message = f'Error in function "{step_method.__name__}" in command: "{e.cmd}": {e.stderr}'
        logger.exception(error_message)
        with open(args.error_file_path, 'a+') as f:
            f.write(error_message)
    except Exception as e:
        logger.exception(f'Error in function "{step_method.__name__}"')
        with open(args.error_file_path, 'a+') as f:
            traceback.print_exc(file=f)


def get_job_logger(log_file_dir, job_name, verbose=False):
    job_id = os.environ.get(JOB_ID_ENVIRONMENT_VARIABLE, '')
    logger = get_logger(log_file_dir / f'{job_name}_{job_id}_log.txt', 'main', verbose)

    return logger


def wait_for_results(logger, script_name, path, num_of_expected_results, error_file_path):
    """waits until path contains num_of_expected_results .done files"""
    start = time()
    logger.info(f'Waiting for {script_name}... Continues when {num_of_expected_results} results will be in: {path}')

    if num_of_expected_results == 0:
        raise ValueError('Number of expected results is 0! Something went wrong in the previous analysis steps.')

    total_time = 0
    i = 0
    current_num_of_results = 0
    while num_of_expected_results > current_num_of_results:
        assert not error_file_path.exists()

        current_num_of_results = sum(1 for _ in path.glob(f'*{JOB_DONE_FILE_SUFFIX}'))
        jobs_left = num_of_expected_results - current_num_of_results
        sleep(CHECK_JOB_DONE_INTERVAL_SECONDS)
        total_time += CHECK_JOB_DONE_INTERVAL_SECONDS
        i += 1
        if i % 5 == 0:  # print status every 5 cycles of $time_to_wait
            logger.info(
                f'\t{timedelta(seconds=total_time)} have passed since started waiting ('
                f'{num_of_expected_results} - {current_num_of_results} = {jobs_left} more files are still missing)')

    end = time()
    total_time_waited = timedelta(seconds=int(end - start))
    logger.info(f'Done waiting for: {script_name} (took {total_time_waited}).')

    assert not error_file_path.exists()


def add_default_step_args(args_parser):
    args_parser.add_argument('--logs_dir', type=Path, help='path to tmp dir to write logs to')
    args_parser.add_argument('--error_file_path', type=Path, help='path to error file')
    args_parser.add_argument('--job_name', help='job name')
    args_parser.add_argument('--cpus', default=1, type=int)
