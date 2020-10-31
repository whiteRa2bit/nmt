import subprocess

from loguru import logger


def _get_gpu_info():
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=utilization.gpu,memory.free',
            '--format=csv,noheader'
        ]).decode("utf-8")
    # Convert lines into a dictionary
    result = result.strip().split('\n')
    gpus_info = [info.split(',') for info in result]
    filter_foo = lambda x: int(''.join(filter(str.isdigit, x)))
    gpus_info = [list(map(filter_foo, info)) for info in gpus_info]
    return gpus_info


def get_gpu_id(min_space=1000):
    gpus_info = _get_gpu_info()
    gpus_info = [[idx] + info for idx, info in enumerate(gpus_info)]
    gpus_info = [info for info in gpus_info if info[2] > min_space]
    gpus_info.sort(key=lambda x: -x[2])  # The first one will have the most free spcae
    gpus_info.sort(key=lambda x: x[1])  # The first one will have the least utilization
    gpu_id = gpus_info[0][0]
    logger.info(f"GPU {gpu_id} was scheduled")
    return gpu_id
