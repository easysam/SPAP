import sys
sys.path.append('..')
import time
import os
import logging
from preprocess import load_data


def build_log(name, path, mode='a', need_console=False):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fh = logging.FileHandler(path, mode=mode)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if need_console:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)
    return logger

def log_result(header, params, filename):
    """
    log result, header == params + (time) + metric
    """

    # add time to metric for log time
    params['time'] = time.strftime("%m-%d %H:%M", time.localtime())

    left_part, right_part = '{:>', 's}'
    rec_header = ''
    rec_hyper_params = ''
    for i, (key, value) in enumerate(header.items()):
        if i == 0:
            rec_header = left_part + str(value) + right_part
            rec_hyper_params = left_part + str(value) + right_part
        else:
            rec_header = rec_header + ' ' + left_part + str(value) + right_part
            rec_hyper_params = rec_hyper_params + ' ' + left_part + str(value) + right_part

        rec_header = rec_header.format(key)
        rec_hyper_params = rec_hyper_params.format(str(params[key]))

    with open(filename, mode='a') as f:
        if os.path.getsize(filename) == 0:
            f.write(rec_header + '\n')
        f.write(rec_hyper_params + '\n')

if __name__ == '__main__':
    pass
