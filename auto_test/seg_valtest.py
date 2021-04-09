# CUDA_VISIBLE_DEVICES=0 python auto_test/seg_valtest.py\
# --cfg cfg_filepath --ckpt ckpt_dirpath

from subprocess import Popen
import argparse
import os 
import time

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str)
parser.add_argument('--ckpt', type=str)
args = parser.parse_args()

# start_epoch, end_epoch
l, r = 1, 24  # [l, r]

# cfg_path, ckpt_path
cfg_path = args.cfg  # f'configs/new10_contra/{cfg}.py'
ckpt_path = args.ckpt  # f'checkpoints/new10_contra/{ckpt}'
print('cfg_path:', cfg_path)
print('ckpt_path:', ckpt_path)

# create/append file
src_path = os.path.join(ckpt_path, 'source_test.txt')  # f'{ckpt}/source_test.txt'
tgt_path = os.path.join(ckpt_path, 'target_test.txt')  # f'{ckpt}/target_test.txt'
src_file = open(src_path, 'a')
tgt_file = open(tgt_path, 'a')

# test
for i in range(l, r+1):
    model_path = os.path.join(ckpt_path, f'epoch_{i}.pth')
    while not os.path.exists(model_path):
        print('sleeping...')
        time.sleep(1800)

    # test source_test
    start = time.time()
    cmd = ['python', './tools/seg_valtest.py',
           cfg_path, model_path,
           '--split', 'source_test']
    print(' '.join(cmd))
    proc = Popen(cmd, stdout=src_file)
    proc.wait()
    end = time.time()
    last = int(end - start)
    print(f'source_test epoch{i}.pth: {last // 60}min {last % 60}s')

    # test target_test
    start = time.time()
    cmd = ['python', './tools/seg_valtest.py',
           cfg_path, model_path,
           '--split', 'target_test']
    print(' '.join(cmd))
    proc = Popen(cmd, stdout=tgt_file)
    proc.wait()
    end = time.time()
    last = int(end - start)
    print(f'target_test epoch{i}.pth: {last // 60}min {last % 60}s')

# close file
src_file.close()
tgt_file.close()
