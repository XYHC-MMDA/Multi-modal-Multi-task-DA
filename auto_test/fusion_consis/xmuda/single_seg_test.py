# CUDA_VISIBLE_DEVICES=0 python auto_test/fusion_consis/xmuda/auto_mmda_test.py --cfg baseline2_usa [--ckpt baseline2_usa]

import argparse
import os 
import time

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str)
parser.add_argument('--ckpt', default='')  # optional; defined when cfg != ckpt
args = parser.parse_args()
cfg = args.cfg
if args.ckpt != '':
    ckpt = args.ckpt
else:
    ckpt = cfg

# start_epoch, end_epoch
l, r = 1, 24  # [l, r]


cfg_path = f'configs/fusion_consis/xmuda/{cfg}.py'
ckpt_path = f'checkpoints/fusion_consis/xmuda/{ckpt}'
filename = f'checkpoints/fusion_consis/xmuda/{ckpt}/result.txt'
print('cfg_path:', cfg_path)
print('ckpt_path:', ckpt_path)
f = open(filename, 'a')
for i in range(l, r+1):
    model_path = os.path.join(ckpt_path, f'epoch_{i}.pth')
    while not os.path.exists(model_path):
        print('sleeping...')
        time.sleep(1800)

    # test target
    start = time.time()
    cmd = ['python', './tools/single_seg_test.py',
           cfg_path, model_path]
    print(' '.join(cmd))
    proc = Popen(cmd, stdout=f)
    # print(f'epoch_{i}.pth finished')
    proc.wait()
    end = time.time()
    last = int(end - start)
    print(f'epoch{i}.pth: {last // 60}min {last % 60}s')
f.close()
