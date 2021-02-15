from subprocess import Popen
import argparse
import os 
import time 

# variants
cfg = 'train_usa_disc_01'
ckpt = 'train_usa_disc_01'
l, r = 1, 16  # [l, r]


filename = f'checkpoints/single_seg/sep_disc/{ckpt}/result.txt'
f = open(filename, 'a')
for i in range(l, r+1):
    start = time.time()
    proc = Popen(['python', './tools/single_seg_test.py',
                  f'configs/single_seg/sep_disc/{cfg}.py',
                  f'checkpoints/single_seg/sep_disc/{ckpt}/epoch_{i}.pth'],
                  stdout=f)
    # print(f'epoch_{i}.pth finished')
    proc.wait()
    end = time.time()
    last = int(end - start)
    print(f'epoch{i}.pth: {last // 60}min {last % 60}s')
f.close()
