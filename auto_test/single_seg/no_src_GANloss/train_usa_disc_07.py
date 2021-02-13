from subprocess import Popen
import argparse
import os 
import time 

# variants
cfg = 'train_usa_disc_07'
ckpt = 'train_usa_disc_07'
l, r = 25, 36  # [l, r]


filename = f'checkpoints/single_seg/no_src_GANloss/{ckpt}/result.txt'
f = open(filename, 'a')
for i in range(l, r+1):
    start = time.time()
    proc = Popen(['python', './tools/single_seg_test.py',
                  f'configs/single_seg/no_src_GANloss/{cfg}.py',
                  f'checkpoints/single_seg/no_src_GANloss/{ckpt}/epoch_{i}.pth'],
                  stdout=f)
    # print(f'epoch_{i}.pth finished')
    proc.wait()
    end = time.time()
    last = int(end - start)
    print(f'epoch{i}.pth: {last // 60}min {last % 60}s')
f.close()
