from subprocess import Popen
import argparse
import os 
import time 

cfg = 'train_day_03'
ckpt = 'train_day_03'
l, r = 18, 36


filename = f'checkpoints/fusion_disc/no_src_GANloss/{ckpt}/result.txt'
f = open(filename, 'a')
for i in range(l, r+1):
    start = time.time()
    proc = Popen(['python', './tools/mmda_test.py',
                  f'configs/fusion_disc/no_src_GANloss/{cfg}.py',
                  f'checkpoints/fusion_disc/no_src_GANloss/{ckpt}/epoch_{i}.pth',
                  '--eval', 'mAP',
                  '--json', f'checkpoints/fusion_disc/no_src_GANloss/{ckpt}/{i}'], stdout=f)
    # print(f'epoch_{i}.pth finished')
    proc.wait()
    end = time.time()
    last = int(end - start)
    print(f'epoch{i}.pth: {last // 60}min {last % 60}s')
f.close()
