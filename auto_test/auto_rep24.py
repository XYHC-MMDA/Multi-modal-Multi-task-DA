from subprocess import Popen
import argparse
import os 
import time 

# variants
ckpt_ver = 'rep_24'
cfg_ver = 'rep_24'
l, r = 1, 24  # [l, r]


filename = f'checkpoints/merge_disc/fusion_train_usa_{ckpt_ver}/testresult.txt'
f = open(filename, 'a')
for i in range(l, r+1):
    start = time.time()
    proc = Popen(['python', './tools/mmda_test.py',
                  f'configs/merge_disc/fusion_train_usa_{cfg_ver}.py',
                  f'checkpoints/merge_disc/fusion_train_usa_{ckpt_ver}/epoch_{i}.pth',
                  '--eval', 'mAP',
                  '--json', f'checkpoints/merge_disc/fusion_train_usa_{ckpt_ver}/{i}'], stdout=f)
    # print(f'epoch_{i}.pth finished')
    proc.wait()
    end = time.time()
    last = int(end - start)
    print(f'epoch{i}.pth: {last // 60}min {last % 60}s')
f.close()
