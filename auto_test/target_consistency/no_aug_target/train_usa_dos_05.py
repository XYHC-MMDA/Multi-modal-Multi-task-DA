from subprocess import Popen
import argparse
import os 
import time 

cfg = 'train_usa_dos_04'
ckpt = 'train_usa_dos_04_resume'
l, r = 17, 24


filename = f'checkpoints/target_consistency/no_aug_target/{ckpt}/result.txt'
f = open(filename, 'a')
for i in range(l, r+1):
    start = time.time()
    proc = Popen(['python', './tools/mmda_test.py',
                  f'configs/target_consistency/no_aug_target/{cfg}.py',
                  f'checkpoints/target_consistency/no_aug_target/{ckpt}/epoch_{i}.pth',
                  '--eval', 'mAP',
                  '--json', f'checkpoints/target_consistency/no_aug_target/{ckpt}/{i}'], stdout=f)
    # print(f'epoch_{i}.pth finished')
    proc.wait()
    end = time.time()
    last = int(end - start)
    print(f'epoch{i}.pth: {last // 60}min {last % 60}s')
f.close()
