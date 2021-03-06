from subprocess import Popen
import argparse
import os 
import time 

# variants
ckpt = 'baseline2_usa_reg32'
cfg = 'baseline2_usa_reg32'
l, r = 1, 24  # [l, r]


filename = f'checkpoints/fusion_consis/{ckpt}/result.txt'
f = open(filename, 'a')
for i in range(l, r+1):
    start = time.time()
    proc = Popen(['python', './tools/mmda_test.py',
                  f'configs/fusion_consis/{cfg}.py',
                  f'checkpoints/fusion_consis/{ckpt}/epoch_{i}.pth',
                  '--eval', 'mAP',
                  '--json', f'checkpoints/fusion_consis/{ckpt}/{i}'], stdout=f)
    # print(f'epoch_{i}.pth finished')
    proc.wait()
    end = time.time()
    last = int(end - start)
    print(f'epoch{i}.pth: {last // 60}min {last % 60}s')
f.close()
