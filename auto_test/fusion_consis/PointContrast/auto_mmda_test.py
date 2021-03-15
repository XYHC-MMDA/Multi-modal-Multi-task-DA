from subprocess import Popen
import argparse
import os 
import time 

# variants
ckpt = 'usa_reg32_v1'
cfg = 'usa_reg32_v1'
l, r = 11, 24  # [l, r]


cfg_path = f'configs/fusion_consis/PointContrast/{cfg}.py'
ckpt_path = f'checkpoints/fusion_consis/PointContrast/{ckpt}'
filename = f'checkpoints/fusion_consis/PointContrast/{ckpt}/result.txt'
f = open(filename, 'a')
for i in range(l, r+1):
    model_path = os.path.join(ckpt_path, f'epoch_{i}.pth')
    while not os.path.exists(model_path):
        print('sleeping...')
        time.sleep(1800)
    start = time.time()
    cmd = ['python', './tools/mmda_test.py',
           cfg_path, model_path,
           '--eval', 'mAP',
           '--json', f'checkpoints/fusion_consis/PointContrast/{ckpt}/{i}']
    print(' '.join(cmd))
    proc = Popen(cmd, stdout=f)
    # print(f'epoch_{i}.pth finished')
    proc.wait()
    end = time.time()
    last = int(end - start)
    print(f'epoch{i}.pth: {last // 60}min {last % 60}s')
f.close()
