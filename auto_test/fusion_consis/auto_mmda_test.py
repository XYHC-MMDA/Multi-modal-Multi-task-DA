from subprocess import Popen
import argparse
import os 
import time 

# variants
ckpt = 'baseline5_usa_reg16'
cfg = 'baseline5_usa_reg16'
l, r = 12, 24  # [l, r]


cfg_path = f'configs/fusion_consis/{cfg}.py'
ckpt_path = f'checkpoints/fusion_consis/{ckpt}'
filename = f'checkpoints/fusion_consis/{ckpt}/result.txt'
f = open(filename, 'a')
for i in range(l, r+1):
    model_path = os.path.join(ckpt_path, f'epoch_{i}.pth')
    while not os.path.exists(model_path):
        time.sleep(1800)
    start = time.time()
    proc = Popen(['python', './tools/mmda_test.py',
                  cfg_path, model_path,
                  '--eval', 'mAP',
                  '--json', f'checkpoints/fusion_consis/{ckpt}/{i}'], stdout=f)
    # print(f'epoch_{i}.pth finished')
    proc.wait()
    end = time.time()
    last = int(end - start)
    print(f'epoch{i}.pth: {last // 60}min {last % 60}s')
f.close()
