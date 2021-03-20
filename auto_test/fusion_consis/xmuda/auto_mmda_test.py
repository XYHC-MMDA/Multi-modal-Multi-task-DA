from subprocess import Popen
import argparse
import os 
import time 

# variants
cfg = 'baseline3_usa'
ckpt = 'baseline3_usa'
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
