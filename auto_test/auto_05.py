from subprocess import Popen
import argparse
import os 
import time 

# parser = argparse.ArgumentParser()
# parser.add_argument('--work-dir', type=str)
# args = parser.parse_args()

ver = '05'
filename = f'checkpoints/merge_disc/fusion_train_usa_{ver}/result.txt'
f = open(filename, 'a')

for i in range(3, 37):
    start = time.time()
    proc = Popen(['python', './tools/mmda_test.py',
                  f'configs/merge_disc/fusion_train_usa_{ver}.py',
                  f'checkpoints/merge_disc/fusion_train_usa_{ver}/epoch_{i}.pth',
                  '--eval', 'mAP',
                  '--json', f'checkpoints/merge_disc/fusion_train_usa_{ver}/{i}'], stdout=f)
    # print(f'epoch_{i}.pth finished')
    proc.wait()
    end = time.time()
    last = int(end - start)
    print(f'epoch{i}.pth: {last // 60}min {last % 60}s')
f.close()
