import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('ckpt_dir')
args = parser.parse_args()

result_file = os.path.join(args.ckpt_dir, 'result.txt')
output_file = os.path.join(args.ckpt_dir, 'filter.txt')
assert not os.path.exists(output_file)

f = open(result_file, 'r')
out = open(output_file, 'w')
lines = f.readlines()
tot = len(lines)
print(tot)
for i in range(tot):
    line = lines[i]
    if line.startswith('overall_iou'):
        table = ''.join(lines[i-10:i+1])
        out.write(table)
    if line.startswith('mAP'):
        table = ''.join(lines[i-1:i+15])
        out.write(table)
        
f.close()
out.close()
