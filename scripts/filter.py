import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('ckpt_dir')
args = parser.parse_args()

result_files = []
output_files = []
for name in ['source_train', 'source_test', 'target_train', 'target_test']:
    result_file = os.path.join(args.ckpt_dir, f'{name}.txt')
    if not os.path.exists(result_file):
        continue
    output_file = os.path.join(args.ckpt_dir, f'{name}.log')
    result_files.append(result_file)
    output_files.append(output_file)

for result_file, output_file in zip(result_files, output_files):
    f = open(result_file, 'r')
    out = open(output_file, 'w')

    lines = f.readlines()
    tot = len(lines)
    print(result_file, output_file, f'total lines: {tot}')
    for i in range(tot):
        line = lines[i]
        if line.startswith('overall_iou'):
            table = ''.join(lines[i-10:i+2])
            out.write(table)
        if line.startswith('mAP'):
            table = ''.join(lines[i-1:i+15])
            out.write(table)

    f.close()
    out.close()
