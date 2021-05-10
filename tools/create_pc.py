import random
import mmcv
import os


def load_shuffle_dump(pkl_file, pc=1):
    assert os.path.isfile(pkl_file)
    dir = os.path.dirname(pkl_file)
    pkl_name = os.path.basename(pkl_file).split('.')[0]
    save_path = os.path.join(dir, f'{pkl_name}_{pc}pc.pkl')

    # load pkl
    data = mmcv.load(pkl_file)  # dict with keys=('infos', 'metadata')
    content = data['infos']  # list of info dict
    metadata = data['metadata']

    # shuffle content
    anum = len(content)
    bnum = (anum * pc // 100)
    print('#samples:', anum, bnum)
    random.shuffle(content)
    save_content = content[:bnum]
    metadata = f'{metadata}_{pc}pc'

    # dump
    mmcv.dump(dict(infos=save_content, metadata=metadata), save_path)


if __name__ == '__main__':
    load_shuffle_dump('/home/xyyue/xiangyu/nuscenes_unzip/mmda_xmuda_split/train_day.pkl', pc=1)
    load_shuffle_dump('/home/xyyue/xiangyu/nuscenes_unzip/mmda_xmuda_split/train_day.pkl', pc=10)
    load_shuffle_dump('/home/xyyue/xiangyu/nuscenes_unzip/mmda_xmuda_split/train_night.pkl', pc=1)
    load_shuffle_dump('/home/xyyue/xiangyu/nuscenes_unzip/mmda_xmuda_split/train_night.pkl', pc=10)


