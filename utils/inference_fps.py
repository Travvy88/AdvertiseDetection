from mmdet.apis.inference import inference_detector, init_detector
from mmcv import Config

import argparse
from tqdm.auto import tqdm
import numpy as np
import time


def check_fps(model, batch_size, frames_number, img_scale=None):
    if img_scale is None:
        img_scale = model.cfg.data.test.pipeline[1].img_scale

    batch = [np.random.rand(img_scale[0], img_scale[1], 3).astype('float32') for i in range(batch_size)]
    start = time.time()
    for i in tqdm(range(0, frames_number, batch_size)):
        inference_detector(model, batch)
    end = time.time()

    print(f'Batch size: {batch_size}')
    print(f'Image scale: {img_scale}')
    print(f'Seconds total: {round(end - start, 2)}')
    print(f'Frames per sec: {round(frames_number / (end - start), 2)}')
    print(f'Sec per 1 frame: {round((end - start) / frames_number , 4)}')


def parse():
    parser = argparse.ArgumentParser(description='Examination script')
    parser.add_argument("cfg", help='Path to cfg')
    parser.add_argument("--frames_number", type=int, default=1000, help='number of batches to ')
    parser.add_argument("--batch_size", type=int, default=1, help='samples per gpu')
    parser.add_argument("--device", type=str, default='cuda', help='cuda or cpu')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse()

    cfg = Config.fromfile(r"{}".format(args.cfg))
    model = init_detector(cfg, device=args.device)
    model.cfg = cfg
    print(f'using {args.device}...')
    check_fps(model, args.batch_size, args.frames_number)
