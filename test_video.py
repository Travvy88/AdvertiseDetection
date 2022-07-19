from mmdet.apis.inference import inference_detector, init_detector
from mmcv import Config
import cv2
import argparse
from tqdm.auto import tqdm
import sys


def parse():
    parser = argparse.ArgumentParser(description='Examination script')

    parser.add_argument("input", type=str, help='Path to video, or strem URL')
    parser.add_argument("output", type=str, help='Path to file for results')
    parser.add_argument("--render_path", type=str, default='', help='is you need to render -- specify path to new video with bboxes')
    parser.add_argument("--fps", type=int, default=1, help='out fps')
    parser.add_argument("--model", default='YOLOv3', help='Path to cfg')
    parser.add_argument("--weights", default='YOLOv3', help='Path to pth')
    parser.add_argument("--batch_size", type=int, default=1, help='samples per gpu')
    parser.add_argument("--workers", type=int, default=1, help='workers per gpu')
    parser.add_argument("--device", type=str, default='cuda', help='cuda or cpu')
    return parser.parse_args()



def write_pred_dict(result, frame_num, d):
    d['frames'].append(
        {'frame_id': frame_num,
        'bboxes': [res for res in result]}
    )


def process_video(video_path, new_fps, model, batch_size, render_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    assert new_fps <= fps, 'output fps dhouldnt be higher than ground fps of the input video'
    if new_fps == 0:
        fps_div = 1
        new_fps = fps
    else:
        fps_div = fps / new_fps

    if render_path:
        width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        video_out = cv2.VideoWriter(
            render_path, cv2.VideoWriter_fourcc(*'mp4v'), new_fps, (int(width), int(height)))
    print(fps_div)
    frame_count = 0
    batch = []
    predict = {'frames': []}
    with tqdm(total=num_frames) as pbar:
        while frame_count < num_frames:
            ret, frame = video.read()
            if frame_count % fps_div == 0:  # новый фпс
                batch.append(frame)
                if len(batch) == batch_size:  # на инференс отправляем только готовый батч
                    result = inference_detector(model, batch)
                    if render_path:
                        for i in range(len(batch)):
                            frame = model.show_result(batch[i], result[i])
                            video_out.write(frame)

                    batch = []
            frame_count += 1
            pbar.update(1)

    result = inference_detector(model, batch)
    if render_path:
        for i in range(len(batch)):
            frame = model.show_result(batch[i], result[i])
            video_out.write(frame)

    video_out.release()
    video.release()


if __name__ == '__main__':
    args = parse()

    cfg = Config.fromfile(r"{}".format(args.model))
    model = init_detector(cfg,
                          checkpoint=r"{}".format(args.weights),
                          device=args.device, cfg_options=None)
    model.cfg = cfg
    process_video(args.input, args.fps, model, args.batch_size, args.render_path)


