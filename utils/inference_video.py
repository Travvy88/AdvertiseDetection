from mmdet.apis.inference import inference_detector, init_detector
from mmcv import Config
import cv2
import argparse
from tqdm.auto import tqdm


def parse():
    parser = argparse.ArgumentParser(description='Test model fps')

    parser.add_argument("input", type=str, help='Path to video, or strem URL')
    parser.add_argument("output", type=str, help='Path to file for results')
    parser.add_argument("model", help='Path to cfg')
    parser.add_argument("weights", help='Path to pth')

    parser.add_argument("--render_path", type=str, default='', help='is you need to render -- specify path to new video with bboxes')
    parser.add_argument("--fps", type=int, default=0, help='out fps')
    parser.add_argument("--batch_size", type=int, default=1, help='samples per gpu')
    parser.add_argument("--device", type=str, default='cuda', help='cuda or cpu')
    return parser.parse_args()


def process_video(video_path, new_fps, model, batch_size, render_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    assert new_fps <= fps, 'output fps shouldnt be higher than ground fps of the input video'
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
                    # TODO save bboxes to file
                    if render_path:
                        for i in range(len(batch)):
                            frame = model.show_result(batch[i], result[i])
                            video_out.write(frame)

                    batch = []
            frame_count += 1
            pbar.update(1)
    if batch_size > 1:
        result = inference_detector(model, batch)
        # TODO save bboxes to file
        if render_path:
            for i in range(len(batch)):
                frame = model.show_result(batch[i], result[i])
                video_out.write(frame)

    if render_path:
        video_out.release()
    video.release()


if __name__ == '__main__':
    args = parse()

    cfg = Config.fromfile(r"{}".format(args.model))
    model = init_detector(cfg,
                          checkpoint=r"{}".format(args.weights),
                          device=args.device, cfg_options=None)
    model.cfg = cfg
    print(f'using {args.device}...')
    process_video(args.input, args.fps, model, args.batch_size, args.render_path)
