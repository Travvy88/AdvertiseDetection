import argparse
import mmcv


def osld2coco_logo(ann_file, out_file):
    data_infos = mmcv.load(ann_file)

    annotations = []
    images = []
    obj_count = 0
    for idx, filename in enumerate(mmcv.track_iter_progress(data_infos.keys())):
        image = data_infos[filename]
        height, width = mmcv.imread('osld/product-images/' + filename).shape[:2]

        images.append(dict(
            id=idx,
            file_name=filename,
            height=height,
            width=width))

        for anno in image:
            x_min, y_min, x_max, y_max = anno[0]

            data_anno = dict(
                image_id=idx,
                id=obj_count,
                category_id=0,
                bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                area=(x_max - x_min) * (y_max - y_min),
                segmentation=None,
                iscrowd=0)
            annotations.append(data_anno)
            obj_count += 1

    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=[{'id': 0, 'name': 'logo'}])
    mmcv.dump(coco_format_json, out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='This script support converting osld dataset format to coco format json')
    parser.add_argument('annotation', type=str,
                        help='path to osld json file')
    parser.add_argument('output', type=str,
                        help='path to output coco json file')
    args = parser.parse_args()
    osld2coco_logo(args.annotation, args.output)