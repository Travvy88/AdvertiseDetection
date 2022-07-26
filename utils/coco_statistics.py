import json
import argparse

def print_coco_statistics(path):
    with open(path) as f:
        data = json.load(f)

    images_num = len(data["images"])
    annos_num = len(data["annotations"])

    print(f'{path}')
    print()
    print(f'Number of images: {images_num}')
    print(f'Number of bboxes: {annos_num}')
    print()
    h = 0
    w = 0
    for image in data['images']:
        h += image['height']
        w += image['width']

    print(f'Mean height: {round(h / images_num)}')
    print(f'Mean width: {round(w / images_num)}')
    print()

    s, m, l = 0, 0, 0
    for anno in data['annotations']:
        if anno['area'] < 32**2:
            s += 1
        elif 32**2 < anno['area'] < 96**2:
            m += 1
        elif 96**2 <  anno['area']:
            l += 1

    print(f'{s} small bboxes, {round(s / annos_num *100)}%')
    print(f'{m} medium bboxes, {round(m / annos_num *100)}%')
    print(f'{l} large bboxes, {round(l / annos_num *100)}%')
    print('-------')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='This script shows statistics of dataset by its json file')
    parser.add_argument('json', type=str,
                        help='path to coco json file')
    args = parser.parse_args()
    print_coco_statistics(args.json)
