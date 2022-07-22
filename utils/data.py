import xml.etree.ElementTree as ET
import os
import cv2
from matplotlib import pyplot as plt
import json

def change_to_one_class(path, label='logo'):
    # сделать в xml один общий класс {label}
    tree = ET.parse(path)
    root = tree.getroot()
    names = root.findall(".//*[name]")
    for name in names:
        name[0].text = label

    tree.write(path)


def add_path_to_xml(path):
    # добавляет путь к фото в xml файл: ктуально, когда фото лежат не в одной папке, а  разных
    tree = ET.parse(path)

    list_path = path.split('/')
    folder = list_path[3]
    brand = list_path[4]
    filename = list_path[5][:-4]
    path_to_jpg = os.path.join(folder, brand,  filename + '.jpg')

    root = tree.getroot()
    paths = root.findall(".//path")
    if len(paths) == 0:
        path_elem = ET.SubElement(root, 'path')
        path_elem.text = path_to_jpg
    else:
        paths[0].text = path_to_jpg
    tree.write(path)


def check_coco_dataset(json_path, idx):
    # выводит сэмп из фото по аннотации
    with open(json_path) as json_file:
        data = json.load(json_file)

    anno = data['annotations'][idx]
    bbox = anno['bbox']
    image = data['images'][anno['image_id']]
    file_name = image['file_name']

    img = cv2.imread('openlogo_coco/JPEGImages/' + file_name)
    pt1 = (bbox[0], bbox[1])
    pt2 = (bbox[0] + bbox[2], bbox[1] + bbox[3])

    img = cv2.rectangle(img, pt1, pt2, (255, 0, 0), 3)
    plt.imshow(img)
