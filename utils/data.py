import xml.etree.ElementTree as ET
import os 

def change_to_one_class(path, label='logo'):
    # сделать в xml один общий класс {label}
    tree = ET.parse(path)
    root = tree.getroot()
    names = root.findall(".//*[name]")
    for name in names:
        name[0].text = label

    tree.write(path)


def add_path_to_xml(path):
    # добавляет путь к фото в xml файл
    tree = ET.parse(path)

    list_path = path.split('/')
    folder = list_path[3]
    brand = list_path[4]
    filename = list_path[5][:-4]
    path_to_jpg = os.path.join(folder, brand,  filename + '.jpg')

    root = tree.getroot()
    path_elem = ET.SubElement(root, 'path')
    path_elem.text = path_to_jpg

    tree.write(path)