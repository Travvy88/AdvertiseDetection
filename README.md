# AdvertiseDetection

Параметры тестирования:
- Обучение на openlogo trainall
- Валидация на openlogo testall
- Видео 45 секунд
- Batch size 100
- На выход 5 fps
- GPU GTX1050ti
- CPU i7-7700HQ

| model      | mAP@0.5   | mAP@0.5:0.05:0.95 | time      |
|------------|-----------|-------------------|-----------|
| YOLOv3     | 0.582     | 0.325             | 1m 10s    |
| YOLOX-tiny | **0.669** | **0.456**         | **1m 5s** |