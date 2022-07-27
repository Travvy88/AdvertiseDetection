# AdvertiseDetection

Параметры тестирования:
- Обучение на openlogo trainall
- Валидация на openlogo testall

Параметры fps теста:
- 10 000 resized кадров
- Batch size 10
- GPU GTX1050ti
- CPU i7-7700HQ

| model      | mAP@0.5   | mAP@0.5:0.05:0.95 | FPS       |
|------------|-----------|-------------------|-----------|
| YOLOv3     | 0.582     | 0.325             | 32fps     |
| YOLOX-tiny | **0.669** | **0.456**         | **48fps** |
