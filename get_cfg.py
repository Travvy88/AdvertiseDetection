from mmcv import Config
cfg = Config.fromfile('mmdetection/configs/yolo/yolov3_mobilenetv2_mstrain-416_300e_coco.py')
del cfg.data.train.dataset.pipeline[3]
del cfg.data.train.dataset.pipeline[2]

from mmdet.apis import set_random_seed

classes = ['logo']
# Modify dataset type and path
cfg.dataset_type = 'CocoDataset'
cfg.data.samples_per_gpu = 24

cfg.data.test.type = 'CocoDataset'
cfg.data.test.data_root = 'data/openlogo_coco/'
cfg.data.test.ann_file = 'test_coco.json'
cfg.data.test.img_prefix = 'JPEGImages'
cfg.data.test.classes = classes

cfg.data.train.dataset.type = 'CocoDataset'
cfg.data.train.dataset.data_root = 'data/openlogo_coco/'
cfg.data.train.dataset.ann_file = 'train_coco.json'
cfg.data.train.dataset.img_prefix = 'JPEGImages'
cfg.data.train.dataset.classes = classes
cfg.data.train.times = 1

cfg.data.val.type = 'CocoDataset'
cfg.data.val.data_root = 'data/openlogo_coco/'
cfg.data.val.ann_file = 'test_coco.json'
cfg.data.val.img_prefix = 'JPEGImages'
cfg.data.val.classes = classes

cfg.load_from = 'models/yolov3_mmdet/yolov3_mobilenetv2_mstrain-416_300e_coco_20210718_010823-f68a07b3.pth'
# cfg.resume_from = '/content/mmdetection/tutorial_exps/latest.pth'
cfg.runner.max_epochs = 100
cfg.data.workers_per_gpu = 2
# modify num classes of the model in box head
cfg.model.bbox_head.num_classes = 1
## cfg.model.roi_head.bbox_head.num_classes =  1

# If we need to finetune a model based on a pre-trained detector, we need to
# use load_from to set the path of checkpoints.
# cfg.load_from = 'checkpoints/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth'

# Set up working dir to save files and logs.
cfg.work_dir = './models/yolov3_mmdet/logs'

# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 8 since we only use one GPU.
# cfg.optimizer.lr = 0.001
cfg.optimizer = dict(type='Adam')
# cfg.optimizer_config = None      ####### ПРОТЕСТИТЬ
cfg.lr_config.warmup = None
cfg.log_config.interval = 100

# Change the evaluation metric since we use customized dataset.
cfg.evaluation.metric = 'bbox'
# We can set the evaluation interval to reduce the evaluation times
cfg.evaluation.interval = 5
# We can set the checkpoint saving interval to reduce the storage cost
cfg.checkpoint_config.interval = 5

# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)
cfg.device = 'cuda'

# We can also use tensorboard to log the training process
cfg.log_config.hooks = [
    dict(type='TextLoggerHook'),
    dict(type='TensorboardLoggerHook')]

# We can initialize the logger for training and have a look
# at the final config used for training
print(f'Config:\n{cfg.pretty_text}')
cfg.dump('models/yolov3_mmdet/cfg.py')
