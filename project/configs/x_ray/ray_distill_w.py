_base_ = [
    '../../../mmdetection/configs/_base_/schedules/schedule_1x.py', '../../../mmdetection/configs/_base_/default_runtime.py'
]
#load_from = '/share/home/yikun/mdev/CVFinal/project/checkpoints/fcos.pth'
load_from = '/share/home/yikun/mdev/CVFinal/project/work_dirs/ray_distill_w/epoch_4.pth'
find_unused_parameters=True

class_weight = [1]*84
d_list = [5, 32, 38, 42, 43, 74, 55, 82]
for id in d_list:
    class_weight[id] = 2


custom_imports = dict(
    imports=['xray.models.dense_heads.ld_ray_head'],
    allow_failed_imports=False)
img_scale = (640, 640)
teacher_ckpt = '/share/home/yikun/mdev/CVFinal/project/checkpoints/best_bbox_mAP_epoch_91.pth'

model = dict(
    type='KnowledgeDistillationSingleStageDetector',
    teacher_config='/share/home/yikun/mdev/CVFinal/project/configs/x_ray/yolox_ray_s.py',
    teacher_ckpt=teacher_ckpt,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/share/home/yikun/mdev/CVFinal/project/checkpoints/resnet50_caffe.pth')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
    type='XRAYHead', 
    num_classes=84, 
    in_channels=256, 
    feat_channels=256, 
    loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            class_weight=class_weight,
            loss_weight=1.0),
    loss_kl=dict(type='KnowledgeDistillationKLDivLoss', loss_weight=0.25, T=10)),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.1, nms=dict(type='nms', iou_threshold=0.65)))
img_norm_cfg = dict(
    mean=[201.8509,215.0482,225.3103], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

dataset_type = 'CocoDataset'
data_root = 'data/'
classes = ('口红', '面包', '鼠标', '雨伞', '衣服', '钳子', '面膜', '薯片', '相册', '眉笔', '手串', '面霜', '手机盒', '报警灯', '棒棒糖', '牙刷', '滤水网', '脚踏', '纸巾盒', '牛奶', '护肤品', '牙膏', '洗发露', '手表', '桔子', '皮带扣', '手机', '蓝牙音响', '夹子', '手机支架', '杯子', '挂件', '打火机', '防晒霜', '感冒灵', '包', '硬币', '档案袋', '瓶子', '耳机', '花', '便签', '酒瓶', '易拉罐', '订书机', '奥特曼', '钱包', '火腿肠', '纽扣', '盒子', '啫喱水', '零食', '粽子', '体温枪', '瓜子', '饮料', '显卡', '盖子', '印章', '口香糖', '艾灸贴', '眼镜盒', '玩具', '耳机盒', '美妆蛋', '秒表', '摆件', '水管', '充电器', '罐子', '饼干', '钥匙', '收音机', '螺丝', '刮胡刀', '矿泉水', '笔筒', '风扇', '洗脸巾', '双面胶', '核桃', '速溶咖啡', '螺丝刀', 'Unknown')

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'ann/train.json',
        img_prefix=data_root + 'train/',
        pipeline=train_pipeline,
        filter_empty_gt=False),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'ann/test.json',
        img_prefix=data_root + 'test/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'ann/test.json',
        img_prefix=data_root + 'test/',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(
    lr=0.01, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
evaluation = dict(
    save_best='auto',
    interval=2,
    classwise=True,
    metric='bbox')
runner = dict(type='EpochBasedRunner', max_epochs=4)
