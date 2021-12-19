_base_ = [
    '../../../mmdetection/configs/_base_/schedules/schedule_1x.py', '../../../mmdetection/configs/_base_/default_runtime.py'
]
custom_imports = dict(
    imports=['xray.models.dense_heads.ld_ray_head'],
    allow_failed_imports=False)
img_scale = (640, 640)
teacher_ckpt = '/root/Desktop/X_ray/project/work_dirs/best_bbox_mAP_epoch_73.pth'

model = dict(
    type='KnowledgeDistillationSingleStageDetector',
    teacher_config='/root/Desktop/X_ray/project/configs/x_ray/yolox_ray_s.py',
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
            checkpoint='open-mmlab://detectron/resnet50_caffe')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='XRAYHead',
        num_classes=84,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_ld=dict(
            type='KnowledgeDistillationKLDivLoss', loss_weight=0.25, T=10),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))

img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
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
data_root = 'data/x_ray/'
classes = ('口红', '面包', '鼠标', '雨伞', '衣服', '钳子', '面膜', '薯片', '相册', '眉笔', '手串', '面霜', '手机盒', '报警灯', '棒棒糖', '牙刷', '滤水网', '脚踏', '纸巾盒', '牛奶', '护肤品', '牙膏', '洗发露', '手表', '桔子', '皮带扣', '手机', '蓝牙音响', '夹子', '手机支架', '杯子', '挂件', '打火机', '防晒霜', '感冒灵', '包', '硬币', '档案袋', '瓶子', '耳机', '花', '便签', '酒瓶', '易拉罐', '订书机', '奥特曼', '钱包', '火腿肠', '纽扣', '盒子', '啫喱水', '零食', '粽子', '体温枪', '瓜子', '饮料', '显卡', '盖子', '印章', '口香糖', '艾灸贴', '眼镜盒', '玩具', '耳机盒', '美妆蛋', '秒表', '摆件', '水管', '充电器', '罐子', '饼干', '钥匙', '收音机', '螺丝', '刮胡刀', '矿泉水', '笔筒', '风扇', '洗脸巾', '双面胶', '核桃', '速溶咖啡', '螺丝刀', 'Unknown')


data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'ann/train.json',
        img_prefix=data_root + 'train/',
        pipeline=train_pipeline,
        filter_empty_gt=False,
        ),
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
runner = dict(type='EpochBasedRunner', max_epochs=12)
