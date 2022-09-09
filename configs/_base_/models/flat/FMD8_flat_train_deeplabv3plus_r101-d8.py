# model settings
norm_cfg = dict(type="SyncBN", requires_grad=True)

checkpoint_file="/netscratch/gautam/semseg/exp_results/FMD7/training/20220726_235911/epoch_10.pth" ###NEW::

model = dict(
    type="EncoderDecoder",
    backbone=dict(
        type="ResNetV1c",
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),  # os8
        strides=(1, 2, 1, 1),
        multi_grid=(1, 2, 4),  # additional
        norm_cfg=norm_cfg,
        #frozen_stages=4,  #### NEW:::::::::::Freezing backbone
        norm_eval=False,
        style="pytorch",
        contract_dilation=True,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')
    ),
    decode_head=dict(
        type="DepthwiseSeparableASPPHead",
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),  #  rate
        c1_in_channels=256,
        c1_channels=48,
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=1.0,
            avg_non_ignore=True
        ),
        sampler=dict(type="OHEMPixelSampler", thresh=0.7, min_kept=32000), # batch_size * 512 * 512 //16
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)