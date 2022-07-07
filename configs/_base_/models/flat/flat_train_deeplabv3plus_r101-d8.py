# model settings
import torch

norm_cfg = dict(type="SyncBN", requires_grad=True)
class_weight = [
                0.0,  # to ignore class index 0
                1.3094,
                7.9307,
                2.5204,
                2.7893,
                3.8451,
                4.7421,
                10.7511,
                8.4911,
                1.4714,
                22.2908,
                3.438,
                3.8198,
                1.1373,
                1.4705,
                2.8392,
                3.0825,
                1.2371,
                2.7741,
                1.711,
                2.1977,
                4.6947,
                3.2293,
                2.4836,
                1.3808,
                2.1606,
                3.1706,
                1.2473,
                3.8424,
                1.2879,
                1.7468,
                1.4803,
                3.3001,
                5.907,
                1.4587,
                2.2922,
                1.1762,
                2.7079,
                4.8334,
                5.1749,
                10.3357,
                15.9915,
                49.5745,
                6.1435,
                16.931,
                11.0705,
                16.0099,
                27.0424,
                20.0899,
                261.9115,
                85.0845,
                12.7646,
                27.5615,
                21.5828,
                5.9982,
                5.44,
                6.7453,
                8.7866,
                10.737,
                24.3963,
                127.1941,
                13.087,
                1.1657,
                1.2672,
                1.1318,
                1.4655,
                1.2017,
                1.2553,
                1.2453,
                1.2389,
                1.6704,
                1.6002,
                1.1658,
                1.1931,
                1.1767,
                1.154,
                1.4831,
                1.5085,
                1.1325,
                1.5408,
                1.1906,
                1.2669,
                1.3932,
                1.24,
                3.4534,
                1.2263,
                2.3743,
                1.2979,
                2.3071,
                1.6897,
                1.5851,
                1.3493,
                1.6241,
                1.1122,
                1.5855,
                1.238,
                1.1246,
                1.5287,
                2.0,
                1.1217,
                1.154,
                1.2089,
                2.2261,
                1.2162,
                1.6166,
                1.1964,
                1.2989,
                1.6594,
                2.2174,
                1.3038,
                2.7984,
                2.6624,
                1.267,
                1.5744,
                1.9203,
                1.5949,
                1.2885,
                1.5045,
                1.5132,
                1.2924,
                1.1356,
                1.1443,
                2.215,
                7.1306,
                1.317,
                2.6426,
                2.4031,
                2.7809,
                1.2184,
                1.409,
                2.386,
                2.0579,
                4.1044,
                1.4857,
                2.0632,
                1.2723,
                1.3247,
                1.1754,
                1.2106,
                1.2511,
                2.2139,
                1.1638,
                2.7909,
                1.1666,
                3.5525,
                1.1659,
                1.1772,
                3.5871,
                1.2415,
                2.002,
                1.5486,
                1.4507,
                2.2593,
                1.2887,
                3.3814,
                2.9056,
                7.666,
                1.1715,
                2.3014,
                1.5286,
                2.3232,
                2.5785,
                2.6018,
                6.7561,
                8.6138,
                3.962,
                3.8984,
                1.477,
                2.2984,
                4.7748,
                2.32,
                1.7453,
                2.4515,
                6.0556,
                8.4226,
                5.3292,
                1.1289,
                1.7619,
                1.2417,
                1.3741,
                3.9259,
                2.3559,
                8.7231,
                1.6582,
                1.6902,
                31.1587,
                104.3145,
                1.2064,
                1.3798,
                2.9461,
                18.8241,
            ]

model = dict(
    type="EncoderDecoder",
    pretrained="open-mmlab://resnet101_v1c",
    backbone=dict(
        type="ResNetV1c",
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),  # os8
        strides=(1, 2, 1, 1),
        multi_grid=(1, 2, 4),  # additional
        norm_cfg=norm_cfg,
        norm_eval=False,
        style="pytorch",
        contract_dilation=True,
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
            avg_non_ignore=True,
            class_weight=class_weight
        ),
        sampler=dict(type="OHEMPixelSampler", thresh=0.7, min_kept=32000), # batch_size * 512 * 512 //16
    ),
    auxiliary_head=dict(
        type="FCNHead",
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=0.4,
            avg_non_ignore=True,
            class_weight=class_weight
        ),
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)
