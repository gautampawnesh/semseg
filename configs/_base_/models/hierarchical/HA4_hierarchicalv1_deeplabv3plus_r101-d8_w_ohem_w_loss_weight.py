
"""
MeletisDubblemann
Introduced Weights for l1 and l2& l3 as 1:0.1:0.1
ignore_index=0 in loss computation.
"""
norm_cfg = dict(type="SyncBN", requires_grad=True)



class_hierarchy_heads = dict(
    level_1_head=dict(
        # ignore=[0], # naming convention doesnt make sense in this: auto hierarchy generated from features.
        vehicles=[1, 14, 26, 38, 46, 63, 64, 65, 69, 74, 76, 78, 89, 93, 100, 101, 109, 113, 129, 133, 134, 138, 142, 159, 160, 170, 178, 188],  #
        flat=[7, 11, 18, 27, 42, 43, 48, 50, 51, 52, 56, 62, 95, 99, 124, 132, 144, 145, 151, 152, 156, 162, 167, 174, 175, 186],  #
        construction=[6, 45, 47, 53, 60, 84, 87, 104, 116, 123, 125, 126, 127, 130, 131, 135, 137, 140, 143, 148, 153, 157, 165, 168, 169, 177], #
        traffic_objects=[4, 8, 9, 13, 16, 23, 24, 25, 31, 41, 61, 83, 88, 90, 92, 94, 97, 106, 111, 115, 117, 141, 146, 149, 158, 181, 182, 189], #
        nature_objects=[5, 19, 20, 22, 29, 49, 54, 72, 80, 81, 85, 96, 103, 107, 108, 114, 128, 136, 147, 150, 161, 164, 171, 173, 176, 180, 187], #
        vru_objects=[3, 12, 17, 21, 33, 34, 36, 39, 40, 55, 57, 58, 59, 75, 77, 91, 98, 119, 120, 121, 122, 139, 154, 163, 166, 172, 190], #
        indoor_objects=[2, 10, 15, 28, 30, 32, 35, 37, 44, 66, 67, 68, 70, 71, 73, 79, 82, 86, 102, 105, 110, 112, 118, 155, 179, 183, 184, 185] #
    ),
    # Vehicle head
    level_2_vehicle_head=dict(
        small_vehicles=[26, 63, 69, 100, 159, 160, 178],
        large_vehicles=[46, 89, 129, 134, 142, 188],
        two_wheelers=[1, 74, 76, 78, 93, 109, 113, 170],
        other_vehicles=[14, 38, 64, 65, 101, 133, 138],
    ),
    level_3_small_vehicles_head=dict(
        rand1=[26],
        rand2=[63],
        rand3=[69],
        rand4=[100],
        rand5=[159],
        rand6=[160],
        rand7=[178]
    ),
    level_3_large_vehicles_head=dict(
        rand11=[46],
        rand21=[89],
        rand31=[129],
        rand41=[134],
        rand51=[142],
        rand61=[188]
    ),
    level_3_two_wheelers_head=dict(
        rand11=[1],
        rand21=[74],
        rand31=[76],
        rand41=[78],
        rand51=[93],
        rand61=[109],
        rand71=[113],
        rand81=[170]
    ),
    level_3_other_vehicles_head=dict(
        rand31=[14],
        rand312=[38],
        rand313=[64],
        rand314=[65],
        rand315=[101],
        rand316=[133],
        rand317=[138]
    ),
    # Road head
    level_2_flat_head=dict(
        normal_road=[7, 11, 27, 48, 50, 51, 56, 62, 99, 132, 144, 151, 162, 175],
        road_marking=[18, 42, 43, 52, 95, 124, 145, 152, 156, 167, 174, 186]
    ),
    level_3_normal_road_head=dict(
        road=[7],
        parking=[11],
        rail_track=[27],
        sidewalk=[48],
        curb=[50],
        pedestrian_area=[51],
        dirt_track=[56],
        runway=[62],
        crosswalk_plain=[99],
        bikelane=[132],
        service_lane=[144],
        path=[151],
        curb_cut=[162],
        rand_=[175]
    ),
    level_3_road_marking_head=dict(
        general_marking=[18],
        zebra_marking=[42],
        manhole=[43],
        catch_basin=[52],
        pothole=[95],
        all_road_marking=[124],
        rand21=[145],
        rand212=[152],
        rand213=[156],
        rand214=[167],
        rand215=[174],
        rand216=[186]
    ),
    # construction
    level_2_construction_head=dict(
        building_infra=[6, 45, 47, 125, 126, 127, 130, 135, 140, 148, 157, 168],
        other_infra=[53, 60, 84, 87, 104, 116, 123, 131, 137, 143, 153, 165, 169, 177]
    ),
    level_3_building_infra_head=dict(
        building=[6],
        grandstand=[45],
        stairs=[47],
        house=[125],
        randc=[126],
        step_stair=[127],
        stage=[130],
        stairway=[135],
        skyscraper=[140],
        column=[148],
        wall=[157],
        bridge=[168]
    ),
    level_3_other_infra_head=dict(
        pier_dock=[53],
        canopy=[60],
        fountain=[84],
        bannister=[87],
        barrier=[104],
        awning_sunshade=[116],
        hovel_hut=[123],
        rand_1=[131],
        rand_2=[137],
        rand_3=[143],
        rand_4=[153],
        rand_5=[165],
        rand_6=[169],
        rand_7=[177]
    ),

    # Road objects
    level_2_traffic_objects_head=dict(
        pole_all=[25, 92, 97, 106, 141, 149, 158],
        sign=[23, 24, 41, 83, 115, 181],
        light=[4, 8, 13, 31, 111, 117, 146, 182],
        other_road_objects=[9, 16, 61, 88, 90, 94, 189]
    ),
    level_3_pole_all_head=dict(
        pole=[25],
        utility_pole=[92],
        tower=[97],
        rand_1=[106],
        rand_2=[141],
        rand_3=[149],
        rand_4=[158]
    ),
    level_3_sign_head=dict(
        traffic_sign=[23],
        billboard=[24],
        banner=[41],
        traffic_sign_frame=[83],
        traffic_sign_back=[115],
        trade_brand=[181]
    ),
    level_3_light_head=dict(
        street_light=[4],
        traffic_light=[8],
        light_source=[13],
        rand_1=[31],
        rand_2=[111],
        rand_3=[117],
        rand_4=[146],
        rand_13=[182]
    ),
    level_3_other_road_objects_head=dict(
        trashcan=[9],
        phone_booth=[16],
        cctv_camera=[61],
        junction_box=[88],
        fyre_hydrant=[90],
        mailbox=[94],
        bikerack=[189]
    ),
    # Nature objects
    level_2_nature_objects_head=dict(
        sky_vegetation=[20, 49, 80, 85, 107, 108, 128, 150, 180],
        ground=[19, 22, 29, 72, 81, 96, 114, 171, 173, 187],
        other_nature_objects=[5, 54, 103, 136, 147, 161, 164, 176]
    ),
    level_3_sky_vegetation_head=dict(
        tree=[20],
        palm_tree=[49],
        sky=[80],
        plant=[85],
        vegetation=[107],
        rand_1=[108],
        rand_2=[128],
        rand_3=[150],
        rand_4=[180]
    ),
    level_3_ground_head=dict(
        terrain=[19],
        mountain=[22],
        earth_ground=[29],
        soil_ground=[72],
        hill=[81],
        field=[96],
        rock=[114],
        rand_1=[171],
        rand_2=[173],
        rand_3=[187]
    ),
    level_3_other_nature_objects_head=dict( #[5, 54, 103, 136, 147, 161, 164, 176]
        water=[5],
        snow=[54],
        sand=[103],
        river=[136],
        lake=[147],
        sea=[161],
        water_fall=[164],
        swimming_pool=[176]
    ),
    # VRU
    level_2_vru_head=dict(
        animal=[17, 34, 36, 55, 57, 58, 75, 77, 98, 119, 120, 121, 163, 166],
        rider=[3, 12, 21, 33, 39, 40, 59, 91, 122, 139, 154, 172, 190]
    ),
    level_3_animal_head=dict(
        animal=[17],
        bird=[34],
        rand_1=[36],
        rand_2=[55],
        rand_3=[57],
        rand_4=[58],
        rand_5=[75],
        rand_6=[77],
        rand_7=[98],
        rand_11=[119],
        rand_12=[120],
        rand_13=[121],
        rand_14=[163],
        rand_15=[166]
    ),
    level_3_rider_head=dict( # [3, 12, 21, 33, 39, 40, 59, 91, 122, 139, 154, 172, 190]
        person=[3],
        motorcyclist=[12],
        bicyclist=[21],
        other_rider=[33],
        rand_1=[39],
        rand_2=[40],
        rand_3=[59],
        rand_4=[91],
        rand_5=[122],
        rand_6=[139],
        rand_7=[154],
        rand_11=[172],
        rand_12=[190]
    ),
    # Indoor objects
    level_2_indoor_objects_head=dict(
        furniture=[15, 66, 67, 82, 110],
        bedroom_objects=[28, 35, 70, 179, 183],
        bathroom_objects=[10, 37, 68, 73, 112],
        other_indoor_objects=[2, 32, 155, 184, 185],
        electronics=[30, 71, 79, 102, 118],
        kitchen_objects=[44, 86, 105]
    ),
    level_3_furniture_head=dict(
        bed=[15],
        cabinet=[66],
        door=[67],
        table=[82],
        chair=[110]
    ),
    level_3_bedroom_objects_head=dict(
        cushion=[28],
        pillow=[35],
        lamp=[70],
        rug=[179],
        mirror=[183]
    ),
    level_3_bathroom_objects_head=dict(
        bathtub=[10],
        toilet=[37],
        towel=[68],
        shower=[73],
        shower_curtain=[112],
    ),
    level_3_other_indoor_objects_head=dict( #[2, 32, 155, 184, 185]
        escalator=[2],
        flag=[32],
        clock=[155],
        radiator=[184],
        glass_drinking=[185]
    ),
    level_3_electronics_head=dict( # [30, 71, 79, 102, 118]
        monitor=[30],
        crt_screen=[71],
        screen_projector=[79],
        television=[102],
        arcade_machine=[118]
    ),
    level_3_kitchen_objects_head=dict( # [44, 86, 105]
        hood_exhaust=[44],
        dish_washer=[86],
        microwave=[105]
    )
)

common_decode_head = dict(
    type="DepthwiseSeparableASPPHead",
    in_channels=2048,
    in_index=3,
    channels=512,
    dilations=(1, 12, 24, 36),
    c1_in_channels=256,
    c1_channels=48,
    dropout_ratio=0.1,
    norm_cfg=norm_cfg,
    ignore_index=0,
    align_corners=False,
    loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.1, avg_non_ignore=True),
    sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=32000),
)


hierarchical_decode_heads_config = dict(
    level_1_head=dict(common_decode_head, **{"num_classes": len(class_hierarchy_heads["level_1_head"])+1,
                                             "loss_decode": dict(type="CrossEntropyLoss", use_sigmoid=False,
                                                                 loss_weight=1.0, avg_non_ignore=True)}), #0
    # Vehicle head
    level_2_vehicle_head=dict(common_decode_head, **{"num_classes": len(class_hierarchy_heads["level_2_vehicle_head"])+1}),  #1
    level_3_small_vehicles_head=dict(common_decode_head, **{"num_classes": len(class_hierarchy_heads["level_3_small_vehicles_head"])+1}), #2
    level_3_large_vehicles_head=dict(common_decode_head, **{"num_classes": len(class_hierarchy_heads["level_3_large_vehicles_head"])+1}), #3
    level_3_two_wheelers_head=dict(common_decode_head, **{"num_classes": len(class_hierarchy_heads["level_3_two_wheelers_head"])+1}),  #4
    level_3_other_vehicles_head=dict(common_decode_head, **{"num_classes": len(class_hierarchy_heads["level_3_other_vehicles_head"])+1}), #5
    # Road head
    level_2_flat_head=dict(common_decode_head, **{"num_classes": len(class_hierarchy_heads["level_2_flat_head"])+1}),  #6
    level_3_normal_road_head=dict(common_decode_head, **{"num_classes": len(class_hierarchy_heads["level_3_normal_road_head"])+1}), #7
    level_3_road_marking_head=dict(common_decode_head, **{"num_classes": len(class_hierarchy_heads["level_3_road_marking_head"])+1}), #8
    # construction
    level_2_construction_head=dict(common_decode_head, **{"num_classes": len(class_hierarchy_heads["level_2_construction_head"])+1}), #9
    level_3_building_infra_head=dict(common_decode_head, **{"num_classes": len(class_hierarchy_heads["level_3_building_infra_head"])+1}), #10
    level_3_other_infra_head=dict(common_decode_head, **{"num_classes": len(class_hierarchy_heads["level_3_other_infra_head"])+1}), #11

    # Road objects: 11
    level_2_traffic_objects_head=dict(common_decode_head, **{"num_classes": len(class_hierarchy_heads["level_2_traffic_objects_head"])+1}), #12
    level_3_pole_all_head=dict(common_decode_head, **{"num_classes": len(class_hierarchy_heads["level_3_pole_all_head"])+1}),  #13
    level_3_sign_head=dict(common_decode_head, **{"num_classes": len(class_hierarchy_heads["level_3_sign_head"])+1}), #14
    level_3_light_head=dict(common_decode_head, **{"num_classes": len(class_hierarchy_heads["level_3_light_head"])+1}), #15
    level_3_other_road_objects_head=dict(common_decode_head, **{"num_classes": len(class_hierarchy_heads["level_3_other_road_objects_head"])+1}), #16
    # Nature objects: 16
    level_2_nature_objects_head=dict(common_decode_head, **{"num_classes": len(class_hierarchy_heads["level_2_nature_objects_head"])+1}), #17
    level_3_sky_vegetation_head=dict(common_decode_head, **{"num_classes": len(class_hierarchy_heads["level_3_sky_vegetation_head"])+1}), #18
    level_3_ground_head=dict(common_decode_head, **{"num_classes": len(class_hierarchy_heads["level_3_ground_head"])+1}), #19
    level_3_other_nature_objects_head=dict(common_decode_head, **{"num_classes": len(class_hierarchy_heads["level_3_other_nature_objects_head"])+1}), #20
    # VRU:20
    level_2_vru_head=dict(common_decode_head, **{"num_classes": len(class_hierarchy_heads["level_2_vru_head"])+1}), #21
    level_3_animal_head=dict(common_decode_head, **{"num_classes": len(class_hierarchy_heads["level_3_animal_head"])+1}), #22
    level_3_rider_head=dict(common_decode_head, **{"num_classes": len(class_hierarchy_heads["level_3_rider_head"])+1}), #23
    # Indoor objects 23
    level_2_indoor_objects_head=dict(common_decode_head, **{"num_classes": len(class_hierarchy_heads["level_2_indoor_objects_head"])+1, "channels":512}), #24
    level_3_furniture_head=dict(common_decode_head, **{"num_classes": len(class_hierarchy_heads["level_3_furniture_head"])+1}), #25
    level_3_bedroom_objects_head=dict(common_decode_head, **{"num_classes": len(class_hierarchy_heads["level_3_bedroom_objects_head"])+1}), #26
    level_3_bathroom_objects_head=dict(common_decode_head, **{"num_classes": len(class_hierarchy_heads["level_3_bathroom_objects_head"])+1}), #27
    level_3_other_indoor_objects_head=dict(common_decode_head, **{"num_classes": len(class_hierarchy_heads["level_3_other_indoor_objects_head"])+1}), #28
    level_3_electronics_head=dict(common_decode_head, **{"num_classes": len(class_hierarchy_heads["level_3_electronics_head"])+1}), #029
    level_3_kitchen_objects_head=dict(common_decode_head, **{"num_classes": len(class_hierarchy_heads["level_3_kitchen_objects_head"])+1}), #30
)



heads_hierarchy = [["level_1_head"],
                   ["level_2_vehicle_head", ["level_3_small_vehicles_head", "level_3_large_vehicles_head", "level_3_two_wheelers_head", "level_3_other_vehicles_head"]],
                   ["level_2_flat_head", ["level_3_normal_road_head", "level_3_road_marking_head"]],
                   ["level_2_construction_head", ["level_3_building_infra_head", "level_3_other_infra_head"]],
                   ["level_2_traffic_objects_head", ["level_3_pole_all_head", "level_3_sign_head", "level_3_light_head", "level_3_other_road_objects_head"]],
                   ["level_2_nature_objects_head", ["level_3_sky_vegetation_head", "level_3_ground_head", "level_3_other_nature_objects_head"]],
                   ["level_2_vru_head", ["level_3_animal_head", "level_3_rider_head"]],
                   ["level_2_indoor_objects_head", ["level_3_furniture_head", "level_3_bedroom_objects_head", "level_3_bathroom_objects_head", "level_3_other_indoor_objects_head",
                                                    "level_3_electronics_head", "level_3_kitchen_objects_head"]]
                  ]


checkpoint_file="/netscratch/gautam/semseg/exp_results/FMD7/training/20220726_235911/epoch_10.pth" ###NEW::
model = dict(
    type="HierarchicalSegmentor",
    backbone=dict(
        type="ResNetV1c",
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        multi_grid=(1, 2, 4),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style="pytorch",
        contract_dilation=True,
        #frozen_stages=4,        #### NEW:::::::::::Freezing backbone
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')
    ),
    levels_class_mapping=class_hierarchy_heads,
    heads_hierarchy=heads_hierarchy,
    decode_head=hierarchical_decode_heads_config,

    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)