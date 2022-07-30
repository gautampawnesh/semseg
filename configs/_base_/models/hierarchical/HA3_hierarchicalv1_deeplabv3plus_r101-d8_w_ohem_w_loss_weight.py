
"""
MeletisDubblemann
Introduced Weights for l1 and l2& l3 as 1:0.1:0.1
ignore_index=0 in loss computation.
"""
norm_cfg = dict(type="SyncBN", requires_grad=True)



class_hierarchy_heads = dict(
    level_1_head=dict(
        # ignore=[0, 184],
        vehicles=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 181, 182, 186],  # 15
        flat=[13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 44, 47, 53, 97, 99, 128, 180, 190],  # 19
        construction=[27, 28, 29, 30, 31, 32, 33, 74, 89, 93, 96, 98, 104, 121, 126, 130, 135, 136, 138, 152, 167], # 21
        traffic_objects=[38, 39, 124, 40, 43, 45, 55, 185, 154, 71, 134, 42, 41, 123, 46, 54, 49, 48, 50, 52, 61], #21
        nature_objects=[183, 115, 36, 177, 35, 37, 187, 188, 112, 78, 82, 25, 24, 26, 105, 157, 75, 145, 141, 34], # 20
        vru_objects=[189, 60, 57, 58, 59, 56], #6
        indoor_objects=[64, 66, 67, 68, 70, 72, 73, 79, 81, 90, 91, 83, 108, 131, 142, 179, 106, 88, 118, 148, 100, 80, 51, 65,
                        86, 102, 84, 77, 76, 69, 110, 107, 160, 129, 94, 63, 62, 163,
                        85, 109, 122, 171, 178,
                        176, 175, 174, 172, 173, 169, 166, 165, 164, 156, 147, 146, 144, 143, 140, 137, 125, 111, 101, 150, 153, 161, 87,
                        170, 168, 159, 127, 120, 117, 103,
                        162, 158, 155, 151, 149, 139, 133, 132, 119, 116, 114, 113, 95, 92] # 87
    ),
    # Vehicle head
    level_2_vehicle_head=dict(
        small_vehicles=[1, 181, 182, 9],
        large_vehicles=[4, 5, 3],
        two_wheelers=[8, 2],
        other_vehicles=[6, 7, 11, 10, 12, 186],
    ),
    level_3_small_vehicles_head=dict(
        car=[1],
        pickup_truck=[181],
        van=[182],
        ego_vehicle=[9],
    ),
    level_3_large_vehicles_head=dict(
        truck=[4],
        train=[5],
        bus=[3]
    ),
    level_3_two_wheelers_head=dict(
        motorcycle=[8],
        bicycle=[2],
    ),
    level_3_other_vehicles_head=dict(
        caravan=[6],
        trailer=[7],
        boat=[11],
        wheeled_slow=[10],
        other_vehicle=[12],
        plane=[186]
    ),
    # Road head
    level_2_flat_head=dict(
        normal_road=[13, 15, 16, 14, 21, 17, 128, 99, 18, 19, 20, 97, 190],
        road_marking=[22, 23, 44, 47, 53, 180]
    ),
    level_3_normal_road_head=dict(
        road=[13],
        parking=[15],
        rail_track=[16],
        sidewalk=[14],
        curb=[21],
        pedestrian_area=[17],
        dirt_track=[128],
        runway=[99],
        crosswalk_plain=[18],
        bikelane=[19],
        service_lane=[20],
        path=[97],
        curb_cut=[190]
    ),
    level_3_road_marking_head=dict(
        general_marking=[22],
        zebra_marking=[23],
        manhole=[44],
        catch_basin=[47],
        pothole=[53],
        all_road_marking=[180]
    ),
    # construction
    level_2_construction_head=dict(
        building_infra=[27, 96, 98, 74, 152, 135, 104, 93, 89, 29, 30, 31, 28, 32],
        other_infra=[167, 138, 136, 130, 33, 126, 121]
    ),
    level_3_building_infra_head=dict(
        building=[27],
        grandstand=[96],
        stairs=[98],
        house=[74],
        step_stair=[152],
        stage=[135],
        stairway=[104],
        skyscraper=[93],
        column=[89],
        wall=[29],
        bridge=[30],
        tunnel=[31],
        fence=[28],
        guard_rail=[32]
    ),
    level_3_other_infra_head=dict(
        pier_dock=[167],
        canopy=[138],
        fountain=[136],
        bannister=[130],
        barrier=[33],
        awning_sunshade=[126],
        hovel_hut=[121]
    ),

    # Road objects
    level_2_traffic_objects_head=dict(
        pole_all=[38, 39, 124],
        sign=[40, 43, 45, 55, 185, 154, 71, 134],
        light=[42, 41, 123],
        other_road_objects=[46, 54, 49, 48, 50, 52, 61]
    ),
    level_3_pole_all_head=dict(
        pole=[38],
        utility_pole=[39],
        tower=[124]
    ),
    level_3_sign_head=dict(
        traffic_sign=[40],
        billboard=[43],
        banner=[45],
        traffic_sign_frame=[55],
        traffic_sign_back=[185],
        trade_brand=[154],
        picture=[71],
        poster=[134]
    ),
    level_3_light_head=dict(
        street_light=[42],
        traffic_light=[41],
        light_source=[123]
    ),
    level_3_other_road_objects_head=dict(
        trashcan=[46],
        phone_booth=[54],
        cctv_camera=[49],
        junction_box=[48],
        fyre_hydrant=[50],
        mailbox=[52],
        bikerack=[61]
    ),
    # Nature objects
    level_2_nature_objects_head=dict(
        sky_vegetation=[183, 115, 36, 177, 34],
        ground=[35, 37, 187, 188, 112, 78, 82],
        other_nature_objects=[25, 24, 26, 105, 157, 75, 145, 141]
    ),
    level_3_sky_vegetation_head=dict(
        tree=[183],
        palm_tree=[115],
        sky=[36],
        plant=[177],
        vegetation=[34]
    ),
    level_3_ground_head=dict(
        terrain=[35],
        mountain=[37],
        earth_ground=[187],
        soil_ground=[188],
        hill=[112],
        field=[78],
        rock=[82]
    ),
    level_3_other_nature_objects_head=dict(
        water=[25],
        snow=[24],
        sand=[26],
        river=[105],
        lake=[157],
        sea=[75],
        water_fall=[145],
        swimming_pool=[141]
    ),
    # VRU
    level_2_vru_head=dict(
        animal=[60, 189],
        rider=[56, 57, 58, 59]
    ),
    level_3_animal_head=dict(
        animal=[60],
        bird=[189]
    ),
    level_3_rider_head=dict(
        person=[56],
        motorcyclist=[57],
        bicyclist=[58],
        other_rider=[59]
    ),
    # Indoor objects
    level_2_indoor_objects_head=dict(
        furniture=[64, 66, 67, 68, 70, 72, 73, 79, 81, 90, 91, 83, 108, 131, 142, 179, 106, 88, 118, 148, 100, 80, 51, 65],
        bedroom_objects=[86, 102, 84, 77, 76, 69, 110, 107, 160, 129, 94, 63, 62, 163],
        bathroom_objects=[85, 109, 122, 171, 178],
        other_indoor_objects=[176, 175, 174, 172, 173, 169, 166, 165, 164, 156, 147, 146, 144, 143, 140, 137, 125, 111, 101, 150, 153, 161, 87],
        electronics=[170, 168, 159, 127, 120, 117, 103],
        kitchen_objects=[162, 158, 155, 151, 149, 139, 133, 132, 119, 116, 114, 113, 95, 92]
    ),
    level_3_furniture_head=dict(
        bed=[64],
        cabinet=[66],
        door=[67],
        table=[68],
        chair=[70],
        sofa=[72],
        shelf=[73],
        armchair=[79],
        desk=[81],
        drawers=[90],
        counter=[91],
        wardrobe=[83],
        coffee_table=[108],
        ottoman=[131],
        stool=[142],
        other_furniture=[179],
        book_case=[106],
        box=[88],
        swivel_chair=[118],
        cradle=[148],
        case_showcase=[100],
        seat=[80],
        bench=[51],
        window=[65],
    ),
    level_3_bedroom_objects_head=dict(
        cushion=[86],
        pillow=[102],
        lamp=[84],
        rug=[77],
        mirror=[76],
        curtain=[69],
        flower=[110],
        blind=[107],
        blanket_cover=[160],
        apparel=[129],
        fireplace=[94],
        ceiling=[63],
        floor=[62],
        sconce=[163],
    ),
    level_3_bathroom_objects_head=dict(
        bathtub=[85],
        toilet=[109],
        towel=[122],
        shower=[171],
        shower_curtain=[178],
    ),
    level_3_other_indoor_objects_head=dict(
        escalator=[176],
        flag=[175],
        clock=[174],
        radiator=[172],
        glass_drinking=[173],
        plate=[169],
        fan=[166],
        tray=[165],
        vase=[164],
        pot=[156],
        bag=[147],
        tent=[146],
        basket=[144],
        barrel=[143],
        playing_toy=[140],
        conveyerbelt=[137],
        chandelier=[125],
        book=[111],
        billard=[101],
        ball=[150],
        tank=[153],
        sculpture=[161],
        base=[87]
    ),
    level_3_electronics_head=dict(
        monitor=[170],
        crt_screen=[168],
        screen_projector=[159],
        television=[127],
        arcade_machine=[120],
        computer=[117],
        screen=[103],
    ),
    level_3_kitchen_objects_head=dict(
        hood_exhaust=[162],
        dish_washer=[158],
        microwave=[155],
        food=[151],
        oven=[149],
        washer=[139],
        buffet=[133],
        bottle=[132],
        bar=[119],
        kitchen_island=[116],
        stove=[114],
        counter_top=[113],
        refrigerator=[95],
        sink=[92],
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

    # Road objects
    level_2_traffic_objects_head=dict(common_decode_head, **{"num_classes": len(class_hierarchy_heads["level_2_traffic_objects_head"])+1}), #12
    level_3_pole_all_head=dict(common_decode_head, **{"num_classes": len(class_hierarchy_heads["level_3_pole_all_head"])+1}),  #13
    level_3_sign_head=dict(common_decode_head, **{"num_classes": len(class_hierarchy_heads["level_3_sign_head"])+1}), #14
    level_3_light_head=dict(common_decode_head, **{"num_classes": len(class_hierarchy_heads["level_3_light_head"])+1}), #15
    level_3_other_road_objects_head=dict(common_decode_head, **{"num_classes": len(class_hierarchy_heads["level_3_other_road_objects_head"])+1}), #16
    # Nature objects
    level_2_nature_objects_head=dict(common_decode_head, **{"num_classes": len(class_hierarchy_heads["level_2_nature_objects_head"])+1}), #17
    level_3_sky_vegetation_head=dict(common_decode_head, **{"num_classes": len(class_hierarchy_heads["level_3_sky_vegetation_head"])+1}), #18
    level_3_ground_head=dict(common_decode_head, **{"num_classes": len(class_hierarchy_heads["level_3_ground_head"])+1}), #19
    level_3_other_nature_objects_head=dict(common_decode_head, **{"num_classes": len(class_hierarchy_heads["level_3_other_nature_objects_head"])+1}), #20
    # VRU
    level_2_vru_head=dict(common_decode_head, **{"num_classes": len(class_hierarchy_heads["level_2_vru_head"])+1}), #21
    level_3_animal_head=dict(common_decode_head, **{"num_classes": len(class_hierarchy_heads["level_3_animal_head"])+1}), #22
    level_3_rider_head=dict(common_decode_head, **{"num_classes": len(class_hierarchy_heads["level_3_rider_head"])+1}), #23
    # Indoor objects
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
        frozen_stages=4,        #### NEW:::::::::::Freezing backbone
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