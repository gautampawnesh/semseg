from src.models.segmentor.hierarchical_segmentor import HierarchicalSegmentor
import pytest
import torch
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC

CLASS_MAPPING = dict(
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


MVP2_class_hierarchy_heads = dict(
    level_1_head=dict(
        # ignore=[123],
        objects=[107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 105, 106, 96, 97, 98, 99, 100, 101, 102, 103, 90,
                 91, 92, 93, 94, 95, 78, 79, 80, 81, 82, 83, 85, 86, 87, 88],
        construction=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
        marking=[38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 35, 36, 37],
        nature=[59, 60, 61, 62, 63, 64, 65],
        void=[118, 119, 120, 121, 122],
        human=[30, 31, 32, 33, 34],
        animal=[0, 1],
        banner=[66],
        bench=[67],
        bike_rack=[68],
        catch_basin=[69],
        cctv_camera=[70],
        fire_hydrant=[71],
        junction_box=[72],
        mail_box=[73],
        manhole=[74],
        parking_meter=[75],
        phone_booth=[76],
        pothole=[77],
        street_light=[84],
        traffic_cone=[89],
        trash_can=[104],
        water_valve=[117]
    ),
    # objects head
    level_2_objects_head=dict(
        vehicles=[107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 105, 106],
        traffic_sign=[96, 97, 98, 99, 100, 101, 102, 103],
        sign=[78, 79, 80, 81, 82, 83],
        traffic_light=[90, 91, 92, 93, 94, 95],
        support=[85, 86, 87, 88]
    ),
    level_3_vehicles_head=dict(
        bus=[107],
        car=[108],
        caravan=[109],
        motorcycle=[110],
        on_rails=[111],
        other_vehicle=[112],
        trailer=[113],
        truck=[114],
        vehicle_group=[115],
        wheeled_slow=[116],
        bicycle=[105],
        boat=[106]
    ),
    level_3_traffic_sign_head=dict(
        ts_amg=[96],
        ts_back=[97],
        ts_dback=[98],
        ts_dfront=[99],
        ts_front=[100],
        ts_parking=[101],
        ts_tempback=[102],
        ts_tempfront=[103]
    ),
    level_3_signs_head=dict(
        sign_ad=[78],
        sign_amg=[79],
        sign_back=[80],
        sign_info=[81],
        sign_other=[82],
        sign_store=[83]
    ),
    level_3_traffic_light_head=dict(
        tl_general=[90],
        tl_pede=[91],
        tl_gup=[92],
        tl_gh=[93],
        tl_cycl=[94],
        tl_other=[95]
    ),
    level_3_support_head=dict(
        pole=[85],
        pole_group=[86],
        tf_signframe=[87],
        utility_pole=[88]
    ),
    # Construction head
    level_2_flat_head=dict(
        flat=[12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
        barrier=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        structure=[26, 27, 28, 29]
    ),
    level_3_flat_head=dict(
        wall=[12],
        bike_lane=[13],
        cross_plain=[14],
        curb_cut=[15],
        driveway=[16],
        parking=[17],
        parking_aisle=[18],
        ped_area=[19],
        rail_track=[20],
        road=[21],
        road_shoulder=[22],
        service_lane=[23],
        sidewalk=[24],
        traffic_island=[25]
    ),
    level_3_barrier_head=dict(
        amg_barrier=[2],
        con_block=[3],
        curb=[4],
        fence=[5],
        gaurd_rail=[6],
        barrier=[7],
        road_median=[8],
        road_side=[9],
        lane_sep=[10],
        temp_barrier=[11]
    ),
    level_3_structure_head=dict(
        bridge=[26],
        building=[27],
        garage=[28],
        tunnel=[29]
    ),
    # marking
    level_2_marking_head=dict(
        marking_discrete=[38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
        marking_only=[55, 56, 57, 58],
        marking_continuous=[35, 36, 37]
    ),
    level_3_marking_discrete_head=dict(
        lm_amg=[38],
        lm_arrl=[39],
        lm_arro=[40],
        lm_arrr=[41],
        lm_arrsl=[42],
        lm_arrsr=[43],
        lm_arrs=[44],
        lm_cross=[45],
        lm_gw_r=[46],
        lm_gw_s=[47],
        lm_hatched=[48],
        lm_hatached_d=[49],
        lm_other=[50],
        lm_stop=[51],
        lm_symb_b=[52],
        lm_sym_o=[53],
        lm_text=[54]
    ),
    level_3_marking_only_head=dict(
        lm_dash=[55],
        lm_cross=[56],
        lm_other=[57],
        lm_test=[58]
    ),
    level_3_marking_continuous_head=dict(
        lm_dashed=[35],
        lm_straight=[36],
        lm_zigzag=[37]
    ),

    # Nature objects
    level_2_nature_head=dict(
        mountain=[59],
        sand=[60],
        sky=[61],
        snow=[62],
        terrain=[63],
        vege=[64],
        water=[65]
    ),
    # void
    level_2_void_head=dict(
        car_mount=[118],
        dynamic=[119],
        ego=[120],
        ground=[121],
        static=[122]
    ),
    # Humans objects
    level_2_humans_head=dict(
        person=[30, 31],
        rider=[32, 33, 34]
    ),
    level_3_person_head=dict(
        person_group=[30],
        individual=[31]
    ),
    level_3_rider_head=dict(
        bicyclist=[32],
        motorcyclist=[33],
        other_rider=[34]
    ),
    level_2_animal_head=dict(
        bird=[0],
        animal=[1]
    )
)

MVP2_heads_hierarchy = [["level_1_head"],
                   ["level_2_objects_head", ["level_3_vehicles_head", "level_3_traffic_sign_head", "level_3_signs_head", "level_3_traffic_light_head", "level_3_support_head"]],
                   ["level_2_flat_head", ["level_3_flat_head", "level_3_barrier_head", "level_3_structure_head"]],
                   ["level_2_marking_head", ["level_3_marking_discrete_head", "level_3_marking_only_head", "level_3_marking_continuous_head"]],
                   ["level_2_nature_head", []],
                   ["level_2_void_head", []],
                   ["level_2_humans_head", ["level_3_person_head", "level_3_rider_head"]],
                   ["level_2_animal_head", []]
                  ]

def test_remapped_gt_semantic_seg(mocker):
    """test the generated gt_maps for each heads"""
    mocker.patch("mmseg.models.segmentors.encoder_decoder.EncoderDecoder.__init__", return_value=None)
    h_segmentor = HierarchicalSegmentor({}, CLASS_MAPPING, heads_hierarchy, {})
    dummy_gt_mask = torch.randint(0, 190, (2, 1, 4, 4))
    # print(dummy_gt_mask[0, :, :, :])
    # print(dummy_gt_mask[1, :, :, :])
    output_gt_maps = h_segmentor.remapped_gt_semantic_seg(dummy_gt_mask)

    # print(output_gt_maps[0])
    # level 1 :
    gt_flat = dummy_gt_mask.flatten().tolist()
    expected_outputs = []
    for indx, each_head in enumerate(h_segmentor.heads_hierarchy_flat):
        expected_output = []
        level_values = list(CLASS_MAPPING[each_head].values())
        for each_l in gt_flat:
            assigned = False
            for cls_ind, value in enumerate(level_values):
                if each_l in value:
                    expected_output.append(cls_ind+1)
                    assigned = True
            if not assigned:
                expected_output.append(0)
        expected_outputs.append(expected_output)

    for i, name in enumerate(h_segmentor.heads_hierarchy_flat):
        print(f"{name} : \n input:   {gt_flat} \n output:   {output_gt_maps[i].flatten().tolist()} \n expected: {expected_outputs[i]}")
        assert output_gt_maps[i].flatten().tolist() == expected_outputs[i]
    # To print print statement : uncomment below line : ugly way :(
    # assert output_gt_maps[0].flatten().tolist() != expected_outputs[0]


def test_seg_preds_to_original(mocker):
    mocker.patch("mmseg.models.segmentors.encoder_decoder.EncoderDecoder.__init__", return_value=None)
    h_segmentor = HierarchicalSegmentor({}, CLASS_MAPPING, heads_hierarchy, {})
    dummy_l1_pred = torch.tensor([[[1, 2, 1, 4], [5, 6, 1, 6]]])
    dummy_l2_veh_pred = torch.tensor([[[3, 0, 1, 2], [0, 0, 4, 0]]])
    dummy_l3_small_pred = torch.tensor([[[1, 2, 3, 4], [4, 3, 2, 1]]])
    dummy_large_pred = torch.tensor([[[1, 2, 3, 0], [1, 2, 3, 0]]])
    dummy_two_w_pred = torch.tensor([[[1, 2, 0, 0], [1, 2, 0, 0]]])
    dummy_other_w_pred = torch.tensor([[[1, 2, 3, 4], [1, 5, 0, 6]]])
    dummy_lst = [dummy_l1_pred, dummy_l2_veh_pred, dummy_l3_small_pred, dummy_large_pred, dummy_two_w_pred, dummy_other_w_pred]
    dummy_level2_vehicle_heads_out = torch.stack(dummy_lst)
    dummy_predictions = torch.randint(0, 3, (25, 1, 2, 4))
    dummy_predictions = torch.cat((dummy_level2_vehicle_heads_out, dummy_predictions))
    output = h_segmentor.seg_preds_to_original(dummy_predictions)
    print(output)
    assert output[0, 0, 0] == 8
    assert output[0, 0, 2] == 182
    assert output[0, 1, 2] == 0

def test_remapped_gt_semantic_seg_and_seg_preds_to_original(mocker):
    mocker.patch("mmseg.models.segmentors.encoder_decoder.EncoderDecoder.__init__", return_value=None)
    h_segmentor = HierarchicalSegmentor({}, CLASS_MAPPING, heads_hierarchy, {})
    dummy_gt_mask_ts = np.loadtxt("./tests/sample_uni_label.txt", dtype=int)
    dummy_gt_mask = torch.from_numpy(dummy_gt_mask_ts).type(torch.LongTensor)
    output_gt_maps = h_segmentor.remapped_gt_semantic_seg(dummy_gt_mask)
    output = h_segmentor.seg_preds_to_original(output_gt_maps)
    print(output.shape)
    print(dummy_gt_mask.shape, type(dummy_gt_mask))
    #np.savetxt("output_pred.txt", output.cpu().detach().numpy(), fmt="%i")
    assert torch.equal(output, dummy_gt_mask) == True


def test_mvp2_remapped_gt_semantic_seg_and_seg_preds_to_original(mocker):
    mocker.patch("mmseg.models.segmentors.encoder_decoder.EncoderDecoder.__init__", return_value=None)
    h_segmentor = HierarchicalSegmentor({}, MVP2_class_hierarchy_heads, MVP2_heads_hierarchy, {})
    dummy_gt_mask_ts = np.loadtxt("./tests/sample_mvp2_label.txt", dtype=int)
    dummy_gt_mask = torch.from_numpy(dummy_gt_mask_ts).type(torch.LongTensor)
    output_gt_maps = h_segmentor.remapped_gt_semantic_seg(dummy_gt_mask)
    output = h_segmentor.seg_preds_to_original(output_gt_maps)
    print(output.shape)
    print(dummy_gt_mask.shape, type(dummy_gt_mask))
    #np.savetxt("output_pred.txt", output.cpu().detach().numpy(), fmt="%i")
    assert torch.equal(output, dummy_gt_mask) == True