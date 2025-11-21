import json

def revert_and_append_ldh_step1():
    # Original structure inferred
    data = {
        "channel_names": {
            "0": "MRI"
        },
        "labels": {
            "background": 0,
            "disc": [1,2,3,4,5],
            "disc_C2_C3": 2,
            "disc_C7_T1": 3,
            "disc_T12_L1": 4,
            "disc_L5_S": 5,
            "vertebrae": [6,7],
            "vertebrae_C1": 7,
            "canal": [8,9],
            "cord": 9,
            "LDH": 10
        },
        "regions_class_order": [1,2,3,4,5,6,7,8,9,10],
        "numTraining": 0,
        "file_ending": ".nii.gz"
    }
    
    with open('totalspineseg/resources/datasets/dataset_step1.json', 'w') as f:
        json.dump(data, f, indent=4)

def revert_and_append_ldh_step2():
    # Original structure inferred
    data = {
        "channel_names": {
            "0": "MRI",
            "1": "noNorm"
        },
        "labels": {
            "background": 0,
            "disc": [1,2,3,4,5],
            "disc_C2_C3": 2,
            "disc_C7_T1": 3,
            "disc_T12_L1": 4,
            "disc_L5_S": 5,
            "vertebrae": [6,7,8,9],
            "vertebrae_O": 7,
            "vertebrae_E": 8,
            "sacrum": 9,
            "canal": [10,11],
            "cord": 11,
            "LDH": 12
        },
        "regions_class_order": [1,2,3,4,5,6,7,8,9,10,11,12],
        "numTraining": 0,
        "file_ending": ".nii.gz"
    }
    
    with open('totalspineseg/resources/datasets/dataset_step2.json', 'w') as f:
        json.dump(data, f, indent=4)

def update_maps():
    # Step 1 Map
    # Original map inferred from previous read + adding LDH=10
    # Original: 64-95->1, 63->2, 71->3, 91->4, 100->5, 11-50->6, 2->8, 1->9
    map1 = {
        "64": 1, "65": 1, "66": 1, "67": 1, "72": 1, "73": 1, "74": 1, "75": 1, "76": 1, "77": 1, "78": 1, "79": 1, "80": 1, "81": 1, "82": 1, "92": 1, "93": 1, "94": 1, "95": 1,
        "63": 2,
        "71": 3,
        "91": 4,
        "100": 5,
        "11": 6, "12": 6, "13": 6, "14": 6, "15": 6, "16": 6, "17": 6, "21": 6, "22": 6, "23": 6, "24": 6, "25": 6, "26": 6, "27": 6, "28": 6, "29": 6, "30": 6, "31": 6, "32": 6, "41": 6, "42": 6, "43": 6, "44": 6, "45": 6, "50": 6,
        "2": 8,
        "1": 9,
        "101": 10
    }
    with open('totalspineseg/resources/labels_maps/nnunet_step1.json', 'w') as f:
        json.dump(map1, f, indent=4)

    # Step 2 Map
    # Original map inferred
    # 64-95 -> 1
    # 63 -> 2
    # 71 -> 3
    # 91 -> 4
    # 100 -> 5
    # ODD/EVEN vertebrae logic:
    # Even (12,14...) -> 7, Odd (11,13...) -> 6 (Wait, need to check original logic carefully or rely on previous file read)
    # Let's look at the diff or memory.
    # Previous read of nnunet_step2.json: 11->7, 12->8.
    # So: Odd->7, Even->8.
    # 50 -> 9 (sacrum)
    # 2 -> 10 (canal)
    # 1 -> 11 (cord)
    # LDH -> 12
    
    map2 = {
        "64": 1, "65": 1, "66": 1, "67": 1, "72": 1, "73": 1, "74": 1, "75": 1, "76": 1, "77": 1, "78": 1, "79": 1, "80": 1, "81": 1, "82": 1, "92": 1, "93": 1, "94": 1, "95": 1,
        "63": 2,
        "71": 3,
        "91": 4,
        "100": 5,
        "11": 7, "13": 7, "15": 7, "17": 7, "22": 7, "24": 7, "26": 7, "28": 7, "30": 7, "32": 7, "42": 7, "44": 7,
        "12": 8, "14": 8, "16": 8, "21": 8, "23": 8, "25": 8, "27": 8, "29": 8, "31": 8, "41": 8, "43": 8, "45": 8,
        "50": 9,
        "2": 10,
        "1": 11,
        "101": 12
    }
    with open('totalspineseg/resources/labels_maps/nnunet_step2.json', 'w') as f:
        json.dump(map2, f, indent=4)

if __name__ == "__main__":
    revert_and_append_ldh_step1()
    revert_and_append_ldh_step2()
    update_maps()
    print("JSONs reverted and updated.")

