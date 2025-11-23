import json

def restore_ldh_map_step2():
    path = 'totalspineseg/resources/labels_maps/nnunet_step2.json'
    with open(path, 'r') as f:
        data = json.load(f)
    
    # Restore mapping: 101 -> 12
    data["101"] = 12
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    restore_ldh_map_step2()
    print("Restored '101': 12 mapping in nnunet_step2.json")

