import json

def revert_step1_json():
    with open('totalspineseg/resources/datasets/dataset_step1.json', 'r') as f:
        data = json.load(f)
    
    # Reorder regions_class_order to put LDH (10) last
    # Expected current: [10, 1, 2, ..., 9]
    # Target: [1, 2, ..., 9, 10]
    
    # Just recreate the list 1..10
    data['regions_class_order'] = list(range(1, 11))
    
    with open('totalspineseg/resources/datasets/dataset_step1.json', 'w') as f:
        json.dump(data, f, indent=4)

def revert_step2_json():
    with open('totalspineseg/resources/datasets/dataset_step2.json', 'r') as f:
        data = json.load(f)
        
    # Reorder regions_class_order to put LDH (12) last
    # Expected current: [12, 1, 2, ..., 11]
    # Target: [1, 2, ..., 11, 12]
    
    # Just recreate the list 1..12
    data['regions_class_order'] = list(range(1, 13))
    
    with open('totalspineseg/resources/datasets/dataset_step2.json', 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    revert_step1_json()
    revert_step2_json()
    print("JSONs updated: LDH is back to last in regions_class_order.")

