import json

def update_step1_json():
    with open('totalspineseg/resources/datasets/dataset_step1.json', 'r') as f:
        data = json.load(f)
    
    # Reorder regions_class_order to put LDH (10) first
    current_order = data['regions_class_order'] # [1, 2, ..., 10]
    if 10 in current_order:
        current_order.remove(10)
        current_order.insert(0, 10) # [10, 1, 2, ..., 9]
    
    data['regions_class_order'] = current_order
    
    with open('totalspineseg/resources/datasets/dataset_step1.json', 'w') as f:
        json.dump(data, f, indent=4)

def update_step2_json():
    with open('totalspineseg/resources/datasets/dataset_step2.json', 'r') as f:
        data = json.load(f)
        
    # Reorder regions_class_order to put LDH (12) first
    current_order = data['regions_class_order'] # [1, 2, ..., 12]
    if 12 in current_order:
        current_order.remove(12)
        current_order.insert(0, 12) # [12, 1, 2, ..., 11]
        
    data['regions_class_order'] = current_order
    
    with open('totalspineseg/resources/datasets/dataset_step2.json', 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    update_step1_json()
    update_step2_json()
    print("JSONs updated: LDH is now first in regions_class_order.")

