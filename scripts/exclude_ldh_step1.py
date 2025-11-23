import json

def update_dataset_step1():
    with open('totalspineseg/resources/datasets/dataset_step1.json', 'r') as f:
        data = json.load(f)
    
    # 1. Remove LDH from labels
    if "LDH" in data['labels']:
        del data['labels']['LDH']
        
    # 2. Update regions_class_order (remove 10)
    current_order = data['regions_class_order']
    if 10 in current_order:
        current_order.remove(10)
    data['regions_class_order'] = current_order
    
    with open('totalspineseg/resources/datasets/dataset_step1.json', 'w') as f:
        json.dump(data, f, indent=4)

def update_map_step1():
    with open('totalspineseg/resources/labels_maps/nnunet_step1.json', 'r') as f:
        data = json.load(f)
    
    # Map 101 to 0 (background)
    data['101'] = 0
    
    with open('totalspineseg/resources/labels_maps/nnunet_step1.json', 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    update_dataset_step1()
    update_map_step1()
    print("Step 1 configuration updated: LDH (101) mapped to 0 and removed from training targets.")

