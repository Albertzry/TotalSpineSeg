import json

def update_step1_map():
    with open('totalspineseg/resources/labels_maps/nnunet_step1.json', 'r') as f:
        data = json.load(f)
    
    new_data = {}
    # Add LDH
    new_data["101"] = 1
    
    for k, v in data.items():
        new_data[k] = v + 1
        
    with open('totalspineseg/resources/labels_maps/nnunet_step1.json', 'w') as f:
        json.dump(new_data, f, indent=4)

def update_step1_dataset():
    with open('totalspineseg/resources/datasets/dataset_step1.json', 'r') as f:
        data = json.load(f)
    
    labels = data['labels']
    new_labels = {"background": 0, "LDH": 1}
    
    for k, v in labels.items():
        if k == "background": continue
        if isinstance(v, int):
            new_labels[k] = v + 1
        elif isinstance(v, list):
            new_labels[k] = [x + 1 for x in v]
            
    data['labels'] = new_labels
    data['regions_class_order'] = list(range(1, 11))
    
    with open('totalspineseg/resources/datasets/dataset_step1.json', 'w') as f:
        json.dump(data, f, indent=4)

def update_step2_map():
    with open('totalspineseg/resources/labels_maps/nnunet_step2.json', 'r') as f:
        data = json.load(f)
        
    new_data = {}
    new_data["101"] = 1
    
    for k, v in data.items():
        new_data[k] = v + 1
        
    with open('totalspineseg/resources/labels_maps/nnunet_step2.json', 'w') as f:
        json.dump(new_data, f, indent=4)

def update_step2_dataset():
    with open('totalspineseg/resources/datasets/dataset_step2.json', 'r') as f:
        data = json.load(f)
        
    labels = data['labels']
    new_labels = {"background": 0, "LDH": 1}
    
    for k, v in labels.items():
        if k == "background": continue
        if isinstance(v, int):
            new_labels[k] = v + 1
        elif isinstance(v, list):
            new_labels[k] = [x + 1 for x in v]
            
    data['labels'] = new_labels
    data['regions_class_order'] = list(range(1, 13))
    
    with open('totalspineseg/resources/datasets/dataset_step2.json', 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    update_step1_map()
    update_step1_dataset()
    update_step2_map()
    update_step2_dataset()
    print("JSON files updated.")

