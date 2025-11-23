import json

def restore_ldh_step2():
    with open('totalspineseg/resources/datasets/dataset_step2.json', 'r') as f:
        data = json.load(f)
    
    # 1. Add LDH to labels if missing
    if "LDH" not in data['labels']:
        data['labels']['LDH'] = 12
        
    # 2. Add 12 to regions_class_order if missing
    current_order = data['regions_class_order']
    if 12 not in current_order:
        current_order.append(12) # Append to end
    data['regions_class_order'] = current_order
    
    with open('totalspineseg/resources/datasets/dataset_step2.json', 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    restore_ldh_step2()
    print("Step 2 configuration updated: LDH (12) restored.")

