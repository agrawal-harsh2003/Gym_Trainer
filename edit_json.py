import json

# Open the JSON file for reading
with open('/Users/harshagrawal/Downloads/COCO-Wholebody/coco_wholebody_train_v1.0.json', 'r') as f:
    gian = json.load(f)

# Modify the data (this is an example, you can modify it as needed)
gian['info']['description'] = 'train'

# Write the updated data back to the JSON file
with open('/Users/harshagrawal/Downloads/COCO-Wholebody/coco_wholebody_train_v1.0.json', 'w') as f:
    json.dump(gian, f, indent=4)