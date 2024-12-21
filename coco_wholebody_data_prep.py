# %%
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# %%
IMG_WIDTH = 192
IMG_HEIGHT = 256
N_DIM = 3 
N_BODY_KEYPOINTS = 17
N_FACE_KEYPOINTS = 68
N_HAND_KEYPOINTS = 21
N_FOOT_KEYPOINTS = 6
N_TOTAL_KEYPOINTS = N_BODY_KEYPOINTS + N_FACE_KEYPOINTS + 2 * N_HAND_KEYPOINTS + N_FOOT_KEYPOINTS

TRAIN_ANNOT_PATH = '/Users/harshagrawal/Downloads/COCO-Wholebody/coco_wholebody_train_v1.0.json' 
VAL_ANNOT_PATH = '/Users/harshagrawal/Downloads/COCO-Wholebody/coco_wholebody_val_v1.0.json' 
PATH_TRAIN = '/Users/harshagrawal/Downloads/COCO-Wholebody/train2017'
PATH_VAL = '/Users/harshagrawal/Downloads/COCO-Wholebody/val2017'

# %%
with open(TRAIN_ANNOT_PATH) as f:
    train_coco = json.load(f)

with open(VAL_ANNOT_PATH) as f:
    val_coco = json.load(f)

N_TOTAL_KEYPOINTS = len(train_coco['categories'][0]['keypoints'])
K_NAMES = train_coco['categories'][0]['keypoints']

# %%
class COCOWholeBodyDataset(Dataset):
    def __init__(self, annotations_file, transform=None):
        with open(annotations_file) as f:
            self.annotations = json.load(f)
        self.transform = transform
        self.k_names = self.annotations['categories'][0]['keypoints']

    def __len__(self):
        return len(self.annotations['annotations'])

    def __getitem__(self, idx):
        annotation = self.annotations['annotations'][idx]
        img_id = str(annotation['image_id']).zfill(12)  # Ensure image ID is zero-padded
        
        print(self.annotations['info']['description'])
        # Determine image path
        img_path = (f'{PATH_VAL}/{img_id}.jpg' if 'val' in self.annotations['info']['description'] 
                    else f'{PATH_TRAIN}/{img_id}.jpg')
        
        og_img = Image.open(img_path).convert('RGB')

        # Extract different keypoint types
        body_keypoints = np.array(annotation['keypoints']).reshape(N_BODY_KEYPOINTS, 3)
        foot_kpts = np.array(annotation['foot_kpts']).reshape(N_FOOT_KEYPOINTS, 3)
        face_kpts = np.array(annotation['face_kpts']).reshape(N_FACE_KEYPOINTS, 3)
        lefthand_kpts = np.array(annotation['lefthand_kpts']).reshape(N_HAND_KEYPOINTS, 3)
        righthand_kpts = np.array(annotation['righthand_kpts']).reshape(N_HAND_KEYPOINTS, 3)

        # Get bounding box
        bbox = annotation['bbox']
        
        # Adjust bounding box to include all keypoints
        bbox = self.check_keypoints_in_bbox(
            bbox, body_keypoints, foot_kpts, face_kpts, 
            lefthand_kpts, righthand_kpts, og_img
        )

        # Crop and resize image
        res_img = self.crop_resize_img(og_img, bbox)

        # Rescale keypoints
        body_keypoints[:, :2] = self.rescale_keypoints(body_keypoints[:, :2], bbox)
        foot_kpts[:, :2] = self.rescale_keypoints(foot_kpts[:, :2], bbox)
        face_kpts[:, :2] = self.rescale_keypoints(face_kpts[:, :2], bbox)
        lefthand_kpts[:, :2] = self.rescale_keypoints(lefthand_kpts[:, :2], bbox)
        righthand_kpts[:, :2] = self.rescale_keypoints(righthand_kpts[:, :2], bbox)

        # Combine all keypoints
        all_keypoints = np.vstack((body_keypoints, foot_kpts, face_kpts, lefthand_kpts, righthand_kpts))
        
        # Apply transforms if specified
        if self.transform:
            res_img = self.transform(res_img)

        return res_img, torch.tensor(all_keypoints[:, :2], dtype=torch.float32), torch.tensor(all_keypoints[:, 2], dtype=torch.float32)

    def check_keypoints_in_bbox(self, bbox, body_keypoints, foot_kpts, face_kpts, lefthand_kpts, righthand_kpts, og_img):
        bbox_x, bbox_y, bbox_w, bbox_h = bbox
        img_w, img_h = og_img.size
        
        x_min, y_min = bbox_x, bbox_y
        x_max, y_max = bbox_x + bbox_w, bbox_y + bbox_h
        
        all_keypoints = [body_keypoints, foot_kpts, face_kpts, lefthand_kpts, righthand_kpts]
        
        for keypoints in all_keypoints:
            for x, y, v in keypoints:
                if v > 0:  # Only consider visible or labeled keypoints
                    x_min = min(x_min, max(0, x - 10))
                    x_max = max(x_max, min(img_w, x + 10))
                    y_min = min(y_min, max(0, y - 10))
                    y_max = max(y_max, min(img_h, y + 10))
        
        # Ensure the bounding box is within image boundaries
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(img_w, x_max)
        y_max = min(img_h, y_max)
        
        return [x_min, y_min, x_max - x_min, y_max - y_min]

    def crop_resize_img(self, og_img, bbox):
        bbox_x, bbox_y, bbox_w, bbox_h = bbox
        res_img = og_img.resize((IMG_WIDTH, IMG_HEIGHT), box=(bbox_x, bbox_y, bbox_x+bbox_w, bbox_y+bbox_h))
        return res_img
    
    def rescale_keypoints(self, keypoints, bbox):
        bbox_x, bbox_y, bbox_w, bbox_h = bbox
        box_start_pos = np.array([bbox_x, bbox_y])
        box_size = np.array([bbox_w, bbox_h])
        res_size = np.array([IMG_WIDTH, IMG_HEIGHT])
        keypoints = np.round((keypoints - box_start_pos) * (res_size / box_size)).astype(int)
        keypoints[keypoints < 0] = 0
        return keypoints

# %%
data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Resize((224, 224)),
])

# %%
def validate_samples(dataset, num_samples=5):

    fig = plt.figure(figsize=(15, 3 * num_samples))
    
    for i in range(min(num_samples, len(dataset))):
        # Unpack the dataset itemx
        resized_img, keypoints, k_vis = dataset[i]
        
        # Get the original annotation
        annotation = dataset.annotations['annotations'][i]
        img_id = str(annotation['image_id']).zfill(12)
        
        # Create subplots for each sample
        ax1 = fig.add_subplot(num_samples, 2, 2*i + 1)
        ax2 = fig.add_subplot(num_samples, 2, 2*i + 2)
        
        # Plot resized image
        if torch.is_tensor(resized_img):
            resized_img = resized_img.permute(1, 2, 0)
        ax1.imshow(resized_img)
        ax1.set_title(f'Resized Image {img_id}')
        ax1.axis('off')
        
        # Plot keypoints on the image
        ax2.imshow(resized_img)
        ax2.set_title(f'Keypoints for Image {img_id}')
        
        # Separate keypoint types
        body_keypoints = keypoints[:N_BODY_KEYPOINTS]
        foot_kpts = keypoints[N_BODY_KEYPOINTS:N_BODY_KEYPOINTS+N_FOOT_KEYPOINTS]
        face_kpts = keypoints[N_BODY_KEYPOINTS+N_FOOT_KEYPOINTS:N_BODY_KEYPOINTS+N_FOOT_KEYPOINTS+N_FACE_KEYPOINTS]
        lefthand_kpts = keypoints[N_BODY_KEYPOINTS+N_FOOT_KEYPOINTS+N_FACE_KEYPOINTS:N_BODY_KEYPOINTS+N_FOOT_KEYPOINTS+N_FACE_KEYPOINTS+N_HAND_KEYPOINTS]
        righthand_kpts = keypoints[N_BODY_KEYPOINTS+N_FOOT_KEYPOINTS+N_FACE_KEYPOINTS+N_HAND_KEYPOINTS:]
        
        # Plot keypoints
        all_kpts = [body_keypoints, foot_kpts, face_kpts, lefthand_kpts, righthand_kpts]
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        
        for j, (kpts, color) in enumerate(zip(all_kpts, colors)):
            for k, (x, y) in enumerate(kpts):
                ax2.scatter(x, y, s=50, color=color)
                # Label body keypoints
                if j == 0 and k < len(dataset.k_names):
                    ax2.text(x + 2, y + 2, dataset.k_names[k], fontsize=8, 
                             bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
        
        ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

# %%
train_dataset = COCOWholeBodyDataset(TRAIN_ANNOT_PATH)
val_dataset = COCOWholeBodyDataset(VAL_ANNOT_PATH)

os.makedirs('datasets', exist_ok=True)

torch.save(train_dataset, 'datasets/train_dataset.pt')
torch.save(val_dataset, 'datasets/val_dataset.pt')

print("Training dataset")
validate_samples(train_dataset, num_samples=20)
print('Validation dataset:')
validate_samples(val_dataset, num_samples=20)


