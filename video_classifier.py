import os
import sys
from pathlib import Path
ROOT = str(Path(__file__).parent.absolute())
sys.path.append(os.path.join(ROOT,'core'))
import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms

from PIL import Image
import torch
import cv2
import numpy as np
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional

def improved_bbox_extraction(
        attention_map: np.ndarray,
        method: str = 'best_region',
        percentile_threshold: float = 80
    ) -> Tuple[int, int, int, int]:

    H, W = attention_map.shape
    threshold = np.percentile(attention_map, percentile_threshold)

    if method == "best_region":
        from skimage.measure import label, regionprops
        mask = attention_map >= threshold
        # Label connected regions
        labeled = label(mask)

        # Find the best region
        best_region = None
        best_score = -np.inf

        for region in regionprops(labeled, intensity_image=attention_map):
            # Example score: total sum of heatmap values inside region
            score = region.intensity_image[region.image].sum()
            if score > best_score:
                best_score = score
                best_region = region

        if best_region is not None:
            # Bounding box: min_row, min_col, max_row, max_col
            min_row, min_col, max_row, max_col = best_region.bbox
            bbox = (min_col, min_row, max_col, max_row)

            return bbox

class VideoClasifier:
    def __init__(self,model_name):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = pe.CLIP.from_config(model_name, pretrained=True)
        self.patch_shape = (14,14)
        self.model = model.to(self.device)
        self.frame_bank_size = 8
        self.preprocess = transforms.get_image_transform(model.image_size)
        self.tokenizer = transforms.get_text_tokenizer(model.context_length)
        self.labels = None
        self.text_label = None
        self.frames_bank = []
        self.patch_bank = []
        self.latest_image_tensor = None

    def encode_labes(self, labels:List[str]):
        self.text_label = labels
        tokenized_labels = self.tokenizer(labels).to(self.device)
        with torch.no_grad():
            encoded_labels = self.model.encode_text(tokenized_labels)
            self.labels = F.normalize(encoded_labels, dim=-1)
    
    def process_frame(self,frame:np.ndarray):
        rgb_image = Image.fromarray(frame)
        image_tensors = self.preprocess(rgb_image)
        image_tensors = image_tensors.unsqueeze(0).to(self.device)
        frame_features, patch = self.model.encode_image(image_tensors)
        self.update_frames_bank(frame_features,patch)
        
    def get_label_propabibility(self):
        # (b, n_frames, emb_dim)
        frames_tensors = torch.cat(self.frames_bank,dim=0).unsqueeze(0).mean(dim=1)
        image_features_norm = F.normalize(frames_tensors, dim=-1)

        logits = 100.0 * image_features_norm @ self.labels.T
        probabilities = F.softmax(logits, dim=-1).cpu().numpy()[0]
        predicted_class = int(probabilities.argmax())
        confidence = float(probabilities[predicted_class])
        return predicted_class, confidence, probabilities
    
    def get_attention_maps_box(self, predicted_class,input_shape):
        w,h = input_shape

        orignal_patch_shape = self.patch_bank[0].size()
        num_frames = len(self.patch_bank)
        patch_features = torch.cat(self.patch_bank,dim=0)
        patch_features = patch_features.reshape(num_frames,1024,-1).permute(0,2,1)
        patch_mean = patch_features.mean(dim=0,keepdim=True)
        patch_norm = F.normalize(patch_mean,dim=-1)
        attention_frame_text = 100.0 * patch_norm @ self.labels.T # size will be 1, patch_size**2, text_class
        attention_frame_text = F.softmax(attention_frame_text,dim=-1)

        attention_frame_text = attention_frame_text.squeeze()
        selected_attention = attention_frame_text.permute(1,0)[predicted_class].reshape(orignal_patch_shape[2:]).cpu().numpy()
        selected_attention_resized = cv2.resize(
                selected_attention, (w, h), interpolation=cv2.INTER_CUBIC
            )
        
        bbox = improved_bbox_extraction(
                selected_attention_resized, 
                percentile_threshold=75, # Higher threshold for more precise localization
                method="best_region"
            )
        
        heatmap_colored = cv2.applyColorMap(
            (selected_attention_resized * 255).astype(np.uint8), 
            cv2.COLORMAP_JET  # Better colormap for attention
            )
        heatmap_colored = cv2.cvtColor(heatmap_colored,cv2.COLOR_BGR2RGB)

        return heatmap_colored, bbox

    def update_frames_bank(self,new_frames, new_patch):
        
        self.frames_bank.append(new_frames)
        self.patch_bank.append(new_patch)
        # Clean unuseful data
        if len(self.frames_bank) > self.frame_bank_size:
            self.frames_bank.pop(0)
            self.patch_bank.pop(0)
        
    def classify_video(self, frame:np.ndarray):
        H,W,_ = frame.shape
        with torch.no_grad():
            self.process_frame(frame)
            predicted_class, confidence, all_confidence = self.get_label_propabibility()
            heatmap, bbox = self.get_attention_maps_box(predicted_class,(W,H))
            
        results = {
            'predicted_class': predicted_class,
            'class_name': self.text_label[predicted_class],
            'confidence': confidence,
            'all_probabilities': all_confidence.tolist(),
            'bounding_box': bbox,  # (x1, y1, x2, y2)
            "heatmap": heatmap
        }

        return results

        



