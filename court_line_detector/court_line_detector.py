import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchvision.models as models
import cv2

class CourtLineDetector:
    def __init__(self, model_path):
        self.model = models.resnet50(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 14 * 2)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))

        self.transforms = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)), # Resize images to 224x224 for ResNet input
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transforms(img_rgb).unsqueeze(0) # unsqueeze: [[img]] -> [img]

        with torch.no_grad():
            outputs = self.model(image_tensor)
        
        keypoints = outputs.squeeze().cpu().numpy() # [28] -> (14, 2)
        original_h, original_w = img_rgb.shape[:2]

        keypoints[::2] *= original_w / 224.0 # Scale x-coordinates
        keypoints[1::2] *= original_h / 224.0 # Scale y

        return keypoints
    
    def draw_keypoints(self, image, keypoints):
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i + 1])
            
            cv2.putText(image, str(i//2), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        return image
    
    def draw_keypoints_on_video(self, video_frames, keypoints):
        output_video_frames = []
        for frame in video_frames:
            output_video_frames.append(self.draw_keypoints(frame, keypoints))
        return output_video_frames