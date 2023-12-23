import torch
import cv2
import numpy as np
import glob as glob
import os
from model import build_model
from torchvision import transforms

# Constants.
DATA_PATH = '/home/garamizo/Pictures/azul/boards2/classes'
IMAGE_SIZE = 34
DEVICE = 'cpu'
# Class names.
class_names = ['background', 'black', 'blue',
               'first', 'red', 'white', 'yellow']
# Load the trained model.
model = build_model(pretrained=False, fine_tune=False,
                    num_classes=len(class_names))
checkpoint = torch.load(
    './outputs/model_pretrained_True.pth', map_location=DEVICE)
print('Loading trained model weights...')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

os.makedirs('./outputs/inference', exist_ok=True)
# print(f"{DATA_PATH}/*/*.jpg")

# Get all the test image paths.
all_image_paths = glob.glob(f"{DATA_PATH}/*/*.jpg", recursive=True)
# Iterate over all the images and do forward pass.
for image_path in all_image_paths:
    # Get the ground truth class name from the image path.
    gt_class_name = image_path.split(os.path.sep)[-2]
    sample_name = image_path.split(os.path.sep)[-1]
    # Read the image and create a copy.
    image = cv2.imread(image_path)
    orig_image = image.copy()

    # Preprocess the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    image = transform(image)
    image = torch.unsqueeze(image, 0)
    image = image.to(DEVICE)

    # Forward pass throught the image.
    outputs = model(image)
    outputs = outputs.detach().numpy()
    pred_class_name = class_names[np.argmax(outputs[0])]
    print(f"GT: {gt_class_name}, Pred: {pred_class_name.lower()[:5]}")
    # Annotate the image with ground truth.
    # cv2.putText(
    #     orig_image, f"{gt_class_name}",
    #     (3, 15), cv2.FONT_HERSHEY_SIMPLEX,
    #     0.4, (0, 255, 0), 1, lineType=cv2.LINE_AA
    # )
    # Annotate the image with prediction.
    if gt_class_name.lower() != pred_class_name.lower():
        cv2.putText(
            orig_image, f"{pred_class_name.lower()[:5]}",
            (3, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.4, (100, 100, 225), 1, lineType=cv2.LINE_AA
        )
    # cv2.imshow('Result', orig_image)
    # cv2.waitKey(0)
    cv2.imwrite(f"./outputs/inference/{sample_name}", orig_image)
