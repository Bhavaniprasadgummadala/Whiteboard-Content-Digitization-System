#download the dependies 
!pip install transformers datasets pytorch-lightning torchvision -q
!pip install easyocr opencv-python-headless numpy Pillow matplotlib
!pip install datasets transformers pytorch-lightning torchvision



import pandas as pd
import requests
import os
from PIL import Image
from io import BytesIO
from datasets import Dataset
from torchvision import transforms
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Load the CSV with correct encoding
df = pd.read_csv("/content/train_datset.csv", encoding="ISO-8859-1")

# Rename for convenience
df = df.rename(columns={"Upload the image here": "image_link", "output": "text"})

# Convert Google Drive shareable links to direct image links
def convert_to_direct_link(share_url):
    file_id = share_url.split("id=")[-1]
    return f"https://drive.google.com/uc?export=download&id={file_id}"

df["direct_link"] = df["image_link"].apply(convert_to_direct_link)

# Download images and keep paths
os.makedirs("whiteboard_images", exist_ok=True)
image_paths = []

for idx, row in df.iterrows():
    try:
        response = requests.get(row["direct_link"])
        img = Image.open(BytesIO(response.content)).convert("RGB")
        path = f"whiteboard_images/image_{idx}.png"
        img.save(path)
        image_paths.append(path)
    except:
        image_paths.append(None)

df["image_path"] = image_paths
df = df.dropna()

# Prepare Dataset
dataset = Dataset.from_pandas(df[["image_path", "text"]])

# Load model and processor
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# Image transform
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor()
])

def preprocess(example):
    image = Image.open(example["image_path"]).convert("RGB")
    pixel_values = processor(images=transform(image), return_tensors="pt").pixel_values[0]
    labels = processor.tokenizer(example["text"], return_tensors="pt", padding="max_length", truncation=True).input_ids[0]
    labels[labels == processor.tokenizer.pad_token_id] = -100  # Ignore pad tokens
    return {"pixel_values": pixel_values, "labels": labels}

dataset = dataset.map(preprocess)

# DataLoader
dataloader = DataLoader(dataset, batch_size=2)

# Training (1 epoch just for demo)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

model.train()
for epoch in range(1):  # Change to more epochs for better training
    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Loss: {loss.item():.4f}")


!pip install easyocr opencv-python-headless numpy Pillow matplotlib

import cv2
import easyocr
import numpy as np
import io
from google.colab.patches import cv2_imshow
from IPython.display import display, HTML
import matplotlib.pyplot as plt


def upload_image():
    from google.colab import files
    uploaded = files.upload()
    for filename in uploaded.keys():
        return filename, uploaded[filename]
    return None, None

print("Whiteboard OCR – Text Detection")

# Upload image
filename, file_bytes = upload_image()
if filename is not None:
    # Convert to OpenCV format
    image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), 1)

    # Display original image
    print("\nOriginal Image:")
    cv2_imshow(image)

    # OCR with EasyOCR
    reader = easyocr.Reader(['en'], gpu=True)  # Use GPU for faster processing
    results = reader.readtext(image)

    # Draw results
    output_image = image.copy()
    extracted_texts = []

    for (bbox, text, score) in results:
        if score > 0.25:  # Confidence threshold
            top_left = tuple(map(int, bbox[0]))
            bottom_right = tuple(map(int, bbox[2]))
            cv2.rectangle(output_image, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(output_image, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            extracted_texts.append(f"{text} (confidence: {score:.2f})")

    # Display processed image
    print("\nDetected Text Output:")
    cv2_imshow(output_image)

    # Display extracted text
    print("\nExtracted Text:")
    for text in extracted_texts:
        print(f"- {text}")

    # Save results
    output_pil = Image.fromarray(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    output_pil.save('detected_text.png')

    # Create download link
    from google.colab import files
    files.download('detected_text.png')
    print("\nResults saved as 'detected_text.png' and downloaded automatically")
else:
    print("No image was uploaded")
