!pip install easyocr opencv-python-headless numpy Pillow matplotlib

import cv2
import easyocr
import numpy as np
from PIL import Image
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

print(" Whiteboard OCR – Text Detection ")

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
