# Whiteboard-Content-Digitization-System
This is an ML project that consist  CV-based tool to extract and monitor whiteboard content and instructor activity

This project presents a hybrid OCR pipeline designed to improve the accuracy of whiteboard 
content recognition. The core system integrates two primary components: a real-time OCR 
interface built with EasyOCR, and a backend fine-tuning mechanism using Microsoft's 
TrOCR model. The system is designed to operate efficiently in a Google Colab environment, 
supporting both GPU acceleration and interactive usage through image upload. 
The front-end real-time OCR system allows users to upload whiteboard images, which are 
then processed using EasyOCR. Detected text regions are highlighted using OpenCV, and 
confidence scores are displayed for each prediction. The processed image with annotated text 
is saved and made downloadable, offering an intuitive, user-friendly interface for quick 
validation.
