# CNN Image Classification Project (SVHN)

## Project Description
This project uses a Convolutional Neural Network (CNN) to classify images of digits from 
""the SVHN dataset.""
The model learns to recognize numbers from 0 to 9 in real-world images.

## Dataset
The dataset used is SVHN (Street View House Numbers).
- It contains RGB images of digits (0–9)
- Image size: 32 × 32 × 3
- It is divided into training and testing sets

## Model Structure
The CNN model has the following layers:

### Convolution Layers:
1. Conv2D (3 → 16), kernel size = 3, padding = 1  
   + Batch Normalization + ReLU + MaxPooling (2×2)

2. Conv2D (16 → 32), kernel size = 3, padding = 1  
   + Batch Normalization + ReLU + MaxPooling (2×2)

3. Conv2D (32 → 64), kernel size = 3, padding = 1  
   + Batch Normalization + ReLU + MaxPooling (2×2)

### Fully Connected Layers:
- Flatten layer  
- Linear (64 × 4 × 4 → 128) + ReLU  
- Dropout (0.3)  
- Linear (128 → 10 classes)


## Training Details
- Loss Function: CrossEntropyLoss  
- Optimizer: Adam  
- Learning Rate: 0.001  
- Batch Size: 32  
- Epochs: 10  

## Results
- Final Training Accuracy: ~89.87%  
- Final Test Accuracy: 91.36%


## How to Run the Project
1. Create a virtual environment:
python -m venv venv


2. Activate the environment:
venv\Scripts\activate


3. Install requirements:
pip install -r requirements.txt


4. Run the code:
python sectiontaskcnn.py


## Conclusion
The CNN model successfully learned to classify digits from images.  
It achieved good performance with a test accuracy above 91%, which shows strong learning ability and good generalization.
