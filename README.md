# Breast Cancer Detection using Convolutional Neural Networks CNN 

Project Objective

The goal of this project is to build a Deep Learning model capable of automatically detecting the presence of breast cancer from mammogram images.
This work demonstrates how a Convolutional Neural Network (CNN) can be trained to distinguish between “cancer” and “negative” images.

### Context

Breast cancer is one of the most common cancers worldwide.
Automated medical image analysis can help radiologists speed up screening and reduce human error.
This project is a pedagogical implementation of a binary classification model based on public image data.

### Data Used

The data comes from the public GitHub repository:
MachineLearnia / breast_cancer_public_data

Two classes:
- Negative → image without tumor
- Cancer → image with tumor

Total number of images: 820
Image size: 224 x 224

### Project Steps
1.Data Loading and Preparation
- Image reading with cv2
- Resizing to (224x224)
- Pixel normalization (values between 0 and 1)
2. Dataset Splitting
- Split into training and testing sets (80/20)
- Class balancing using stratify
3. CNN Model Creation
- 3 Convolutional + MaxPooling layers
- 1 Dense layer + Dropout
- Final activation: sigmoid (binary output)
- Optimizer: Adam, Loss function: binary_crossentropy
4. Improvements
- Data Augmentation (rotation, zoom, horizontal flip, etc.)
- Reduction of overfitting
5.Performance Evaluation
- Recall (sensitivity) = 0.939 → the model detects about 94% of true cancer cases
- AUC (Area Under the Curve) = 0.9917 → excellent class distinction
- Visualization of the ROC curve

### Results
- Accuracy ≈ 96%
- Recall ≈ 94%
- AUC ≈ 0.99
The model distinguishes “cancer” and “negative” images very effectively.

### Technologies Used
- Python
- TensorFlow / Keras
- NumPy / Pandas
- OpenCV
- Matplotlib / Seaborn
- Scikit-learn
