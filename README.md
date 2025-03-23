# Ocular Disease Detection

## Overview
This project focuses on developing a machine learning model for ocular disease detection. The model is trained on a dataset containing normal and cataract images and achieves an accuracy of 89%. The goal is to classify eye conditions and assist in early detection of ocular diseases using deep learning techniques.

## Features
- Classification of ocular diseases (Normal vs. Cataract)
- Deep learning model trained on labeled eye images
- Uses VGG-16 with PyTorch for training
- Real-time detection with OpenCV (planned)
- Deployment via a web application (planned)

## Dataset
The dataset consists of images of normal and cataract-affected eyes. It has been processed and used to train the model for accurate classification.

## Requirements
To run this project, install the following dependencies:
```bash
pip install torch torchvision numpy opencv-python matplotlib tensorflow keras tqdm pandas os opencv-python-headless
```

## Additional Imports Used in the Project
```python
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import random
import itertools
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
```

## Displaying Cataract Images with Labels
This snippet loads and displays a set of cataract images along with their filenames using OpenCV and Matplotlib:
```python
plt.figure(figsize=(8,8))
for i in range(9):
    img = df_cat_filenames[i]
    image = cv2.imread(os.path.join(img_dir, img))

    # Convert image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Subplot variables - (# of rows, # of columns, iterate through locations on grid)
    plt.subplot(3,3,i+1)
    plt.imshow(image_rgb)
    
    # Label with filename and diagnosis
    plt.xlabel('Filename: {}
''Cataract'.format(df_cat_filenames[i]))

plt.tight_layout()
```
### Explanation:
- The function `cv2.imread()` reads the image from the specified directory.
- The image is then converted from BGR to RGB format using `cv2.cvtColor()`.
- The `plt.subplot()` function is used to display a grid of 3x3 images.
- The filenames and corresponding labels are added as captions using `plt.xlabel()`.
- `plt.tight_layout()` ensures the plots are properly spaced and visually clear.

## Displaying Normal Images with Labels
This snippet loads and displays a set of normal images along with their filenames using OpenCV and Matplotlib:
```python
plt.figure(figsize=(8,8))
for i in range(9):
    img = df_norm_filenames_random[i]
    image = cv2.imread(os.path.join(img_dir, img))

    # Convert image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Subplot variables - (# of rows, # of columns, iterate through locations on grid)
    plt.subplot(3,3,i+1)
    plt.imshow(image_rgb)
    
    # Label with filename and diagnosis
    plt.xlabel('Filename: {}
''Normal'.format(df_norm_filenames_random[i]))

plt.tight_layout()
```
### Explanation:
- Similar to the cataract image display function, this snippet loads normal images.
- The images are converted from BGR to RGB format.
- The `plt.subplot()` function arranges images in a 3x3 grid.
- Filenames and labels are added using `plt.xlabel()`.
- `plt.tight_layout()` ensures better visualization.

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ocular-disease-detection.git
   cd ocular-disease-detection
   ```
2. Run the Jupyter Notebook to train the model:
   ```bash
   jupyter notebook OCULAR1.ipynb
   ```
3. To test the model, place your images in the designated folder and modify the evaluation script.

## Results
- Model Accuracy: 89%
- Loss: 0.89765
- Sample predictions are included in the `output` directory.

## Future Enhancements
- Improve accuracy with data augmentation and hyperparameter tuning
- Implement real-time detection using OpenCV
- Develop a user-friendly web interface for diagnosis

## Contributing
Feel free to contribute to this project by submitting pull requests or reporting issues.

## License
This project is open-source and available under the MIT License.

