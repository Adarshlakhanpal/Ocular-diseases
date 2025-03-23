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
pip install torch torchvision numpy opencv-python matplotlib
```

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

