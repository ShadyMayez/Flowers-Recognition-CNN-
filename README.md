# Flowers Recognition using Convolutional Neural Networks (CNN)

## Project Overview and Purpose
This project is a Deep Learning application designed to recognize and classify images of five different types of flowers: Daisy, Dandelion, Rose, Sunflower, and Tulip. Using a custom-built Convolutional Neural Network (CNN) implemented in Keras and TensorFlow, the model learns to identify distinct visual patterns for each flower species.

## Key Technologies and Libraries
- **Deep Learning**: `TensorFlow`, `Keras`
- **Computer Vision**: `OpenCV` (`cv2`), `PIL`
- **Data Manipulation**: `numpy`, `pandas`
- **Visualization**: `matplotlib`, `seaborn`
- **Machine Learning Support**: `scikit-learn`

## Methodology and Analysis Workflow
1. **Data Loading**: Fetches the "Flowers Recognition" dataset from Kaggle.
2. **Preprocessing**: 
   - Images are resized to 150x150 pixels.
   - Categorical labels are encoded using `LabelEncoder`.
   - Data is split into training and validation sets.
3. **Model Architecture**:
   - **Convolutional Layers**: Multiple `Conv2D` layers with ReLU activation to extract spatial features.
   - **Pooling**: `MaxPooling2D` layers to reduce dimensionality.
   - **Regularization**: `Dropout` and `BatchNormalization` to prevent overfitting.
   - **Output**: A final Dense layer with a Softmax activation for 5-way classification.
4. **Optimization**: 
   - Used the `Adam` optimizer.
   - Implemented `ReduceLROnPlateau` and `ModelCheckpoint` callbacks to ensure the best weights are saved and training remains stable.

## Results and Insights
- **Model Training**: The notebook tracks accuracy and loss curves for both training and validation phases.
- **Visualization**: Random predictions are visualized with labels to verify model intuition.
- **Outcome**: The CNN demonstrates the ability to differentiate between species with similar colors (like Dandelions and Sunflowers) by focusing on texture and petal structure.


## How to Run
1. Since the data is imported from Kaggle, ensure you have a Kaggle API key or download the [Flowers Recognition Dataset](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition) manually.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
