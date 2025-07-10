# Comment Toxicity Classifier

## Overview
This project implements a machine learning model to classify text comments for toxicity across six categories: toxic, severe_toxic, obscene, threat, insult, and identity_hate. The model, trained on the Jigsaw Toxic Comment Classification dataset, uses a Bidirectional LSTM neural network with TensorFlow, achieving a validation macro F1 score of approximately 0.44. A Flask web application provides a user friendly interface for real time toxicity prediction.

## Dataset
The model is trained on the Jigsaw Toxic Comment Classification dataset (`train.csv`), containing ~159,568 comments labeled for toxicity. The dataset exhibits significant class imbalance (~90% non-toxic), with sparse positive labels for categories like `threat` and `identity_hate` (~1-2%). This imbalance contributed to overfitting, particularly for underrepresented classes, limiting the macro F1-score.

## Model Architecture
- **Input Processing**: `TextVectorization` layer (max tokens: 100,000, sequence length: 1,800) for comment preprocessing.
- **Layers**:
  - Embedding layer (64 dimensions).
  - Bidirectional LSTM (32 units, tanh activation).
  - Dense layers (128, 256, 128 units, ReLU activation) with L2 regularization (0.01) and dropout (0.4).
  - Output layer (6 units, sigmoid activation) for multi-label classification.
- **Total Parameters**: ~6.5M.
- **Loss Function**: Weighted binary cross-entropy to address class imbalance.
- **Metrics**: Accuracy, precision, recall, and custom F1-score (threshold: 0.3).
- **Training**:
  - Batch size: 128.
  - Epochs: 10 (early stopping on validation F1-score).
  - Oversampling: 3x for positive samples.
  - Mixed precision training for efficiency on T4 GPU.
  - Validation macro F1-score: 0.4373 (toxic: 0.7734, obscene: 0.7558, insult: 0.6929, severe_toxic: 0.4018, threat: 0.0, identity_hate: 0.0).

## Performance
The model achieves a validation macro F1-score of 0.4373, with strong performance on `toxic`, `obscene`, and `insult` categories but lower scores on `severe_toxic`, `threat`, and `identity_hate` due to limited positive samples. Overfitting was observed, primarily due to the dataset’s class imbalance and sparsity in certain categories, which constrained generalization.

## User Interface
The Flask web application provides an intuitive interface for toxicity prediction, built with Tailwind CSS. Users can input a comment and receive predictions for each toxicity category. Below is a screenshot of the UI:

![Toxicity Classifier UI](screenshot.png)  
*Paste your UI screenshot here (e.g., `screenshot.png`) and ensure it’s placed in the project directory.*
<img src="Ss.png" alt="App Preview" width="600">

## Setup Instructions
### Prerequisites
- Python 3.11
- TensorFlow 2.15
- Dependencies: `pandas`, `numpy`, `scikit-learn`, `flask`

### Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Comment_Toxicity
   ```
2. Install dependencies:
   ```bash
   pip install tensorflow==2.15 pandas numpy scikit-learn flask
   ```

### Training
1. Ensure `train.csv` is in the project directory or Google Drive (`/content/drive/MyDrive/Data/Comment_Toxicity/train.csv`).
2. In Google Colab with T4 GPU:
   ```python
   !pip install tensorflow==2.15 pandas numpy scikit-learn
   !python train_toxicity_model_optimized.py
   ```
3. Download output files: `toxicity_model.keras`, `vectorizer_assets.pkl`, `class_weights.pkl`.

### Running the Flask App
1. Place `toxicity_model.keras`, `vectorizer_assets.pkl`, `class_weights.pkl`, `train.csv`, and `index.html` in the project directory.
2. Run the Flask app:
   ```bash
   python app.py
   ```
3. Access the web interface at `http://127.0.0.1:5000`.

## Usage
- Enter a comment in the web interface’s text area.
- Click “Analyze” to view toxicity predictions for each category (True/False based on a 0.3 threshold).
- Example output:
  ```
  toxic: True
  severe_toxic: False
  obscene: True
  threat: False
  insult: True
  identity_hate: False
  ```

## Files
- `train_toxicity_model_optimized.py`: Training script for the LSTM model.
- `app.py`: Flask application for serving the prediction UI.
- `index.html`: Web interface template using Tailwind CSS.
- `toxicity_model.keras`: Trained model file.
- `vectorizer_assets.pkl`: Text vectorizer weights.
- `class_weights.pkl`: Class weights for weighted loss.
- `screenshot.png`: UI screenshot (add your own).

## Notes
- The dataset’s class imbalance and limited positive samples for certain categories (`threat`, `identity_hate`) impacted model performance, leading to overfitting. Future improvements could involve augmenting the dataset with additional labeled examples or using advanced techniques like transfer learning.
- Training takes ~17-20 minutes on a T4 GPU for 10 epochs.