from flask import Flask, request, render_template
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle

from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras import backend as K
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('float32')   
app = Flask(__name__)

with open('class_weights.pkl', 'rb') as f:
    class_weights = pickle.load(f)

def weighted_binary_crossentropy(y_true, y_pred):
    loss = 0
    for i in range(6):
        weight = tf.where(y_true[:, i] == 1, class_weights[i][1], class_weights[i][0])
        weight = tf.cast(weight, tf.float32)
        loss += tf.reduce_mean(weight * tf.keras.losses.binary_crossentropy(y_true[:, i], y_pred[:, i]))
    return loss / 6

from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable()
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = Precision(thresholds=0.3)
        self.recall = Recall(thresholds=0.3)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + K.epsilon()))

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

model = tf.keras.models.load_model(
    'toxicity_model.keras',
    custom_objects={
        'weighted_binary_crossentropy': weighted_binary_crossentropy,
        'F1Score': F1Score
    }
)

df = pd.read_csv("train.csv")

def score_comment(comment):
    PREDICTION_THRESHOLD = 0.3
    input_tensor = tf.convert_to_tensor([comment])  
    results = model.predict(input_tensor)
    text = ''
    for idx, col in enumerate(df.columns[2:]):
        text += f'{col}: {results[0][idx] > PREDICTION_THRESHOLD}\n'
    return text



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    comment = request.form['comment']
    result = score_comment(comment)
    return render_template('index.html', prediction=result, comment=comment)

if __name__ == '__main__':
    app.run(debug=True)
