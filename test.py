import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load frozen ResNet50
base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)
base_model.trainable = False

def extract_features(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    features = base_model(img, training=False)
    return features.numpy().squeeze()

# Test
feat = extract_features("test.jpg")
feat_2d = feat.reshape(-1, 1)
print(feat_2d.shape)
print(feat_2d)
