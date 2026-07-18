# Day 6 - Handwritten Digit Recognition (Neural Network)

First project in this series using a **real dataset** instead of made-up data, and also my first neural network.

## The dataset

Used scikit-learn's built-in Digits dataset - **1,797 actual scanned images** of handwritten digits (0-9), originally from the UCI Machine Learning Repository. This is real handwriting from real people, not something I typed up myself. Each image is tiny - just 8x8 pixels, and each pixel is a brightness value. When flattened, that's 64 numbers representing one digit.

Source: https://scikit-learn.org/stable/datasets/toy_dataset.html#digits-dataset

## What's different about image data

Up to now every project used "normal" tabular features (income, area, TF-IDF scores). Images are different - each pixel IS a feature. An 8x8 image = 64 features per sample, where each feature is just "how bright is this one pixel." The model has no idea it's looking at a picture - it just sees 64 numbers and has to learn that certain patterns of numbers = certain digits.

## The model: Neural Network (MLPClassifier)

Used a Multi-Layer Perceptron with two hidden layers (64 neurons, then 32 neurons). This is a "real" neural network - the same basic idea behind deep learning, just small and simple enough to train on a laptop in seconds. It processes the 64 pixel-brightness numbers through those layers, learning to combine them into patterns that map to a digit.

Scaled the pixel values with StandardScaler first - neural networks are pretty sensitive to input scale, they train faster and more reliably when features are roughly centered around 0.

## Results

**96.67% accuracy** on the test set (360 images it never saw during training). Digging into the classification report, digit `1` and `8` had the lowest precision/recall (around 91-92%) - which actually makes visual sense, since handwritten 1s and 8s can look ambiguous depending on someone's handwriting style. The confusion matrix (`confusion_matrix.png`) shows exactly which digits get mixed up with which - useful for seeing WHERE the model struggles, not just that it's 97% accurate overall.

## Visualizations produced

- `sample_digits.png` - a peek at 8 raw digit images from the dataset before any training
- `confusion_matrix.png` - which digits get confused with which
- `predictions_sample.png` - actual test images with the model's prediction vs. the real label, color coded green (correct) / red (wrong)

## Stack
Python, Scikit-learn (MLPClassifier, load_digits), Matplotlib, NumPy

## Running it
```bash
pip install scikit-learn matplotlib numpy
python digit_recognition.py
```

## What I'd try next
- Try a Convolutional Neural Network (CNN) with a deep learning library like PyTorch or TensorFlow - CNNs are specifically built for images and should do even better
- Try the full MNIST dataset (28x28 pixels, 70,000 images) instead of this smaller 8x8 version, and see how accuracy and training time change
- Look closer at the specific misclassified 1s and 8s to understand what's confusing the model
