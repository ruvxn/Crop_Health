🌱 Crop Health Classification Project

📌 Overview

This project focuses on classifying plant diseases using a Convolutional Neural Network (CNN) based on MobileNetV2. Over multiple iterations, we made significant updates to improve model accuracy, generalization, and efficiency.

This model is optimized to run on Google Colab and can classify 38 different plant diseases based on image input.

🚀 Project Timeline & Improvements

🛠 Initial Model (Baseline)

Setup & First Training Run

Pretrained Model: MobileNetV2 (Frozen)

Loss Function: Categorical Crossentropy

Data Augmentation: Basic (Horizontal Flip, Rotation)

Optimizer: Adam (Fixed LR: 0.001)

Batch Size: 32

Class Weights: None (No handling for class imbalance)

Early Stopping: Implemented

Learning Rate Scheduler: None

Performance:

Training Accuracy: ~85%

Validation Accuracy: ~54%

Test Accuracy: ~49.6%

Issue: High overfitting (training accuracy much higher than validation accuracy)

🔄 Second Cycle - Handling Class Imbalance & Data Preprocessing Fixes

Updates:

Class Imbalance Handling:

Implemented Class Weights using compute_class_weight

Converted one-hot labels to class indices before computing weights

Data Pipeline Optimization:

Used tf.data for faster, efficient dataset loading

Batch Size Increased: From 32 to 64 (improves stability)

Regularization:

Increased Dropout from 0.3 → 0.5

Used L2 weight decay in the fully connected layer

Performance:

Training Accuracy: ~90%

Validation Accuracy: ~65.6%

Test Accuracy: ~54.8%

Issue: Validation loss fluctuated a lot → Potential label noise in dataset

🚀 Third Cycle - Fixing Overfitting & Advanced Learning Rate Scheduling

Updates:

Unfreezing Last 70 Layers of MobileNetV2 for fine-tuning

Switched Loss Function:

Used Focal Loss (handles class imbalance better)

Improved Learning Rate Strategy:

Switched from Fixed LR → Cosine Decay Scheduler

Implemented ReduceLROnPlateau for dynamic adjustments

Further Regularization:

Increased L2 weight decay

Applied Batch Normalization (stabilizes training)

Increased Augmentation Variability (brightness, contrast, zoom)

Performance:

Training Accuracy: ~93.5%

Validation Accuracy: ~85.4%

Test Accuracy: ~84.9%

Issue: Still some misclassifications in less frequent classes

🔥 Final Cycle - Optimizing for Google Colab & Saving Model in .keras

Updates:

Google Colab Compatibility:

Dataset stored in Google Drive

Ensured all dependencies installed (pip install tensorflow numpy etc.)

Fixed Model Saving:

Switched from .h5 (legacy) → .keras format

Optimized Evaluation & Reporting:

Confusion Matrix visualization added

Used classification report for per-class performance

Final Performance:

Training Accuracy: 93.7%

Validation Accuracy: 85.6%

Test Accuracy: 84.9% 🚀✅

📊 Accuracy Improvements Across Cycles

Cycle

Training Acc

Validation Acc

Test Acc

Key Changes

Baseline

85%

54%

49.6%

Basic MobileNetV2, Fixed LR, No Class Weights

Cycle 2

90%

65.6%

54.8%

Class Weights, L2 Regularization, Larger Batch Size

Cycle 3

93.5%

85.4%

84.9%

Focal Loss, Cosine Decay, Fine-Tuning 70 Layers

Final Model

93.7%

85.6%

84.9%



🎯 Future Improvements

✅ What Worked Well:

Using Focal Loss significantly improved class balance.

Fine-tuning 70 layers of MobileNetV2 boosted accuracy.

Cosine Decay LR Scheduler helped stabilize training.

⚠️ Challenges Remaining:

Some classes still have low recall due to dataset imbalance.

Model misclassifies visually similar diseases.

🎯 Next Steps:

Experiment with Vision Transformers (ViTs) or EfficientNetV2 for better accuracy.

Implement Semi-Supervised Learning to improve rare class predictions.





