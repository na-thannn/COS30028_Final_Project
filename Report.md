# COS30028 Applied Machine Learning
# Final Project Report: Facial Recognition with Emotion and Liveness

Student Name: Le Luu Phuoc Thinh  
Student ID: SWD00132 
Date: 9 April 2026

---

## 1. Objective

This project implements a facial verification pipeline for an attendance-style system. The core objective is to determine whether two face images belong to the same identity. In addition, the system design includes integration points for anti-spoofing (liveness) and emotion recognition modules.

The project requirements were addressed by:

- Implementing two face embedding training approaches (classification-based and metric-learning based).
- Evaluating verification performance with ROC/AUC on validation pairs.
- Producing a final test submission CSV with one prediction per verification pair.
- Providing a local app pipeline for checkpoint-based inference.

---

## 2. Dataset and Task Definition

The project uses the provided dataset structure:

- `data/classification_data/train_data`: identity-labeled images for model training.
- `data/classification_data/val_data`: identity-labeled images for validation.
- `data/classification_data/test_data`: identity-labeled images for classification checks.
- `data/verification_data`: images referenced by verification trials.
- `data/verification_pairs_val.txt`: validation verification pairs with labels.
- `data/verification_pairs_test.txt`: test verification pairs without labels.

Task definition:

- Input: pair of face images.
- Output: binary label (`1` same person, `0` different people).

---

## 3. Methodology

### 3.1 Face Embedding Learning

Two embedding-learning strategies were implemented.

1. Classification-based embedding learning
- Train a CNN backbone with an identity classification head.
- Use the feature vector before the classification head as face embedding.

2. Metric-learning embedding learning
- Train embeddings directly with triplet loss.
- Enforce smaller distance for same-identity samples and larger distance for different-identity samples.

### 3.2 Verification Pipeline

The same verification workflow is used for both embedding models:

1. Encode each image in a pair into embeddings.
2. Compute similarity (cosine metric used for final run).
3. Apply threshold selected from validation trials.
4. Convert scores into binary predictions for submission.

### 3.3 Engineering Choices for Local CPU Training

Because cloud GPU runtime was limited during experimentation, training was completed locally on CPU with:

- Overnight sequential training scripts.
- Resume-capable checkpoints.
- Lightweight backbone option for CPU feasibility.
- Post-training model selection script.

### 3.4 Training Configuration

The finalized local training configuration was:

- Backbone: `mobilenet_v3_small`
- Epochs: `25`
- Batch size: `32`
- Image size: `112`
- Embedding dimension: `128`
- Optimizer: `Adam`
- Learning rate: `1e-3`
- Classification loss: `CrossEntropyLoss`
- Metric-learning loss: `TripletMarginLoss` with margin `0.4`
- Verification metric for final post-processing: `cosine`
- Pairwise validation sampling during training: up to `500` pairs per epoch (`--max-eval-pairs 500`)

This configuration was selected to balance model quality and runtime stability on CPU.

---

## 4. Implementation Summary

Main implementation components:

- `src/training/train_classification_local.py`: local classification training.
- `src/training/train_metric_local.py`: local metric-learning training.
- `src/training/post_training.py`: checkpoint comparison, thresholding, submission generation.
- `src/ml/embedding.py`: backbone and embedding model definitions.
- `src/ml/verification.py`: robust checkpoint loading and inference.
- `src/ui/streamlit_app.py`: local demo app.

Produced checkpoints:

- `checkpoints/best_embedding_classification_local.pt`
- `checkpoints/best_embedding_metric_local.pt`
- `checkpoints/embedding_classification_local.pt`
- `checkpoints/embedding_metric_local.pt`

Produced submission:

- `results/submission_local.csv`

End-to-end execution flow:

1. Train classification embeddings: `src/training/train_classification_local.py`
2. Train metric embeddings: `src/training/train_metric_local.py`
3. Compare checkpoints and generate predictions: `src/training/post_training.py`
4. Run local interface and qualitative checks: `src/ui/streamlit_app.py`

User interface scope:

- The app supports local checkpoint loading and face verification inference.
- The app architecture includes extension points for liveness and emotion modules.
- This supports the attendance-system framework requirement while keeping the verification pipeline as the evaluated core.

---

## 5. Results and Discussion

### 5.1 Classification-Based Training Results

From `results/logs/classification_overnight.log`:

- Final epoch (25/25):
  - Train loss: `3.4269`
  - Validation accuracy: `0.3149`
  - Validation AUC: `0.8698`
- Peak observed validation AUC during run: `0.8840` (epoch 24).
- Best recorded validation accuracy: `0.3149`.

### 5.2 Metric-Learning Training Results

From `results/logs/metric_overnight.log`:

- Final epoch (25/25):
  - Triplet loss: `0.0754`
  - Validation AUC: `0.8243`
- Peak observed validation AUC during run: `0.8539` (epoch 11).
- Best recorded triplet loss: `0.0743`.

### 5.3 Model Comparison

Based on recorded validation AUC in these runs, the classification-based embedding model achieved stronger peak verification performance (`0.8840`) than the metric-learning run (`0.8539`). The metric-learning model still produced competitive embeddings and remains useful as an alternative candidate.

Interpretation:

- Classification training reached the highest validation AUC in this project, so it is the preferred primary model for final verification.
- Metric learning provided a robust secondary embedding approach and improved system diversity for future ensemble or fallback strategies.

### 5.4 Final Submission Validation

The final submission file was verified against expected test-pair count:

- File: `results/submission_local.csv`
- Total lines: `51836`
- Interpretation: 1 header + 51835 predictions (matches `verification_pairs_test.txt`).

---

## 6. Anti-Spoofing and Emotion Modules

The current project includes integration scaffolding for anti-spoofing (liveness) and emotion recognition in the local app pipeline. These components are structured for extension and demonstration, while the primary trained/evaluated models in this submission focus on face verification.

Integration status:

- **Liveness Detection**: Module placeholder and pipeline hooks are implemented for spoof resistance checking.
- **Emotion Recognition**: UI integration for emotion classification with confidence metrics.
- **Extensibility**: The app architecture supports easy integration of trained liveness and emotion detectors in future iterations.

Planned extensions:

- Train or integrate a dedicated liveness detector for improved spoof resistance.
- Link a production-grade emotion classifier with quantitative evaluation metrics.
- Conduct full stand-alone training/evaluation for these modules in subsequent work.

This was a deliberate scope decision to prioritize completion and quality of the core face verification requirement while maintaining a flexible framework for module expansion.

---

## 7. Limitations and Future Work

**Current limitations:**

- Training on CPU significantly increased runtime compared with GPU training, limiting hyperparameter exploration.
- Hyperparameter search was constrained by computational budget and time availability.
- Liveness and emotion modules are at the integration-stage; full quantitative benchmarking and training are deferred to future work.

**Recommendations for future work:**

- Transition training to GPU infrastructure to enable larger-scale experiments and hypercomplementary embedding-learning approaches (classification-based and metric-learning), comprehensive validation-based evaluation, robust checkpoint handling, and a complete final test submission CSV. The classification-based approach achieved the strongest validation AUC (0.8840) in the completed experiments. The project architecture is well-positioned to support future expansion of liveness detection, emotion analysis, and integration into production attendance systems. The combination of local CPU training capability, resume-able checkpoints, and a flexible web interface provides a solid foundation for ongoing research and deployment
- Conduct dedicated liveness and emotion detector training with comprehensive validation metrics.
- Implement ensemble methods combining multiple embedding approaches for improved robustness.
- Explore advanced architectures (e.g., Vision Transformers, ArcFace) if computational resources permit.

---

## 8. Conclusion

This project successfully delivered an end-to-end facial verification solution with two embedding-learning approaches, validation-based evaluation, robust checkpoint handling, and a complete final test submission CSV. The classification-based approach achieved the strongest validation AUC in the completed experiments, and the project is structured to support future expansion of liveness and emotion modules.

---
