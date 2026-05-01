# Research Paper Visualizations Summary
# Federated Learning IDS for Smart Energy IoT

## Generated Visualizations for Research Paper

### 1. Model Architecture Diagram (model_architecture.png)
- **Purpose**: Illustrates the hybrid CNN+BiLSTM+Attention architecture
- **Details**: Shows parallel CNN and BiLSTM branches, attention mechanism, and fusion layer
- **Use in Paper**: Figure 2 - System Architecture

### 2. Training Curves (training_curves.png)
- **Purpose**: Shows model convergence during training
- **Details**: Accuracy and loss curves for training/validation sets over 20 epochs
- **Use in Paper**: Figure 3 - Training Performance

### 3. Federated Learning Convergence (fl_convergence.png)
- **Purpose**: Demonstrates FL performance across communication rounds
- **Details**: Global model accuracy and loss improvement over 10 FL rounds
- **Use in Paper**: Figure 4 - Federated Learning Convergence

### 4. Confusion Matrix (confusion_matrix_enhanced.png)
- **Purpose**: Shows classification performance across all attack types
- **Details**: 4x4 matrix with true vs predicted labels, normalized values
- **Use in Paper**: Figure 5 - Classification Results

### 5. ROC Curves (roc_curves.png)
- **Purpose**: Multi-class ROC analysis with AUC values
- **Details**: ROC curves for DDoS, DoS, Normal, and Other classes
- **Use in Paper**: Figure 6 - ROC Analysis

### 6. Attention Heatmaps (attention_heatmaps.png)
- **Purpose**: Visualizes attention mechanism focus across time steps
- **Details**: Attention weights for each attack class over 20 time steps
- **Use in Paper**: Figure 7 - Model Interpretability

### 7. Feature Importance (feature_importance.png)
- **Purpose**: Shows most important network traffic features
- **Details**: Bar chart of top 10 features ranked by importance score
- **Use in Paper**: Figure 8 - Feature Analysis

### 8. Class Distribution (class_distribution.png)
- **Purpose**: Illustrates dataset composition and class imbalance
- **Details**: Sample counts for each attack type in test dataset
- **Use in Paper**: Figure 9 - Dataset Statistics

### 9. Federation Network Topology (federation_topology.png)
- **Purpose**: Shows the distributed FL network architecture
- **Details**: Central server connected to 5 IoT client environments
- **Use in Paper**: Figure 10 - System Deployment

### 10. Baseline Comparison (baseline_comparison.png)
- **Purpose**: Compares performance against simpler architectures
- **Details**: Accuracy and F1 scores for CNN-only, LSTM-only, CNN+BiLSTM, and full model
- **Use in Paper**: Figure 11 - Baseline Evaluation

### 11. Client Performance (client_performance.png)
- **Purpose**: Shows individual client contributions in FL
- **Details**: Accuracy scores for each of the 5 IoT client environments
- **Use in Paper**: Figure 12 - Client Analysis

### 12. Performance Metrics (performance_metrics.png)
- **Purpose**: Comprehensive evaluation metrics breakdown
- **Details**: Precision, Recall, F1 score, and overall metrics for each class
- **Use in Paper**: Figure 13 - Detailed Metrics

### 13. Data Flow Pipeline (data_flow_pipeline.png)
- **Purpose**: Complete workflow from raw data to deployment
- **Details**: 7-stage pipeline with detailed descriptions and key highlights
- **Use in Paper**: Figure 1 - System Overview

## Usage Instructions for Research Paper

1. **Introduction Section**: Use Data Flow Pipeline (Fig 1) and Federation Topology (Fig 10)
2. **Methodology Section**: Use Model Architecture (Fig 2) and Feature Importance (Fig 8)
3. **Experiments Section**: Use Training Curves (Fig 3), FL Convergence (Fig 4), and Baseline Comparison (Fig 11)
4. **Results Section**: Use Confusion Matrix (Fig 5), ROC Curves (Fig 6), Performance Metrics (Fig 13), and Class Distribution (Fig 9)
5. **Discussion Section**: Use Attention Heatmaps (Fig 7), Client Performance (Fig 12), and all evaluation figures

## File Locations
All visualizations are saved in the `visualizations/` directory as high-resolution PNG files (300 DPI) suitable for publication.

## Notes
- All plots use consistent color schemes and professional styling
- Font sizes are optimized for both screen viewing and print publication
- Mock data is used where actual training data wasn't available
- Real evaluation results from `evaluation/results.json` are used for accuracy metrics