import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, Arrow
import numpy as np

# Set up the figure
fig, ax = plt.subplots(1, 1, figsize=(16, 8))
ax.set_xlim(0, 16)
ax.set_ylim(0, 8)
ax.axis('off')

# Colors for different stages
data_color = '#FF6B6B'
preprocess_color = '#4ECDC4'
model_color = '#45B7D1'
fl_color = '#96CEB4'
eval_color = '#FFEAA7'
deploy_color = '#DDA0DD'

# Define pipeline stages
stages = [
    ('Raw Datasets', 'CICIoT2023\nEdgeIIoTset', data_color, 1),
    ('Preprocessing', 'Clean + Encode +\nNormalize', preprocess_color, 3),
    ('Windowing', 'Sliding Windows\n(20×5 stride)', preprocess_color, 5),
    ('Model Training', 'CNN+BiLSTM+\nAttention', model_color, 7),
    ('Federated\nLearning', 'FedAvg across\n5 clients', fl_color, 9),
    ('Evaluation', 'Metrics +\nConfusion Matrix', eval_color, 11),
    ('Dashboard\nDeployment', 'Streamlit App\n+ Real-time Detection', deploy_color, 13)
]

# Draw pipeline stages
for name, desc, color, x_pos in stages:
    # Main box
    box = FancyBboxPatch((x_pos-0.8, 4), 1.6, 2, boxstyle="round,pad=0.2",
                        facecolor=color, edgecolor='black', linewidth=2)
    ax.add_patch(box)

    # Text
    ax.text(x_pos, 5, name, ha='center', va='center',
            fontsize=11, fontweight='bold')
    ax.text(x_pos, 4.3, desc, ha='center', va='center',
            fontsize=9)

# Draw arrows between stages
for i in range(len(stages)-1):
    start_x = stages[i][3] + 0.8
    end_x = stages[i+1][3] - 0.8

    # Arrow
    arrow = Arrow(start_x, 5, end_x-start_x, 0, width=0.3,
                 facecolor='black', edgecolor='black')
    ax.add_artist(arrow)

# Add detailed information boxes below
details = [
    ('1.35M + 69K samples\n46 features each', 1),
    ('Missing values,\ncategorical encoding,\nfeature scaling', 3),
    ('Temporal sequences\nfrom tabular data\nfor LSTM processing', 5),
    ('Hybrid architecture:\nCNN for spatial,\nBiLSTM for temporal,\nAttention for focus', 7),
    ('5 IoT clients:\nSmart Home, EV Station,\nGrid Sensor, Solar, Industrial\nFedAvg aggregation', 9),
    ('94.65% accuracy\nMacro F1: 68.4%\nPer-class metrics\nROC curves', 11),
    ('11-page dashboard:\nLive detection,\nModel explainability,\nReal-time monitoring', 13)
]

for detail_text, x_pos in details:
    detail_box = FancyBboxPatch((x_pos-0.7, 1.5), 1.4, 1.8, boxstyle="round,pad=0.1",
                                facecolor='lightgray', edgecolor='black', linewidth=1)
    ax.add_patch(detail_box)
    ax.text(x_pos, 2.4, detail_text, ha='center', va='center', fontsize=8)

# Connect main stages to details
for x_pos in [1, 3, 5, 7, 9, 11, 13]:
    con = ConnectionPatch((x_pos, 4), (x_pos, 3.3), "data", "data",
                         arrowstyle="-", shrinkA=0, shrinkB=0,
                         color='gray', linewidth=1, alpha=0.5)
    ax.add_artist(con)

# Add title
plt.title('Complete Data Flow Pipeline - Federated Learning IDS for Smart Energy IoT',
          fontsize=16, fontweight='bold', pad=30)

# Add key highlights
highlights = [
    '🔒 Privacy: Raw data never leaves clients',
    '🧠 Hybrid Model: CNN + BiLSTM + Attention',
    '📊 Multi-class: DDoS, DoS, Normal, Other',
    '⚡ Real-time: Live detection capability',
    '🌐 Federated: 5 IoT environments'
]

for i, highlight in enumerate(highlights):
    ax.text(8, 7.5 - i*0.3, highlight, ha='center', va='center',
            fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3",
            facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.savefig('visualizations/data_flow_pipeline.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Data Flow Pipeline Diagram saved")