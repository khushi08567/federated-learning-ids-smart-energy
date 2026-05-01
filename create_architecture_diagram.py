import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

# Set up the figure
fig, ax = plt.subplots(1, 1, figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')

# Colors
cnn_color = '#FF6B6B'
lstm_color = '#4ECDC4'
attention_color = '#45B7D1'
merge_color = '#96CEB4'
dense_color = '#FFEAA7'
input_color = '#DDA0DD'
output_color = '#98D8C8'

# Input layer
input_box = FancyBboxPatch((0.5, 7), 2, 1.5, boxstyle="round,pad=0.1",
                          facecolor=input_color, edgecolor='black', linewidth=2)
ax.add_patch(input_box)
ax.text(1.5, 7.75, 'Input\n(20 × 46)', ha='center', va='center', fontsize=10, fontweight='bold')

# CNN Branch
# Conv1D 64
cnn1_box = FancyBboxPatch((3.5, 8), 1.5, 0.8, boxstyle="round,pad=0.1",
                         facecolor=cnn_color, edgecolor='black', linewidth=2)
ax.add_patch(cnn1_box)
ax.text(4.25, 8.4, 'Conv1D\n64', ha='center', va='center', fontsize=9, fontweight='bold')

# BatchNorm
bn1_box = FancyBboxPatch((3.5, 7), 1.5, 0.8, boxstyle="round,pad=0.1",
                        facecolor=cnn_color, edgecolor='black', linewidth=2)
ax.add_patch(bn1_box)
ax.text(4.25, 7.4, 'Batch\nNorm', ha='center', va='center', fontsize=9, fontweight='bold')

# MaxPool
pool1_box = FancyBboxPatch((3.5, 6), 1.5, 0.8, boxstyle="round,pad=0.1",
                          facecolor=cnn_color, edgecolor='black', linewidth=2)
ax.add_patch(pool1_box)
ax.text(4.25, 6.4, 'MaxPool\n2', ha='center', va='center', fontsize=9, fontweight='bold')

# Dropout
drop1_box = FancyBboxPatch((3.5, 5), 1.5, 0.8, boxstyle="round,pad=0.1",
                          facecolor=cnn_color, edgecolor='black', linewidth=2)
ax.add_patch(drop1_box)
ax.text(4.25, 5.4, 'Dropout\n0.3', ha='center', va='center', fontsize=9, fontweight='bold')

# Conv1D 128
cnn2_box = FancyBboxPatch((3.5, 4), 1.5, 0.8, boxstyle="round,pad=0.1",
                         facecolor=cnn_color, edgecolor='black', linewidth=2)
ax.add_patch(cnn2_box)
ax.text(4.25, 4.4, 'Conv1D\n128', ha='center', va='center', fontsize=9, fontweight='bold')

# Global Avg Pool
gap_box = FancyBboxPatch((3.5, 2.5), 1.5, 0.8, boxstyle="round,pad=0.1",
                        facecolor=cnn_color, edgecolor='black', linewidth=2)
ax.add_patch(gap_box)
ax.text(4.25, 2.9, 'Global\nAvg Pool', ha='center', va='center', fontsize=9, fontweight='bold')

# BiLSTM Branch
# BiLSTM 64 (1)
bilstm1_box = FancyBboxPatch((6.5, 7.5), 1.5, 0.8, boxstyle="round,pad=0.1",
                            facecolor=lstm_color, edgecolor='black', linewidth=2)
ax.add_patch(bilstm1_box)
ax.text(7.25, 7.9, 'BiLSTM\n64', ha='center', va='center', fontsize=9, fontweight='bold')

# Dropout
drop_lstm1_box = FancyBboxPatch((6.5, 6.5), 1.5, 0.8, boxstyle="round,pad=0.1",
                               facecolor=lstm_color, edgecolor='black', linewidth=2)
ax.add_patch(drop_lstm1_box)
ax.text(7.25, 6.9, 'Dropout\n0.3', ha='center', va='center', fontsize=9, fontweight='bold')

# BiLSTM 64 (2)
bilstm2_box = FancyBboxPatch((6.5, 5.5), 1.5, 0.8, boxstyle="round,pad=0.1",
                            facecolor=lstm_color, edgecolor='black', linewidth=2)
ax.add_patch(bilstm2_box)
ax.text(7.25, 5.9, 'BiLSTM\n64', ha='center', va='center', fontsize=9, fontweight='bold')

# Dropout
drop_lstm2_box = FancyBboxPatch((6.5, 4.5), 1.5, 0.8, boxstyle="round,pad=0.1",
                               facecolor=lstm_color, edgecolor='black', linewidth=2)
ax.add_patch(drop_lstm2_box)
ax.text(7.25, 4.9, 'Dropout\n0.3', ha='center', va='center', fontsize=9, fontweight='bold')

# Attention
attn_dense_box = FancyBboxPatch((6.5, 3.5), 1.5, 0.8, boxstyle="round,pad=0.1",
                               facecolor=attention_color, edgecolor='black', linewidth=2)
ax.add_patch(attn_dense_box)
ax.text(7.25, 3.9, 'Dense\n1, tanh', ha='center', va='center', fontsize=9, fontweight='bold')

attn_softmax_box = FancyBboxPatch((6.5, 2.5), 1.5, 0.8, boxstyle="round,pad=0.1",
                                 facecolor=attention_color, edgecolor='black', linewidth=2)
ax.add_patch(attn_softmax_box)
ax.text(7.25, 2.9, 'Softmax\n(axis=1)', ha='center', va='center', fontsize=9, fontweight='bold')

attn_multiply_box = FancyBboxPatch((6.5, 1.5), 1.5, 0.8, boxstyle="round,pad=0.1",
                                  facecolor=attention_color, edgecolor='black', linewidth=2)
ax.add_patch(attn_multiply_box)
ax.text(7.25, 1.9, 'Multiply\n+ Sum', ha='center', va='center', fontsize=9, fontweight='bold')

# Merge
merge_box = FancyBboxPatch((10, 2), 1.5, 0.8, boxstyle="round,pad=0.1",
                          facecolor=merge_color, edgecolor='black', linewidth=2)
ax.add_patch(merge_box)
ax.text(10.75, 2.4, 'Concatenate\n(256)', ha='center', va='center', fontsize=9, fontweight='bold')

# Dense layers
dense1_box = FancyBboxPatch((12.5, 3), 1.5, 0.8, boxstyle="round,pad=0.1",
                           facecolor=dense_color, edgecolor='black', linewidth=2)
ax.add_patch(dense1_box)
ax.text(13.25, 3.4, 'Dense\n256, ReLU', ha='center', va='center', fontsize=9, fontweight='bold')

dense2_box = FancyBboxPatch((12.5, 2), 1.5, 0.8, boxstyle="round,pad=0.1",
                           facecolor=dense_color, edgecolor='black', linewidth=2)
ax.add_patch(dense2_box)
ax.text(13.25, 2.4, 'Dense\n128, ReLU', ha='center', va='center', fontsize=9, fontweight='bold')

# Output
output_box = FancyBboxPatch((12.5, 1), 1.5, 0.8, boxstyle="round,pad=0.1",
                           facecolor=output_color, edgecolor='black', linewidth=2)
ax.add_patch(output_box)
ax.text(13.25, 1.4, 'Softmax\n4 classes', ha='center', va='center', fontsize=9, fontweight='bold')

# Draw connections
def draw_connection(start, end, color='black', linewidth=2):
    con = ConnectionPatch(start, end, "data", "data",
                         arrowstyle="->", shrinkA=5, shrinkB=5,
                         mutation_scale=15, fc=color, color=color, linewidth=linewidth)
    ax.add_artist(con)

# Input to CNN and BiLSTM
draw_connection((2.5, 7.75), (3.5, 8.4), cnn_color)
draw_connection((2.5, 7.75), (6.5, 7.9), lstm_color)

# CNN flow
draw_connection((4.25, 7.6), (4.25, 7.4), cnn_color)
draw_connection((4.25, 6.6), (4.25, 6.4), cnn_color)
draw_connection((4.25, 5.6), (4.25, 5.4), cnn_color)
draw_connection((4.25, 4.6), (4.25, 4.4), cnn_color)
draw_connection((4.25, 3.1), (4.25, 2.9), cnn_color)

# BiLSTM flow
draw_connection((7.25, 7.1), (7.25, 6.9), lstm_color)
draw_connection((7.25, 6.1), (7.25, 5.9), lstm_color)
draw_connection((7.25, 5.1), (7.25, 4.9), lstm_color)
draw_connection((7.25, 4.1), (7.25, 3.9), attention_color)
draw_connection((7.25, 3.1), (7.25, 2.9), attention_color)
draw_connection((7.25, 2.1), (7.25, 1.9), attention_color)

# Merge
draw_connection((5.0, 2.9), (10.0, 2.4), cnn_color)
draw_connection((8.0, 1.9), (10.0, 2.4), attention_color)

# Dense flow
draw_connection((11.5, 2.4), (12.5, 3.4), dense_color)
draw_connection((13.25, 2.6), (13.25, 2.4), dense_color)
draw_connection((13.25, 1.6), (13.25, 1.4), output_color)

# Labels
ax.text(4.25, 9.2, 'CNN Branch\n(Spatial Patterns)', ha='center', va='center',
        fontsize=12, fontweight='bold', color=cnn_color)
ax.text(7.25, 9.2, 'BiLSTM + Attention Branch\n(Temporal + Focus)', ha='center', va='center',
        fontsize=12, fontweight='bold', color=lstm_color)
ax.text(10.75, 3.5, 'Fusion Layer', ha='center', va='center',
        fontsize=12, fontweight='bold', color=merge_color)
ax.text(13.25, 4.0, 'Classification\nHead', ha='center', va='center',
        fontsize=12, fontweight='bold', color=dense_color)

plt.title('CNN + BiLSTM + Attention Architecture for Federated IDS', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('visualizations/model_architecture.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Model Architecture Diagram saved")