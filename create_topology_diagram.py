import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, FancyBboxPatch, ConnectionPatch
import numpy as np

# Set up the figure
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Colors
server_color = '#FF6B6B'
client_colors = ['#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
connection_color = '#666666'

# Central Server
server_circle = Circle((7, 5), 1.2, facecolor=server_color, edgecolor='black', linewidth=3)
ax.add_patch(server_circle)
ax.text(7, 5, 'Federated\nServer\n(FedAvg)', ha='center', va='center',
        fontsize=12, fontweight='bold', color='white')

# Client positions (arranged in a circle around server)
client_positions = [
    (3, 8),   # Smart Home - top left
    (11, 8),  # EV Station - top right
    (2, 2),   # Grid Sensor - bottom left
    (12, 2),  # Solar Controller - bottom right
    (7, 9)    # Industrial - top center
]

client_names = [
    'Smart Home\nIoT Network',
    'EV Charging\nStation',
    'Grid Sensor\nNetwork',
    'Solar/Wind\nController',
    'Industrial\nEnergy System'
]

client_descriptions = [
    'Residential\ndevices',
    'Vehicle\ncharging',
    'Power\ngrid telemetry',
    'Renewable\ncontrol',
    'SCADA\nsystems'
]

# Draw clients
for i, (pos, name, desc) in enumerate(zip(client_positions, client_names, client_descriptions)):
    # Client circle
    client_circle = Circle(pos, 0.8, facecolor=client_colors[i], edgecolor='black', linewidth=2)
    ax.add_patch(client_circle)

    # Client label
    ax.text(pos[0], pos[1], f'{name}\n{desc}', ha='center', va='center',
            fontsize=9, fontweight='bold')

    # Connection to server
    con = ConnectionPatch(pos, (7, 5), "data", "data",
                         arrowstyle="<->", shrinkA=10, shrinkB=15,
                         mutation_scale=15, fc=connection_color, color=connection_color,
                         linewidth=2, alpha=0.7)
    ax.add_artist(con)

# Add round information
round_box = FancyBboxPatch((0.5, 0.5), 3, 1.5, boxstyle="round,pad=0.2",
                          facecolor='lightyellow', edgecolor='black', linewidth=2)
ax.add_patch(round_box)
ax.text(2, 1.25, 'Federated Learning Process:\n• Round 1-10\n• 3 local epochs\n• FedAvg aggregation\n• Model weights only',
        ha='center', va='center', fontsize=10)

# Add data flow information
flow_box = FancyBboxPatch((10.5, 0.5), 3, 1.5, boxstyle="round,pad=0.2",
                         facecolor='lightblue', edgecolor='black', linewidth=2)
ax.add_patch(flow_box)
ax.text(12, 1.25, 'Privacy Preservation:\n• Raw data stays local\n• Only model updates shared\n• No data leakage\n• Differential privacy ready',
        ha='center', va='center', fontsize=10)

# Add legend
legend_elements = [
    patches.Patch(facecolor=server_color, edgecolor='black', label='Central Server'),
    patches.Patch(facecolor=client_colors[0], edgecolor='black', label='IoT Clients'),
    plt.Line2D([0], [0], color=connection_color, linewidth=2, label='Secure Model Updates')
]

ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05),
          ncol=3, fontsize=10)

# Title
plt.title('Federated Learning Network Topology - Smart Energy IoT IDS', fontsize=16, fontweight='bold', pad=20)

# Add some decorative elements
# Data privacy shield
shield = patches.RegularPolygon((7, 5), 3, radius=0.3, facecolor='gold', edgecolor='black', linewidth=2)
ax.add_patch(shield)
ax.text(7, 5, '🔒', ha='center', va='center', fontsize=20)

plt.tight_layout()
plt.savefig('visualizations/federation_topology.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Federation Network Topology saved")