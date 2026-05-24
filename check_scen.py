import matplotlib.pyplot as plt
import matplotlib.patches as patches
from config import *

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(XLIM); ax.set_ylim(YLIM); ax.set_aspect('equal')

# Obstacles
for rect in RECTANGLE_OBSTACLES:
    x, y, w, h = rect[0], rect[1], rect[2], rect[3]
    ax.add_patch(patches.Rectangle((x, y), w, h, facecolor='gray', edgecolor='black'))

# Waypoints
for i, wp in enumerate(TAR_WAYPOINTS):
    ax.plot(wp[0], wp[1], 'r*', markersize=15)
    ax.annotate(f'WP{i}', (wp[0], wp[1]))

# Robot starts
for i, s in enumerate(STARTS):
    ax.plot(s[0], s[1], 'go', markersize=10)
    ax.annotate(f'R{i}', (s[0], s[1]))

ax.grid(True)
ax.set_title('Scenario 5 verification')
plt.show()