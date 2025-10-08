import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.graph_objects as go

import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors


sortedParticles = np.array([[ 0.0473    ,  0.1186    ,  0.4978    ,  0.00102015],
	   [ 0.5123    ,  0.1745    , -0.2604    ,  0.00102045],
	   [ 0.4684    ,  0.2318    ,  0.166     ,  0.00102108],[ 0.0473    ,  0.1186    ,  0.4978    ,  0.00102015],
	   [ 0.5123    ,  0.1745    , -0.2604    ,  0.00102045],
	   [ 0.4684    ,  0.2318    ,  0.166     ,  0.00102108],[ 0.0473    ,  0.1186    ,  0.4978    ,  0.00102015],
	   [ 0.5123    ,  0.1745    , -0.2604    ,  0.00102045],
	   [ 0.4684    ,  0.2318    ,  0.166     ,  0.00102108]])


probs = [0.00099198,
	   0.00107265,
	   0.000992 ,
	   0.00099198,
	   0.00099199,
	   0.00099198,
	   0.00105562,
	   0.00100602,
	   0.0010362]


norm = mcolors.Normalize(vmin=min(probs), vmax=max(probs))
colormap = cm.get_cmap("Blues")  # Choose a color map

# 10 * 100

# Parameters
grid_size = 50 * 20
square_size = 15.0 # Side length of each small square

# Create a figure and axis
# fig, ax = plt.subplots(figsize=(16,9))

fig = go.Figure(layout=go.Layout(width=1300,height=900))


pi = 0

x_centers = []
y_centers = []
hover_labels = []

grid_width = int(grid_size * 1.5) 



# Loop to place squares in a grid
for i in range(50):
	for j in range(int(20)):

		# particle = sortedParticles[pi]

		# prob = probs[pi]  # Probability for this square
		# color = mcolors.to_hex(colormap(norm(prob))) 

		pi += 1


		x = j * square_size
		y = (grid_size - 1 - i) * square_size  # Invert y-axis to have (0, 0) in bottom-left

		# square = patches.Rectangle((x, y), square_size, square_size, edgecolor='blue', facecolor='cyan')
		# ax.add_patch(square)

		x0 = j * (2 * square_size)  # Keep square size fixed
		y0 = (grid_size - 1 - i) * (2 * square_size)

		x1 = x0 + (2 * square_size)
		y1 = y0 + (2 * square_size)



		fig.add_shape(
			type="rect",
			x0=x0, y0=y0, x1=x1, y1=y1,
			line=dict(color="black"),
			# fillcolor=color
		)

		x_centers.append((x0 + x1) / 2)
		y_centers.append((y0 + y1) / 2)
		# hover_labels.append(f"({particle})")  




# To remove the whole axis
fig.update_layout(
    xaxis={'visible': False},
    yaxis={'visible': False}
)

# To remove grid lines
fig.update_layout(
    xaxis={'showgrid': False},
    yaxis={'showgrid': False}
)

# To remove tick labels
fig.update_layout(
    xaxis={'showticklabels': False},
    yaxis={'showticklabels': False}
)
		
fig.add_trace(go.Scatter(
	x=x_centers, 
	y=y_centers, 
	text=hover_labels,
	mode="markers",
	marker=dict(size=1, color="rgba(0,0,0,0)"),  # Invisible markers
	hoverinfo="text"
))


fig.show()
