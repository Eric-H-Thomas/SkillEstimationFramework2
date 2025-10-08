from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import rpy2.robjects.numpy2ri as rpyn

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import code

package = importr("maotai")



# info = np.random.uniform(size=100)
# info /= np.sum(info)

shape, scale = 2., 2.  # mean=4, std=2*sqrt(2)
info = np.random.gamma(shape,scale,1000)



pandas2ri.activate()

k = 3

results = package.epmeans(info,k)


clusters = np.array(results[0])
centers = np.array(results[1])


# Define a colormap with the right number of colors
# cmap = plt.cm.get_cmap('viridis',max(clusters)-min(clusters)+1)
cmap = plt.colormaps.get_cmap('viridis')


bounds = range(min(clusters),max(clusters)+2)
norm = colors.BoundaryNorm(bounds,cmap.N)

plt.scatter(info,[1]*len(info),c=clusters,s=50,cmap=cmap,norm=norm)


# Add a colorbar. Move the ticks up by 0.5, so they are centred on the colour.
cb = plt.colorbar(ticks=np.array(clusters)+0.5)
cb.set_ticklabels(clusters)

# plt.show()

plt.savefig("clusters.png")
plt.clf()
plt.close()



for ki in range(1,k+1):
	
	fig = plt.figure()
	
	idk = np.where(clusters==ki)[0]
	
	# plt.scatter(info[[idk]][0],[1]*len(info[[idk]][0]))
	
	plt.scatter(info[[idk]][0],centers[[ki-1]][0](info[[idk]][0]), c="r")

	plt.title(f"class {ki} (size={len(idk)})")

	plt.savefig(f"{ki}.png")
	plt.clf()
	plt.close()

	# code.interact("...", local=dict(globals(), **locals()))


