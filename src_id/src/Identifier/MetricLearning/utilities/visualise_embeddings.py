# Core libraries
import os
import sys
import cv2
import argparse
import numpy as np

# Matplotlib / TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.patheffects as PathEffects

# Define our own plot function
def scatter(x, labels, filename, highlight=True):
	# Get the number of classes (number of unique labels)
	# num_classes = np.unique(labels).shape[0]

	# Choose a color palette with seaborn.
	palette = np.array(sns.color_palette("hls", 200))

	# Randomly shuffle with the same seed
	np.random.seed(42)
	np.random.shuffle(palette)

	# Convert labels to int
	labels = labels.astype(int)

	# Map the colours to different labels
	label_colours = np.array([palette[labels[i]] for i in range(labels.shape[0])])

	# Create our figure/plot
	f = plt.figure(figsize=(8, 8))
	ax = plt.subplot(aspect='equal')

	# Do we want to highlight some particular (e.g. difficult) labels
	if highlight:
		# Which labels should we highlight (the "difficult" individuals)
		highlight_labels = [54, 69, 73, 173]

		# Colour for non-highlighted points
		label_colours = np.zeros(label_colours.shape)

		# Alpha value for non-highlighted points
		alpha = 1.0

		# Plot all the points with some transparency
		ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=label_colours, marker="o", alpha=alpha)

		# Highlighted points
		h_pts = np.array([x[i,:] for i in range(labels.shape[0]) if labels[i] in highlight_labels])

		# Colours
		h_colours = np.array([palette[labels[i]] for i in range(labels.shape[0]) if labels[i] in highlight_labels])

		# There may not have been any occurences of that label
		if h_pts.size != 0:
			# Replot highlight points with no alpha
			ax.scatter(h_pts[:,0], h_pts[:,1], lw=0, s=40, c=h_colours, marker="o")
		else:
			print(f"Didn't find any embeddings with the label: {highlight_labels}")

	# Just colour each point normally
	else:
		# Plot the points
		ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=label_colours, marker="o")

	# Do some formatting
	plt.xlim(-25, 25)
	plt.ylim(-25, 25)
	ax.axis('off')
	ax.axis('tight')
	plt.tight_layout()

	# Save it to file
	# plt.show()
	plt.savefig(filename+".pdf")

# Load and visualise embeddings via t-SNE
def plotEmbeddings(args):
	# Ensure there's something there
	if not os.path.exists(args.embeddings_file):
		print(f"No embeddings file at path: {args.embeddings_file}, exiting.")
		sys.exit(1)

	# Load the embeddings into memory
	embeddings = np.load(args.embeddings_file)

	print("Loaded embeddings")

	# Visualise the learned embedding via t-SNE
	visualiser = TSNE(n_components=2, perplexity=args.perplexity)

	# Reduce dimensionality
	reduction = visualiser.fit_transform(embeddings['embeddings'])

	print("Visualisation computed")

	# Plot the results and save to file
	scatter(reduction, embeddings['labels'], os.path.basename(args.embeddings_file)[:-4])

# Main/entry method
if __name__ == '__main__':
	# Collate command line arguments
	parser = argparse.ArgumentParser(description='Parameters for visualising the embeddings via TSNE')
	parser.add_argument('--embeddings_file', type=str, required=True,
						help="Path to embeddings .npz file you want to visalise")
	parser.add_argument('--perplexity', type=int, default=30,
						help="Perplexity parameter for t-SNE, consider values between 5 and 50")
	args = parser.parse_args()

	# Let's plot!
	plotEmbeddings(args)
	