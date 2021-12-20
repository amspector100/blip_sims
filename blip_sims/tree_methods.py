import numpy as np
from scipy.cluster import hierarchy
from statsmodels.stats.multitest import multipletests

class PNode():

	def __init__(
		self,
		group, # group of features,
		p=None, # p-value
		parent=None,
		children=None,
		node_id=None
	):
		self.p = p
		self.group = set(group)
		self.parent = parent
		self.children = children if children is not None else []
		self.synth_root = False
		self.node_id = node_id

	def mark_synth_root(self):
		self.synth_root = True

class PTree():

	def __init__(self, nodes):
		# Find roots of tree
		self.roots = [
			n for n in nodes if n.parent is None
		]
		self.nodes = nodes

		# Synthetic root for whole tree
		self._synth_root = PNode(
			node_id=-1,
			group=set([]),
			parent=None,
			children=self.roots,
		)
		self._synth_root.mark_synth_root()

	def _single_root_yekutieli(self, node, q=0.1):
		"""
		Runs Yekutieli for one node and its descendants.
		"""
		# If node is not marginally significant, we're done
		if not node.synth_root:
			if node.p > q:
				return [], []

		# Test children
		rej = [node] if not node.synth_root else []
		outer_rej = []
		if len(node.children) == 0:
			if not node.synth_root:
				outer_rej.append(node)
			return rej, outer_rej

		child_pvals = [x.p for x in node.children]
		child_rej = multipletests(
			child_pvals, alpha=q, method='fdr_bh',
		)[0]
		# No children are rejected
		if np.all(child_rej == 0):
			if not node.synth_root:
				outer_rej.append(node)
			return rej, outer_rej
		# Recursively apply to all children
		else:
			threshold = max([
				p for p, flag in zip(child_pvals, child_rej) if flag
			])
			for child in node.children:
				if child.p <= threshold:
					desc_rej, outer_desc_rej = self._single_root_yekutieli(
						node=child, q=q
					)
					rej.extend(desc_rej)
					outer_rej.extend(outer_desc_rej)
		return rej, outer_rej

	def outer_nodes_yekutieli(self, q=0.1):
		"""
		Outer-nodes FDR control based on 
		Hierarchical False Discovery Rate-Controlling Methodology
		(Yekutieli 2008).
		"""
		return self._single_root_yekutieli(self._synth_root, q=q)

	def tree_fbh(self, q=0.1):
			"""
			Focused-BH for outer-nodes control on trees.
			"""
			# Iterate through pvals from largest to smallest
			all_pvals = sorted(
				np.unique([node.p for node in self.nodes]),
				key = lambda x: -1*x
			)
			for threshold in all_pvals:
				R = [node for node in self.nodes if node.p <= threshold]
				outer_nodes = [
					node for node in R
					if np.all(np.array([x.p for x in node.children]) > threshold)
				]
				# FDP estimate
				hat_FDP = len(self.nodes) * threshold / len(outer_nodes)
				if hat_FDP <= q:
					return outer_nodes, threshold
			
			# If nothing controls FDP estimate, 
			return [], 0

	def find_subtree(self, node):
		"""
		Returns a list of node + all nodes descending from node.
		"""
		if len(node.children) == 0:
			return [node]
		else:
			output = [node]
			for child in node.children:
				output.extend(self.find_subtree(child))
			return output

def dist_matrix_to_pval_tree(dist_matrix, levels=0, max_size=25, **kwargs):
	"""
	Hierarchically clusters based on a distance matrix,
	then cuts the clustering tree at ``levels`` different
	locations between no groupings and the first grouping
	exceeding the max_size.

	Returns
	-------
	pTree : A list of PNodes comprising a tree.
	"""
	if levels == 0:
		p = dist_matrix.shape[0]
		roots = [
			PNode(group=set([j])) for j in range(p)
		]
		pTree = PTree(roots=roots, nodes=roots)
		return pTree

	# Perform hierarchical clustering
	dist_matrix -= np.diagflat(np.diag(dist_matrix))
	condensed_dist_matrix = ssd.squareform(dist_matrix)
	link = hierarchy.average(condensed_dist_matrix)

	# Get rid of groupings whose max group size exceeds max_size
	max_group_sizes = np.maximum.accumulate(link[:, 3])
	subset = link[max_group_sizes < max_size]

	# Create cutoffs
	spacing = int(subset.shape[0] / levels)
	cutoffs = subset[:, 2]

	# Add 0 to beginning (this is our baseline - no groups)
	# and then add spacing
	cutoffs = np.insert(cutoffs, 0, 0)
	cutoffs = cutoffs[::spacing]

	# If cutoffs aren't unique, only consider some
	# (This occurs only in simulated data)
	if np.unique(cutoffs).shape[0] != cutoffs.shape[0]:
		cutoffs = np.unique(cutoffs)
		
	# Sort so we move from the top of the tree to the bottom
	cutoffs = np.flip(np.sort(cutoffs))
		
	# Cut the tree at the cutoffs and add them to a tree.
	groupings = []
	pTree = []
	roots = []
	prev_level = []
	current_level = []
	for level, cutoff in enumerate(cutoffs):
		groups = hierarchy.fcluster(link, cutoff, criterion="distance")
		groupings.append(groups)
		for group_id in np.unique(groups):
			group = np.where(groups == group_id)[0]
			node = PNode(group=set(group.tolist()))
			# Leaf nodes
			if level == 0:
				roots.append(node)
			else:
				# Representative
				rep = group[0]
				parent = None
				for prev_node in prev_level:
					if rep in prev_node.group:
						parent = prev_node
						break
				if parent is None:
					raise ValueError(f"Unexpectedly could not find parent for node={node}")
				# Parent/child information
				node.parent = parent
				parent.children.append(node)
				
			pTree.append(node)
			current_level.append(node)
		
		# Signal that we're moving down a level in the tree
		prev_level = current_level
		current_level = []

	# pTree
	pTree = PTree(nodes=pTree)
	return pTree