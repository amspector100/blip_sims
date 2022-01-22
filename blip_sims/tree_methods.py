import numpy as np

import cvxpy as cp
import scipy.linalg
from scipy.cluster import hierarchy
import scipy.spatial.distance as ssd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from sklearn import linear_model as lm

import time
import pandas as pd
from . import gen_data
from .utilities import elapsed

# Find default solver
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS
if 'GUROBI' in INSTALLED_SOLVERS:
	DEFAULT_SOLVER = 'GUROBI'
elif 'CBC' in INSTALLED_SOLVERS:
	DEFAULT_SOLVER = 'CBC'
else:
	DEFAULT_SOLVER = 'ECOS'

def check_mle_unique(X, y):
	""" Checks if logistic reg. problem is separable """
	n, p = X.shape
	beta = cp.Variable(p)
	y = (y > 0).astype(int)
	mu = X @ beta
	constraints = [
		mu[y == 0] <= 0,
		mu[y != 0] >= 0
	]
	problem = cp.Problem(
		objective=cp.Maximize(1), constraints=constraints
	)
	problem.solve(solver=DEFAULT_SOLVER)
	if problem.status == 'infeasible':
		return True
	else:
		return False

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

def corr_matrix_to_pval_tree(corr_matrix, levels, max_size):
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
		p = corr_matrix.shape[0]
		roots = [
			PNode(group=set([j])) for j in range(p)
		]
		pTree = PTree(roots=roots, nodes=roots)
		return pTree

	# Perform hierarchical clustering
	dist_matrix = 1 - np.abs(corr_matrix)
	dist_matrix -= np.diagflat(np.diag(dist_matrix))
	dist_matrix = (dist_matrix + dist_matrix.T) / 2
	condensed_dist_matrix = ssd.squareform(dist_matrix)
	link = hierarchy.average(condensed_dist_matrix)

	# Get rid of groupings whose max group size exceeds max_size
	max_group_sizes = np.maximum.accumulate(link[:, 3])
	subset = link[max_group_sizes < max_size]

	# Create cutoffs
	spacing = max(1, int(subset.shape[0] / levels))
	cutoffs = subset[:, 2]

	# Add 0 to beginning (this is our baseline - no groups)
	# and then add spacing
	if max_size == 1:
		cutoffs = np.array([0])
	else:
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

				if node.group == parent.group:
					current_level.append(parent)
					continue

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

#### Wrappers on top of the ptree
class RegressionTree():

	def __init__(self, X, y, levels=0, max_size=25):

		# Check dimensionality
		self.n, self.p = X.shape

		# Create ptree
		self.X = X
		self.y = y
		corr_matrix = np.corrcoef(self.X.T)
		self.ptree = corr_matrix_to_pval_tree(
			corr_matrix, levels=levels, max_size=max_size
		)

	def precompute(self, family):
		"""
		Precompute some useful quantities.
		"""
		# Ensures X = U D VT
		self.Q, self.R = np.linalg.qr(self.X)
		self.H = np.dot(self.Q, self.Q.T)
		# denominator in all F tests
		if family == 'gaussian':
			self.F_test_denom = np.dot(
				np.dot(self.y, np.eye(self.n) - self.H), self.y
			) / (self.n - self.p)
		else:
			# Run logistic regression to get likelihood
			model = lm.LogisticRegression(penalty='none')
			model.fit(self.Q, self.y)
			beta = model.coef_.reshape(-1)
			mu = np.dot(self.Q, beta)
			self.ll1 = np.sum(self.y * mu)
			self.ll1 -= np.sum(np.log(1 + np.exp(mu)))

	def _downdate_Q(self, group):
		"""
		QR for X with columns in group removed
		"""
		neg_group = [j for j in range(self.p) if j not in group]
		if len(group) < 3:
			Qneg = self.Q.copy()
			Rneg = self.R.copy()
			group = sorted(list(group), key=lambda x: -1*x)
			for j in group:
				Qneg, Rneg = scipy.linalg.qr_delete(
					Q=Qneg,
					R=Rneg,
					k=j,
					p=1,
					which='col',
					overwrite_qr=False
				)
			assert np.allclose(np.dot(Qneg, Rneg), self.X[:, neg_group])
		else:
			Qneg, _ = np.linalg.qr(self.X[:, neg_group])
		return Qneg

	def F_test(self, group):
		
		# Compute Hneg, projection matrix for all outside of group
		Qneg = self._downdate_Q(group)
		Hneg = np.dot(Qneg, Qneg.T)

		# Compute F statistic
		num = np.dot(
			np.dot(self.y, self.H - Hneg), self.y
		) / (len(group))
		F = num / self.F_test_denom
		# Return p-value
		return 1.0 - stats.f.cdf(
			F, dfn=len(group), dfd=self.n-self.p
		)

	def lrt_test(self, group):
		# Precompute q r decomp for speed/numerical stability
		Qneg = self._downdate_Q(group)
		model = lm.LogisticRegression(penalty='none')
		model.fit(Qneg, self.y)
		beta = model.coef_.reshape(-1)
		mu = np.dot(Qneg, beta)
		ll0 = np.sum(self.y * mu)
		ll0 -= np.sum(np.log(1 + np.exp(mu)))
		# Compute lrt statistic
		lrt_stat = -2 * (ll0 - self.ll1)
		# return p-value
		return stats.chi2.cdf(
			lrt_stat, df=len(group)
		)

	def fit(self, family='gaussian'):

		if self.n < self.p:
			raise ValueError(
				f"Cannot perform tests since n ({self.n}) < p ({self.p})"
			)

		# For each node, compute a p-value
		self.precompute(family=family)
		# Check separability
		if family != 'gaussian':
			if not check_mle_unique(self.X, self.y):
				for node in self.ptree.nodes:
					node.p = 1
				return []

		for node in self.ptree.nodes:
			if family == 'gaussian':
				node.p = self.F_test(node.group)
			elif family == 'binomial':
				node.p = self.lrt_test(node.group)
			else:
				raise ValueError(f"Unrecognized family={family}")

### Functions for computing p-values with oracle test-statistics
### based on posterior error probabilities.
def group_metrics(gid, g, Sigma, beta):
	g = list(g)
	nnulls = np.where(beta != 0)[0]
	bnull = beta.copy()
	bnull[g] = 0
	null_cov = np.dot(Sigma, beta)
	mean_null_cov = np.mean(np.abs(null_cov[g]))
	dist_to_nnull = np.min(np.abs(
		np.array(g).reshape(1, -1) - nnulls.reshape(-1, 1)
	))
	return {
		'id':gid,
		'size':len(g),
		'group':sorted(g),
		'mean_null_cov':mean_null_cov,
		'dist_to_nnull':dist_to_nnull,
		'null':np.all(beta[g] == 0),
	}

def compute_bayesian_pvals(
	beta_fnames, 
	dgp_seed,
	sample_kwargs,
	qbins,
	levels=8,
	max_size=25,
	how_compute='ref_dist',
):
	"""
	Computes p values based on bayesian posterior error probabilities
	using binning to compute the p-values.
	"""
	t0 = time.time()
	p = sample_kwargs.get('p', 500)
	sparsity = sample_kwargs.get('sparsity', 0.05)
	p0 = 1 - sparsity
	sample_kwargs.pop('kappa', None) # unnecessary

	# Get parameters of data generating process
	X, y, beta = gen_data.generate_regression_data(
		dgp_seed=dgp_seed, 
		n=50*p,
		**sample_kwargs
	)
	print(f"Computing oracle pvals, sampling done at {elapsed(t0)}")

	Sigma = np.cov(X.T) # cov matrix
	# Create regtree to create groupings
	regtree = RegressionTree(
		X=X, y=y, levels=levels, max_size=max_size
	)
	# Calculate PEPs
	peps = dict()
	groups = [n.group for n in regtree.ptree.nodes]
	ngroups = len(groups)
	# Maps index to group
	group_attr = {}
	group_dict = {i:list(g) for i, g in enumerate(groups)}
	# Initialize metadata
	for i in range(ngroups):
		peps[i] = list()
		g = list(group_dict[i])
		group_attr[i] = group_metrics(
			gid=i, g=g, Sigma=Sigma, beta=beta
		)
	# Compute peps
	for fb in beta_fnames:
		b = np.load(fb)
		b = b != 0
		for i in range(ngroups):
			g = list(group_dict[i])
			pep = 1 - np.any(b[:, g], axis=1).mean()
			peps[i].append(pep)
	print(f"Computing oracle pvals, peps finished at {elapsed(t0)}")

	# Group by metrics based on quantiles
	group_attr = pd.DataFrame.from_dict(
		group_attr, orient='index'
	)
	group_attr['mnc_bin'] = pd.qcut(
		group_attr['mean_null_cov'], q=qbins
	)
	size_bins = [0, 1, 2, 3, 5, 10, 15]
	if max_size >= 15:
		size_bins.append(max_size + 1)
	group_attr['size_bin'] = pd.cut(
		group_attr['size'], bins=size_bins
	)
	group_attr['dist_bin'] = pd.cut(
		group_attr['dist_to_nnull'], bins=[0, 2, 4, 6, np.inf], right=False
	)

	# Bin by size of bin and sim_metric
	sim_metric = 'mnc_bin'
	if how_compute == 'bayes_rule':
		reps = len(peps[0])
		pvals = dict()
		for s in group_attr['size_bin'].unique():
			for c in group_attr[sim_metric].unique():
				sub = group_attr.loc[
					(group_attr['size_bin'] == s) &
					(group_attr[sim_metric] == c)
				]
				if sub.shape[0] == 0:
					continue
				ids = sorted(sub['id'].unique().tolist())
				# pep_bin is peps from ids concatenated in order
				sizes = []
				pep_bin = []
				for j in ids:
					size = len(group_dict[j])
					pep_bin.extend(peps[j])
					sizes.extend([size for _ in range(reps)])
				pep_bin = np.maximum(0, np.array(pep_bin)) # clip floating point errors
				sizes = np.array(sizes)
				# Compute P(pep <= observed value)
				r = pep_bin.shape[0]
				sortinds = np.argsort(pep_bin)
				rev_inds = np.zeros(r).astype(int)
				for i, j in enumerate(sortinds):
					rev_inds[j] = i
				marg = rev_inds / pep_bin.shape[0]
				# Compute P(null | Pepj <= observed value)
				sortpeps = pep_bin[sortinds]
				pnull = np.cumsum(sortpeps) / np.arange(1, r+1)
				pnull = pnull[rev_inds]
				# Compute pvals
				denom = np.exp(sizes * np.log(p0))
				pvals_bin = marg * pnull / denom
				
				# Add to pvals dictionary
				for i, j in enumerate(ids):
					pvals[j] = pvals_bin[int(i*reps):int((i+1)*reps)].tolist()
		

	# Perform binning to find reference for p-values
	else:
		ref_dict = dict()
		for s in group_attr['size_bin'].unique():
			for c in group_attr[sim_metric].unique():
				sub = group_attr.loc[
					(group_attr['size_bin'] == s) &
					(group_attr[sim_metric] == c)
				]
				if sub.shape[0] == 0:
					continue

				# have to make the bin bigger if there are no nulls
				# with which can compute a reference distribution
				if np.sum(sub['null']) == 0:
					ref = group_attr.loc[
						group_attr['size_bin'] == s
					]
					if np.sum(ref['null']) == 0:
						ref = group_attr
				else:
					ref = sub

				# Label the other indices with which to compute
				# the reference distribution
				ref_dict[(s, c)] = ref.loc[ref['null'], 'id'].unique().tolist()
		print(f"Computing oracle pvals, ref dists finished at {elapsed(t0)}")

		# Compute p-values
		pvals = dict() # maps node id to list of p-values
		ref_dist_dict = dict()
		for i in range(ngroups):
			s = group_attr.loc[i, 'size_bin']
			c = group_attr.loc[i, sim_metric]
			ref_dist_dict[i] = np.array([
				x for j in ref_dict[(s,c)] for x in peps[j]
			])
			ref_dist = ref_dist_dict[i]
			counts = np.sum(
				np.array(peps[i]).reshape(-1, 1) > ref_dist.reshape(1, -1),
				axis=1
			)
			pvals[i] = (counts + 1) / (ref_dist.shape[0] + 1)
			pvals[i] = pvals[i].tolist()

	print(f"Oracle pvals done at {elapsed(t0)}.")
	return pvals, group_attr, group_dict, peps, beta, regtree