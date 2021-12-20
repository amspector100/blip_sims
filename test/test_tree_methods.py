import numpy as np
import unittest
import pytest
from test_context import blip_sims
from blip_sims import tree_methods as tm

def lists_to_pval_tree(
	groups, pvals, parents
):
	node_ids = np.arange(len(pvals)).astype(int)
	nodes = [
		tm.PNode(p=p, group=group) for p, group in zip(pvals, groups)
	]
	for j in node_ids:
		nodes[j].node_id = j
		if parents[j] is not None:
			parent_node = nodes[parents[j]]
			nodes[j].parent = parent_node
			parent_node.children.append(nodes[j])
		else:
			nodes[j].parent = None
	return tm.PTree(nodes=nodes)


class TestTreeMethods(unittest.TestCase):

	def test_multiple_tests(self):
		"""
		Make sure FBH / yekutieli run correctly.
		"""

		# Create simple p-value tree
		# Level 0: 0.001
		# Level 1: 0.03, 0.1, 0.3, 0.01
		# Level 2: 0.1, 0.2 coming from the 0.03 node.
		num_nodes = 8
		groups = [set([]) for _ in range(num_nodes)] # not important
		pvals = [
			0.001, 0.1, # level 0
			0.03, 0.1, 0.3, 0.01, # level 1, descendents of node 0
			0.1, 0.2 # level 2, decscendents of the 0.03 node in level 1
		]
		parents = [None, None, 0, 0, 0, 0, 2, 2]
		pvalTree = lists_to_pval_tree(pvals=pvals, groups=groups, parents=parents)
		rej, outer_rej = pvalTree.outer_nodes_yekutieli(q=0.1)
		outer_rej_ids = set([n.node_id for n in outer_rej])
		expected = set([1, 2, 5])
		self.assertEqual(
			outer_rej_ids,
			expected,
			f"Outer nodes yekutieli yields incorrect outer nodes rej: {outer_rej_ids} vs {expected}"
		)
		rej_ids = set([n.node_id for n in rej])
		expected_rej_ids = set([0, 1, 2, 5])
		self.assertEqual(
			rej_ids,
			expected_rej_ids,
			f"Outer nodes yekutieli yields incorrect rejections: {rej_ids} vs {expected_rej_ids}"
		)

		# Try again with focused BH
		rej, threshold = pvalTree.tree_fbh(q=0.1)
		self.assertEqual(
			threshold, 
			0.01, 
			f"Focused BH yields incorrect threshold"
		)
		rej_ids = set([node.node_id for node in rej])
		expected = set([5])
		self.assertEqual(
			rej_ids, expected, f"Focused BH yields incorrect outer nodes"
		)

if __name__ == "__main__":
	unittest.main()
