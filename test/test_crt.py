import numpy as np
from scipy import stats
import unittest
import pytest
from test_context import blip_sims
from blip_sims import crt, dap, gen_data, tree_methods


class TestDCRT(unittest.TestCase):
    """ Tests sample_data function """

    def test_single_pval(self):
        """
        Check marginal distributions of p-values are uniform.
        """
        np.random.seed(110)
        n = 100
        p = 30
        mu = np.zeros((p,))
        reps = 200
        grouped_pvals = []
        grouped_pvals_max = []
        ungrouped_pvals = []
        for j in range(reps):
            # Sample data
            X, y, beta, Sigma = gen_data.generate_regression_data(
                n=n, p=p, covmethod='ark', k=1, return_cov=True
            )
            nulls = np.where(beta == 0)[0]
            # Run CRT
            crt_model = crt.MultipleDCRT(
                y=y, X=X, Sigma=Sigma, screen=False
            )
            ungrouped_pvals.append(crt_model.p_value(inds=int(nulls[0])))
            # Hacky way to save computation
            node = tree_methods.PNode(p=None, group=set(nulls[1:]))
            grouped_pvals.append(crt_model.p_value(inds=nulls[1:], node=node, agg='l2'))
            grouped_pvals_max.append(crt.maxpval(node.data['inner_product']))

        # Check that null distribution is uniform
        for pvals in ungrouped_pvals, grouped_pvals, grouped_pvals_max:
            pvals = np.array(pvals)
            _, nonuniform_pval = stats.kstest(pvals, "uniform")
            self.assertTrue(
                nonuniform_pval > 1e-2, 
                f"CRT p-values are not uniform: KS Test p-value is {nonuniform_pval}"
            )

    def test_multiple_pvals(self):
        """ Just makes sure that multiple_pval doesn't error in 
        a real application."""

        # Sample data
        np.random.seed(110)
        n = 200
        p = 10#0
        X, y, beta = gen_data.generate_regression_data(
            p=p, covmethod='ark', n=n, sparsity=0.1,
        )
        Sigma = np.corrcoef(X.T)
        # Run CRT
        crt_model = crt.MultipleDCRT(y=y, X=X, Sigma=Sigma, screen=False)
        crt_model.multiple_pvals(levels=2, max_size=30)

        # Yekutieli, focused BH
        crt_model.pTree.outer_nodes_yekutieli(q=0.1)
        crt_model.pTree.tree_fbh(q=0.1)

        # print(beta)
        # for node in crt_model.pTree.nodes:
        #     print(node.group, node.p)
        # raise ValueError()


if __name__ == "__main__":
    unittest.main()
