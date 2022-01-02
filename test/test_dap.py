import numpy as np
from scipy import stats
import unittest
import pytest
from test_context import blip_sims
from blip_sims import dap, gen_data

class TestDap(unittest.TestCase):
	""" Tests ability to run dap from python """
	def test_dap(self):

		# Fake regression data
		np.random.seed(111)
		sparsity = 0.01
		q = 0.99
		X, y, beta = gen_data.generate_regression_data(
			y_dist='gaussian',
			covmethod='ark',
			n=500,
			p=300,
			sparsity=sparsity,
			k=1,
			coeff_dist='normal',
			coeff_size=1,
			min_coeff=0.1
		)

		# Run dap
		clusters, pips, models = dap.run_dap(
			X=X, y=y, file_prefix="test/dap_test_data/",
			pi1=str(sparsity),
			q=q,
		)
		nfd = sum([np.all(beta[c]==0) for c in clusters])
		fdr = nfd / max(len(clusters), 1)
		# Very crude sanity check
		self.assertTrue(
			fdr < 0.5,
			f"FDR={fdr} for DAP with q={q}"
		)

if __name__ == "__main__":
	unittest.main()

