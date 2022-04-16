import numpy as np
from scipy import stats
import unittest
import pytest
from test_context import blip_sims, pyblip
from blip_sims import finemap, gen_data

class TestFinemap(unittest.TestCase):
	""" Tests ability to run dap from python """
	def test_finemap(self):

		# Fake regression data
		np.random.seed(111)
		q = 0.1
		sparsity = 0.02
		X, y, beta = gen_data.generate_regression_data(
			y_dist='gaussian',
			covmethod='ark',
			n=200,
			p=300,
			sparsity=sparsity,
			k=1,
			coeff_dist='normal',
			coeff_size=1,
			min_coeff=0.1
		)

		# Run dap
		csets, cand_groups = finemap.run_finemap(
			X=X, 
			y=y,
			file_prefix="test/finemap_test_data/seed0",
			q=q,
			pi1=sparsity,
			max_nsignal=10,
			max_pep=0.25,
		)
		
		# Check FDR for original method
		nfd = sum([np.all(beta[list(c)]==0) for c in csets])
		fdr = nfd / max(len(csets), 1)
		# Very crude sanity check
		self.assertTrue(
			fdr < 0.5,
			f"FDR={fdr} for FINEMAP with q={q}"
		)

		# Check FDR for original method + BLiP
		detections = pyblip.blip.BLiP(
			cand_groups=cand_groups,
			q=q,
			error='fdr',
			max_pep=0.25,
			verbose=True,
		)
		nfd = sum([np.all(beta[list(cg.group)] == 0) for cg in detections])
		blip_fdp = nfd / len(detections)
		self.assertTrue(
			blip_fdp < 0.5,
			f"FDR={blip_fdp} for FINEMAP+BLiP with q={q}"
		)

if __name__ == "__main__":
	unittest.main()

