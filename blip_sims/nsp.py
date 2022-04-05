import numpy as np
import pandas as pd
import pyblip
import rpy2
import rpy2.robjects.numpy2ri as numpy2ri
import rpy2.robjects as ro
from rpy2.rinterface_lib.embedded import RRuntimeError

def run_nsp(
	y,
	q,
	sigma2=1
):
	"""
	Runs nsp on changepoint data y with significance 
	level sigma2.
	"""
	ro.conversion.py2ri = numpy2ri
	numpy2ri.activate()
	from rpy2.robjects.packages import importr
	nsp = importr('nsp')
	nsp_fit = nsp.nsp_poly(
		y, 
		sigma=1,
		alpha=0.05
	)
	interval_df = pd.DataFrame(nsp_fit.rx2("intervals"))
	detections = []
	for j in range(interval_df.shape[0]):
		start = interval_df.iloc[j]['starts']
		end = interval_df.iloc[j]['ends']
		detections.append(np.arange(start, end+1, 1).astype(int))

	return detections