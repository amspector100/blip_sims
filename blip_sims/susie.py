import numpy as np
import rpy2
import rpy2.robjects.numpy2ri as numpy2ri
import rpy2.robjects as ro
from rpy2.rinterface_lib.embedded import RRuntimeError

def run_susie(
	X,
	y,
	L,
	q,
	**kwargs
):
	# This code adapted from the polyfun package
	# load SuSiE R package
	ro.conversion.py2ri = numpy2ri
	numpy2ri.activate()
	from rpy2.robjects.packages import importr
	susieR = importr('susieR')
	R_null = ro.rinterface.NULL
	# Run susie
	susie_obj = susieR.susie(
		X=X, y=y, L=L, coverage=1-q, **kwargs
	)
	# Extract output
	alphas = susie_obj.rx2('alpha')
	susie_sets = susie_obj.rx2('sets')[0]
	try:
		susie_sets = [
			np.array(s)-1 for s in susie_sets
		]
	except TypeError:
		susie_sets = []

	return alphas, susie_sets

def run_susie_trendfilter(
	x,
	order,
	L,
	q,
	**kwargs
):
		# This code adapted from the polyfun package
	# load SuSiE R package
	ro.conversion.py2ri = numpy2ri
	numpy2ri.activate()
	from rpy2.robjects.packages import importr
	susieR = importr('susieR')
	R_null = ro.rinterface.NULL
	# Run susie
	try:
		susie_obj = susieR.susie_trendfilter(
			y=x, order=order, L=L, coverage=1-q, **kwargs
		)
	except RRuntimeError:
		return None, []
	# Extract output
	alphas = susie_obj.rx2('alpha')
	susie_sets = susie_obj.rx2('sets')[0]
	try:
		susie_sets = [
			np.array(s)-1 for s in susie_sets
		]
	except TypeError:
		susie_sets = []

	return alphas, susie_sets
