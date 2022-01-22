import numpy as np
import rpy2
import rpy2.robjects.numpy2ri as numpy2ri
import rpy2.robjects as ro
from rpy2.rinterface_lib.embedded import RRuntimeError

def run_bcp(
	y,
	nsample=10000,
	chains=5,
	**kwargs
):
	ro.conversion.py2ri = numpy2ri
	numpy2ri.activate()
	from rpy2.robjects.packages import importr
	bcp = importr('bcp')
	#R_null = ro.rinterface.NULL
	# Assemble inputs to bcp
	kwargs['return.mcmc'] = True
	kwargs['mcmc'] = nsample
	kwargs['burnin'] = kwargs.get('burnin', int(0.1*nsample))
	burnin = kwargs.get('burnin')
	# Run chains
	for _ in range(chains):
		bcp_obj = bcp.bcp(y=y, **kwargs)
		# Extract output
		inclusions = bcp_obj.rx2('mcmc.rhos').T
		# Process for consistency with BLiP / susie (no burnin)
		inclusions = inclusions[burnin:]
		inclusions = np.concatenate(
			[np.zeros((nsample, 1)),
			 inclusions[:, 0:-1]],
			 axis=1
		)

	return inclusions