"""
Template for running simulations.
"""

import os
import sys
import time

import numpy as np
import pandas as pd
from context import pyblip, blip_sims
from blip_sims import parser
from blip_sims import utilities
from blip_sims.gen_data import generate_regression_data
from statsmodels.stats.multitest import multipletests

from sklearn import linear_model

# Specifies the type of simulation
DIR_TYPE = os.path.split(os.path.abspath(__file__))[1].split(".py")[0]

# columns of dataframe
COLUMNS = [
	'method',
	'cgroups',
	'num_cands',
	'model_time',
	'blip_time',
	'power', 
	'ntd',
	'nfd',
	'fdr',
	'well_specified',
	'nsample',
	'y_dist',
	'covmethod',
	'kappa',
	'p',
	'sparsity',
	'k',
	'coeff_size',
	'delta',
	'seed', 
]

def single_seed_sim(
	seed,
	y_dist,
	covmethod,
	kappa,
	p,
	sparsity,
	k,
	coeff_size,
	delta,
	args,
	tol=1e-3
):
	np.random.seed(seed)
	output = []
	dgp_args = [
		y_dist, covmethod, kappa, p, sparsity, k, coeff_size, delta, seed
	]

	# Create data
	n = int(kappa * p)
	sample_kwargs = dict(
		y_dist=y_dist,
		covmethod=covmethod,
		n=n,
		p=p,
		sparsity=sparsity,
		k=k,
		coeff_dist=args.get('coeff_dist', ['normal'])[0],
		coeff_size=coeff_size,
		min_coeff=args.get('min_coeff', [0.1 * coeff_size])[0],
		dgp_seed=args.get("dgp_seed", [seed])[0],
		delta=delta,
		return_cov=True
	)
	X, y, beta, V = generate_regression_data(**sample_kwargs)

	# Kwargs for LSS sampling
	bsize = args.get('bsize', [1])[0]
	chains = args.get('chains', [10])[0]
	nsample = args.get('nsample', [1000])[0]
	skwargs = dict(N=nsample, chains=chains)
	if y_dist == 'gaussian':
		skwargs['bsize'] = bsize
	
	# Parse args for cand groups
	q = args.get('q', [0.1])[0]
	max_pep = args.get('max_pep', [2*q])[0]
	max_size = args.get('max_size', [25])[0]
	prenarrow = args.get('prenarrow', [0])[0]	
	levels = args.get('levels', [8])[0]

	# CRT + FBH/Yekutieli
	t0 = time.time()
	screen = args.get('screen', [False])[0]
	# Run CRT
	crt_model = blip_sims.crt.MultipleDCRT(
		y=y, X=X, Sigma=V, screen=screen, suppress_warnings=args.get("suppress_warnings", [True])[0]
	)
	crt_model.create_tree(levels=levels, max_size=max_size)
	model_type = args.get('model_type', ['lasso'])[0]
	#full_pval = args.get('full_pval', [False])[0]
	M = args.get("m", [200])[0]
	for full_pval in args.get('full_pval', [False]):
		if model_type != 'bayes' and not full_pval:
			crt_kwargs = dict(
				max_iter=args.get('max_iter', [500])[0],
				model_type=model_type,
				tol=args.get('tol', [5e-3])[0],
			)
		else:
			crt_kwargs = dict(
				p0=1-sparsity,
				update_p0=False,
				tau2=coeff_size,
				update_tau2=False,
				sigma2=1,
				update_sigma2=False,
			)
			if not full_pval:
				crt_kwargs['model_type'] = model_type

		# Cheating for fast testing by setting null pvals to be uniform
		# (not for use in final simulations)
		if args.get('cheat_null_pvals', [1])[0]:
			for node in crt_model.pTree.nodes:
				if np.all(beta[list(node.group)] == 0):
					node.p = np.random.uniform()
				elif full_pval:
					node.p = crt_model.full_p_value(
						inds=list(node.group),
						M=M,
						params=crt_kwargs,
						sample_kwargs=skwargs,
					)
				else:
					node.p = crt_model.p_value(
						inds=list(node.group),
						node=node,
						**crt_kwargs
					)
		else:
			crt_model.multiple_pvals(
				max_size=max_size,
				levels=levels,
				**crt_kwargs
			)


		crt_mtime = time.time() - t0
		_, rej_yek = crt_model.pTree.outer_nodes_yekutieli(q=q)
		rej_fbh, _ = crt_model.pTree.tree_fbh(q=q)
		nnodes = len(crt_model.pTree.nodes)
		for mname, rej in zip(['CRT + FBH', 'CRT + Yekutieli'], [rej_fbh, rej_yek]):
			nfd, fdr, power = utilities.nodrej2power(rej, beta)
			ntd = len(rej) - nfd
			if not full_pval:
				mname = 'd' + mname # dCRT
			output.append(
				[mname, "fbh", nnodes, crt_mtime, 0, power, ntd, nfd, fdr, True, 0] + dgp_args
			)

		# CRT and BLiP at various fixed levels
		mname = 'CRT' if full_pval else 'dCRT'
		for level in range(levels+1):
			level_nodes = crt_model.levels[level]
			level_rej = multipletests(
				[n.p for n in level_nodes], alpha=q, method='fdr_bh',
			)[0]
			level_rej = [n for (n,flag) in zip(level_nodes, level_rej) if flag == True]
			l_nfd, l_fdr, l_power = utilities.nodrej2power(
				level_rej, beta
			)
			l_ntd = len(level_rej) - l_nfd
			output.append(
				[
					mname, level, len(level_nodes), crt_mtime, 0, l_power, 
					l_ntd, l_nfd, l_fdr, True, 0
				] + dgp_args
			)

	# Gibbs + BLiP
	for well_specified in args.get('well_specified', [False, True]):
		if well_specified:
			p0 = 1 - sparsity
			min_p0 = 0
			tau2 = coeff_size
			sigma2 = 1
			update = False
		else:
			p0 = 0.99
			min_p0 = 0.9
			tau2 = 1
			sigma2 = 1
			update = True

		# Method type 1: BLiP + SpikeSlab
		# Run Gaussian sampler if (1) not well-specified or (2) gaussian response
		if y_dist == 'gaussian' or not well_specified:
			lss_model = pyblip.linear.LinearSpikeSlab(
				X=X,
				y=y.astype(np.float64),
				p0=p0,
				min_p0=min_p0,
				update_p0=update,
				tau2=tau2,
				update_tau2=update,
				sigma2=sigma2,
				update_sigma2=update,
			)
			models = [lss_model]
			method_names = ['LSS + BLiP']
		else:
			models = []
			method_names = []
		if y_dist != 'gaussian':
			models.append(pyblip.probit.ProbitSpikeSlab(
				X=X, y=y.astype(int), p0=p0, update_p0=update, #min_p0=min_p0 TODO
			))
			method_names.append('PSS + BLiP')

		for model, mname in zip(models, method_names):
			t0 = time.time()
			model.sample(**skwargs)
			inclusions = model.betas != 0
			mtime = time.time() - t0
			# Calculate PIPs and cand groups
			for cgroup in args.get('cgroups', ['all', 'fbh'] + list(range(levels+1))):
				t0 = time.time()
				if cgroup == 'all':
					cand_groups = pyblip.create_groups.all_cand_groups(
						inclusions=inclusions,
						q=q,
						max_pep=max_pep,
						max_size=max_size,
						prenarrow=prenarrow
					)
				elif cgroup == 'seq':
					cand_groups = pyblip.create_groups.sequential_groups(
						inclusions,
						q=q,
						max_pep=max_pep,
						max_size=max_size,
						prenarrow=prenarrow,
					)
				elif cgroup == 'fbh' or isinstance(cgroup, int):
					if cgroup == 'fbh':
						groups = [x.group for x in crt_model.pTree.nodes]
					else:
						groups = [x.group for x in crt_model.levels[cgroup]]
					peps = [
						1 - np.any(inclusions[:, list(g)], axis=1).mean() for g in groups
					]
					cand_groups = [
						pyblip.create_groups.CandidateGroup(
							pep=pep, group=g
						) for pep, g in zip(peps, groups)
					]
				else:
					raise ValueError(f"Unrecognized cgroup={cgroup}")

				# Run BLiP
				detections = pyblip.blip.BLiP(
					cand_groups=cand_groups,
					q=q,
					error='fdr',
					max_pep=max_pep,
					perturb=True,
					deterministic=True
				)
				blip_time = time.time() - t0
				nfd, fdr, power = utilities.nodrej2power(detections, beta)
				ntd = len(detections) - nfd
				output.append(
					[mname, cgroup, len(cand_groups), mtime, blip_time, power, ntd, nfd, fdr, well_specified, nsample] + dgp_args
				)
		
	return output

def main(args):
	# Parse arguments
	args = parser.parse_args(args)
	print(args['description'])
	sys.stdout.flush()
	reps = args.get('reps', [1])[0]
	num_processes = args.get('num_processes', [1])[0]

	# Save args, create output dir
	output_dir, today, hour = utilities.create_output_directory(
		args, dir_type=DIR_TYPE, return_date=True
	)
	result_path = output_dir + "/results.csv"

	# Run outputs
	time0 = time.time()
	all_outputs = []
	seed_start = max(args.get('seed_start', [1])[0], 1)
	for covmethod in args.get('covmethod', ['ark']):
		for p in args.get('p', [500]):
			for kappa in args.get('kappa',[0.2]):
				for sparsity in args.get('sparsity', [0.05]):
					for k in args.get('k', [1]):
						for y_dist in args.get('y_dist', ['gaussian']):
							for coeff_size in args.get('coeff_size', [1]):
								for delta in args.get('delta', [1]):
									constant_inputs=dict(
										y_dist=y_dist,
										covmethod=covmethod,
										kappa=kappa,
										p=p,
										sparsity=sparsity,
										k=k,
										coeff_size=coeff_size,
										delta=delta,
									)
									msg = f"Finished with {constant_inputs}"
									constant_inputs['args'] = args
									outputs = utilities.apply_pool(
										func=single_seed_sim,
										seed=list(range(seed_start, reps+seed_start)), 
										constant_inputs=constant_inputs,
										num_processes=num_processes, 
									)
									msg += f" at {np.around(time.time() - time0, 2)}"
									print(msg)
									for out in outputs:
										all_outputs.extend(out)

									# Save
									out_df = pd.DataFrame(all_outputs, columns=COLUMNS)
									out_df.to_csv(result_path, index=False)
									groupers = [
										'cgroups', 'delta', 'method', 'y_dist', 'covmethod', 'kappa', 
										'p', 'sparsity', 'k', 'well_specified', 'nsample'
									]
									meas = ['model_time', 'blip_time', 'num_cands', 'power', 'ntd', 'fdr']
									summary_df = out_df.groupby(groupers)[meas].mean()
									print(summary_df.reset_index())



if __name__ == '__main__':
	main(sys.argv)