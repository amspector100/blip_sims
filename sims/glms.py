"""
Template for running simulations.
"""

import os
import sys
import time
import copy

import numpy as np
import pandas as pd
from context import pyblip, blip_sims
from blip_sims import parser
from blip_sims import utilities
from blip_sims.gen_data import generate_regression_data

from sklearn import linear_model

# Specifies the type of simulation
DIR_TYPE = os.path.split(os.path.abspath(__file__))[1].split(".py")[0]

# columns of dataframe
MAX_COUNT_SIZE = 25
COLUMNS = [
	'method',
	'cgroups',
	'model_time',
	'blip_time',
	'well_specified',
	'nsample',
	'y_dist',
	'covmethod',
	'kappa',
	'p',
	'sparsity',
	'k',
	'coeff_size',
	'seed', 
	# metrics,
	'power', 
	'ntd',
	'nfd',
	'fdr',
	'med_group_size',
	'n_indiv_disc',
	'med_purity',
]
# n disc
for j in range(MAX_COUNT_SIZE):
	COLUMNS.append(f"ntd_le_{j+1}")

def purity(region, hatSig):
	if len(region) == 1:
		return 1.0
	hatSigR = hatSig[region][:, region]
	return np.abs(hatSigR).min()

def power_fdr_metrics(rej, beta, hatSig, how='cgs'):
	# depending on the object involved call different fns
	if how != 'cgs':
		# main metrics
		nfd, fdr, power = utilities.rejset_power(rej, beta=beta)
		# true positive flags + gsizes
		tp_flags = np.array([np.any(beta[s] != 0) for s in rej])
		gsizes = np.array([len(x) for x in rej])
		# other metrics
		med_group_size = np.median(gsizes)
		n_indiv_disc = len([x for x in rej if len(x) == 1 and np.any(beta[x]) != 0])
		med_purity = np.median([purity(x, hatSig) for x in rej])
		ntd = len(rej) - nfd
		metrics = [power, ntd, nfd, fdr, med_group_size, n_indiv_disc, med_purity]
		for j in range(1, MAX_COUNT_SIZE+1):
			freqj = np.sum(((gsizes <= j) * (tp_flags)).astype(int))
			metrics.append(freqj)
		return metrics 
	else:
		return power_fdr_metrics(
			[list(x.group) for x in rej], 
			beta=beta,
			hatSig=hatSig,
			how='rejset'
		)

def single_seed_sim(
	global_t0,
	seed,
	y_dist,
	covmethod,
	kappa,
	p,
	sparsity,
	k,
	coeff_size,
	args,
	dap_prefix,
	tol=1e-3
):
	np.random.seed(seed)
	output = []
	dgp_args = [
		y_dist, covmethod, kappa, p, sparsity, k, coeff_size, seed
	]
	print(f"At seed={seed}, dgp_args={dgp_args} at time {utilities.elapsed(global_t0)}.")
	sys.stdout.flush()

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
		alpha0=args.get("alpha0", [0.2])[0],
		max_corr=args.get("max_corr", [0.99])[0],
		min_coeff=args.get('min_coeff', [0.1 * coeff_size])[0],
		dgp_seed=seed,
		return_cov=True
	)
	X, y, beta, V = generate_regression_data(**sample_kwargs)
	hatSig = np.corrcoef(X.T) # for purity computations

	# Parse args for cand groups
	q = args.get('q', [0.1])[0]
	max_pep = args.get('max_pep', [2*q])[0]
	max_size = args.get('max_size', [25])[0]
	prenarrow = args.get('prenarrow', [0])[0]	
	levels = args.get('levels', [8])[0]
	finemap_chains = args.get("finemap_chains", [5])[0]

	# Method 0: DAP
	if p <= 1000 and args.get("run_dap", [False])[0]:
		t0 = time.time()
		rej_dap, _, _ = blip_sims.dap.run_dap(
			X=X, 
			y=y, 
			q=q, 
			file_prefix=dap_prefix + str(seed), 
			pi1=str(sparsity),
			msize=str(int(1.1 * sparsity * p)),
		)
		dap_time = time.time() - t0
		metrics = power_fdr_metrics(rej_dap, beta=beta, hatSig=hatSig, how='rejset')
		output.append(
			["dap-g", "NA", dap_time, 0] +  [True, 0] + dgp_args + metrics
		)
		print(f"At seed={seed}, dgp_args={dgp_args}, finished DAP at time {utilities.elapsed(global_t0)}.")
		sys.stdout.flush()
		
	# if args.get("run_finemap", [False])[0]:
	# 	t0 = time.time()
	# 	rej_finemap, cand_groups = blip_sims.finemap.run_finemap(
	# 		file_prefix=dap_prefix + "_finemap" + str(seed),
	# 		X=X,
	# 		y=y,
	# 		q=q,
	# 		pi1=args.get("finemap_pi1", [1 / p])[0],
	# 		max_nsignal=args.get("max_nsignal", [int(1.2 * sparsity * p)])[0],
	# 		n_iter=args.get("n_iter_finemap", [10000])[0],
	# 		n_config=args.get("n_config_finemap", [50000])[0],
	# 		sss_tol=args.get("sss_tol", [0.001])[0],
	# 		max_pep=max_pep,
	# 		max_size=max_size,
	# 		prenarrow=prenarrow,
	# 		corr_config=args.get("corr_config", [0.95])[0],
	# 		finemap_chains=finemap_chains,
	# 	)
	# 	# For fairness same max size (also allows disjointness)
	# 	rej_finemap = [x for x in rej_finemap if len(x) <= max_size]
	# 	fmap_time = time.time() - t0
	# 	t0 = time.time()
	# 	detections = pyblip.blip.BLiP(
	# 		cand_groups=cand_groups,
	# 		q=q,
	# 		error='fdr',
	# 		max_pep=max_pep,
	# 		perturb=True,
	# 		deterministic=True
	# 	)
	# 	blip_time = time.time() - t0
	# 	detect_sets = [
	# 		list(cg.group) for cg in detections
	# 	]
	# 	for method, csets, btime in zip(
	# 		['FINEMAP', 'FINEMAP + BLiP'],
	# 		[rej_finemap, detect_sets],
	# 		[0, blip_time],
	# 	):
	# 		metrics = power_fdr_metrics(csets, beta=beta, hatSig=hatSig, how='rejset')
	# 		output.append(
	# 			[method, "all", fmap_time, btime] + [True, 0] + dgp_args + metrics
	# 		)

	# # F-tests + FBH/Yekutieli
	# if kappa > 1 and args.get('run_ftests', [False])[0]:
	# 	t0 = time.time()
	# 	regtree = blip_sims.tree_methods.RegressionTree(
	# 		X=X, y=y, levels=levels, max_size=max_size
	# 	)
	# 	regtree.fit(family='gaussian') # this works better than using family = binomial 
	# 	# even when y follows a binomial distribution
	# 	mtime = time.time() - t0
	# 	_, rej_yek = regtree.ptree.outer_nodes_yekutieli(q=q)
	# 	rej_fbh, _ = regtree.ptree.tree_fbh(q=q)
	# 	for mname, rej in zip(['FBH', 'Yekutieli'], [rej_fbh, rej_yek]):
	# 		metrics = power_fdr_metrics(rej, beta=beta, hatSig=hatSig, how='cgs')
	# 		output.append(
	# 			[mname, "fbh", mtime, 0] + [True, 0] + dgp_args + metrics
	# 		)

	# CRT + FBH/Yekutieli
	if args.get('run_crt', [True])[0]:
		t0 = time.time()
		screen = args.get('screen', [True])[0]
		# Run CRT
		crt_model = blip_sims.crt.MultipleDCRT(y=y, X=X, Sigma=V, screen=screen)
		crt_model.multiple_pvals(levels=levels, max_size=max_size, verbose=False)
		mtime = time.time() - t0
		_, rej_yek = crt_model.pTree.outer_nodes_yekutieli(q=q)
		rej_fbh, _ = crt_model.pTree.tree_fbh(q=q)
		rej_bh, _ = crt_model.pTree.leaf_bh(q=q)
		for mname, rej in zip(
			['CRT + FBH', 'CRT + Yekutieli', 'CRT + BH'],
			[rej_fbh, rej_yek, rej_bh]
		):
			metrics = power_fdr_metrics(rej, beta=beta, hatSig=hatSig, how='cgs')
			output.append(
				[mname, "tree", mtime, 0] + [True, 0] + dgp_args + metrics
			)
		print(f"At seed={seed}, dgp_args={dgp_args}, finished CRT at time {utilities.elapsed(global_t0)}.")
		sys.stdout.flush()

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
		if (y_dist == 'gaussian' or not well_specified) and args.get('run_lss', [True])[0]:
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
		if y_dist != 'gaussian' and args.get('run_pss', [True])[0]:
			models.append(pyblip.probit.ProbitSpikeSlab(
				X=X,
				y=y,
				p0=p0,
				min_p0=min_p0,
				update_p0=update,
				tau2=tau2,
				update_tau2=update,
				sigma2=sigma2,
				update_sigma2=update,
			))
			method_names.append('PSS + BLiP')

		bsize = args.get('bsize', [1])[0]
		chains = args.get('chains', [10])[0]
		for nsample in args.get("nsample", [1000]):
			for model, mname in zip(models, method_names):
				t0 = time.time()
				skwargs = dict(N=nsample, chains=chains, bsize=bsize, burn=int(0.1*nsample))
				model.sample(**skwargs)
				inclusions = model.betas != 0
				mtime = time.time() - t0

				### Baseline (only indiv discoveries)
				peps = 1 - inclusions.mean(axis=0)
				fdrs = np.cumsum(np.sort(peps)) / np.arange(1, p+1)
				if np.any(fdrs <= q):
					thresh = np.where(fdrs <= q)[0].max() + 1
					rej = np.argsort(peps)[0:thresh]
				else:
					rej = []
				rej = [[r] for r in rej]
				metrics = power_fdr_metrics(rej, beta=beta, hatSig=hatSig, how='rejset')
				basename = mname.split("+")[0] + "(indiv only)"
				output.append(
					[basename, 'none', mtime, 0] + [well_specified, nsample] + dgp_args + metrics
				)

				# Calculate PIPs and cand groups
				for cgroup in args.get('cgroups', ['all']):
					t0 = time.time()
					if cgroup == 'all':
						cand_groups = pyblip.create_groups.all_cand_groups(
							samples=inclusions,
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
					elif cgroup == 'fbh':
						if not args.get('run_crt', [True])[0]:
							continue
						else:
							groups = [x.group for x in crt_model.pTree.nodes]
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
					metrics = power_fdr_metrics(detections, beta=beta, hatSig=hatSig, how='cgs')
					output.append(
						[mname, cgroup, mtime, blip_time] + [well_specified, nsample] + dgp_args + metrics
					)
					print(f"At seed={seed}, dgp_args={dgp_args}, finished {mname} at time {utilities.elapsed(global_t0)}.")
					sys.stdout.flush()

	# Method Type 2: susie-based methods
	if args.get('run_susie', [True])[0]:
		t0 = time.time()
		susie_alphas, susie_sets = blip_sims.susie.run_susie(
			X, y, L=np.ceil(p*sparsity), q=q
		)
		susie_time = time.time() - t0
		metrics = power_fdr_metrics(susie_sets, beta=beta, hatSig=hatSig, how='rejset')
		output.append(
			['susie', 'susie', susie_time, 0] + [True, 0] + dgp_args + metrics
		)
		# Now apply BLiP on top of susie
		t0 = time.time()
		cand_groups = pyblip.create_groups.susie_groups(
			alphas=susie_alphas, 
			X=X, 
			q=q,
			prenarrow=prenarrow,
			max_size=max_size,
			max_pep=max_pep
		)
		detections = pyblip.blip.BLiP(
			cand_groups=cand_groups,
			q=q,
			error='fdr',
			max_pep=max_pep,
			perturb=True,
			deterministic=True
		)
		blip_time = time.time() - t0
		metrics = power_fdr_metrics(detections, beta=beta, hatSig=hatSig, how='cgs')
		output.append(
			['susie + BLiP', "susie", susie_time, blip_time] + [True, 0] + dgp_args + metrics
		)
		print(f"At seed={seed}, dgp_args={dgp_args}, finished SuSiE at time {utilities.elapsed(global_t0)}.")
		sys.stdout.flush()
		
	return output

def main(args):
	# Parse arguments
	args = parser.parse_args(args)
	print(args['description'])
	sys.stdout.flush()
	reps = args.get('reps', [1])[0]
	num_processes = args.get('num_processes', [1])[0]
	# parse job id and seed start
	seed_start = max(args.get('seed_start', [1])[0], 1)
	job_id = int(args.pop("job_id", [0])[0])

	# Save args, create output dir
	output_dir, today, hour = utilities.create_output_directory(
		args, dir_type=DIR_TYPE, return_date=True
	)
	result_path = output_dir + f"/results_id{job_id}_seedstart{seed_start}.csv"

	# Run outputs
	time0 = time.time()
	all_outputs = []
	for covmethod in args.get('covmethod', ['ark']):
		for p in args.get('p', [500]):
			for kappa in args.get('kappa',[0.2]):
				for sparsity in args.get('sparsity', [0.05]):
					for k in args.get('k', [1]):
						for y_dist in args.get('y_dist', ['gaussian']):
							for coeff_size in args.get('coeff_size', [1]):
								constant_inputs=dict(
									y_dist=y_dist,
									covmethod=covmethod,
									kappa=kappa,
									p=p,
									sparsity=sparsity,
									k=k,
									coeff_size=coeff_size,
									global_t0=time0,
								)
								msg = f"Finished with {constant_inputs}"
								dap_prefix = blip_sims.utilities.create_dap_prefix(
									today=today,
									hour=hour,
									**constant_inputs,
								)
								constant_inputs['args'] = args
								constant_inputs['dap_prefix'] = dap_prefix
								outputs = utilities.apply_pool(
									func=single_seed_sim,
									seed=list(range(seed_start, reps+seed_start)), 
									constant_inputs=constant_inputs,
									num_processes=num_processes, 
								)
								msg += f" at {np.around(time.time() - time0, 2)}"
								print(msg)
								os.rmdir(os.path.dirname(dap_prefix))
								for out in outputs:
									all_outputs.extend(out)

								# Save
								out_df = pd.DataFrame(all_outputs, columns=COLUMNS)
								out_df.to_csv(result_path, index=False)
								groupers = [
									'method', 'cgroups', 'y_dist', 'covmethod', 'kappa', 
									'p', 'sparsity', 'k', 'well_specified', 'nsample'
								]
								meas = ['model_time', 'blip_time', 'power', 'ntd', 'fdr', 'med_group_size', 'med_purity', 'n_indiv_disc']
								summary_df = out_df.groupby(groupers)[meas].mean()
								print(summary_df.reset_index())



if __name__ == '__main__':
	main(sys.argv)