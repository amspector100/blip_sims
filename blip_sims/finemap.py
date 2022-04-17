import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy import stats
import subprocess
from functools import reduce
import time
from pyblip import create_groups

def create_finemap_data(X, y, pi1, max_nsignal, file_prefix, maf=0.5):
	n, p = X.shape
	ld = np.corrcoef(X.T)
	# Note E[X] = 0 
	correst = np.dot(X.T, y) / n
	ses = np.std(X.T * y, axis=1) / np.sqrt(n)
	zvals = correst / ses
	# Iteratively create master file
	master_header = "z;ld;k;snp;config;cred;log;n_samples"
	master_line = ""
	# 1. Create z file
	zdata = pd.DataFrame()
	zdata['rsid'] = [f"rs{x}" for x in range(p)]
	zdata['chromosome'] = 1
	zdata['position'] = list(range(1, p+1))
	zdata['allele1'] = 'A'
	zdata['allele2'] = 'G'
	zdata['maf'] = np.zeros(p) + maf
	zdata['beta'] = correst
	zdata['se'] = ses
	zfname = f"{file_prefix}.z"
	zdata.to_csv(
		zfname,
		sep=' ',
		index=False,
		header=True
	)
	master_line += zfname + ";"
	# 2. LD file
	ldfname = file_prefix + ".ld"
	np.savetxt(
		fname=ldfname,
		X=ld,
		delimiter=' ',
	)
	master_line += ldfname + ";"
	# 3. k file
	kfname = f"{file_prefix}.k"
	probs = stats.binom(p, pi1).pmf(np.arange(1, max_nsignal+1))
	probs = probs / probs.sum()
	np.savetxt(
		fname=kfname,
		X=probs,
		delimiter=' ',
		newline=' ',
	)
	master_line += kfname + ";"
	# 4. Config/SNP/log files and nsample. 
	# BCOR file is apparently not needed?
	master_line += f"{file_prefix}.snp;"
	master_line += f"{file_prefix}.config;"
	master_line += f"{file_prefix}.cred;"
	master_line += f"{file_prefix}.log;"
	master_line += str(n)
	# 5. Write master file
	with open(f"{file_prefix}", 'w') as thefile:
		thefile.write(master_header)
		thefile.write("\n")
		thefile.write(master_line)
	return 0

def run_finemap(
	X, 
	y, 
	file_prefix, 
	q, 
	pi1, 
	max_nsignal,
	finemap_chains=10,
	corr_config=0.95,
	n_iter=100000,
	n_config=50000,
	sss_tol=0.001,
	remove_data=True,
	maf=0.5,
	**kwargs, # kwargs for all_cand_groups
):
	# Create data
	n, p = X.shape
	# locate finemap
	finemap_executable = "finemap/finemap_v1.4.1_x86_64"
	for i in range(3):
		if os.path.exists(finemap_executable):
			break
		else:
			finemap_executable = "../" + finemap_executable
		if i == 2:
			raise ValueError("Could not find FINEMAP source")

	configfiles = []
	probk0 = np.exp(p * np.log(1 - pi1))
	for chain in range(finemap_chains):
		file_prefix_chain = file_prefix + f"chain{chain}"
		create_finemap_data(
			X=X,
			y=y,
			pi1=pi1,
			file_prefix=file_prefix_chain,
			max_nsignal=max_nsignal,
			maf=maf,
		)
		# Example: dap-g -d_z sim.1.zval.dat -d_ld sim.1.LD.dat -t 4 -o output.zval -l log.zval
		cmd = [finemap_executable]
		cmd.extend(["--sss"])
		cmd.extend(["--in-files", file_prefix_chain])
		#cmd.extend(["--n-causal-snps", str(max_nsignal)])
		cmd.extend(["--n-iter", str(n_iter)])
		cmd.extend(["--n-configs-top", str(n_config)])
		cmd.extend(["--prob-conv-sss-tol", str(sss_tol)])
		cmd.extend(["--corr-config", str(corr_config)])
		cmd.extend(["--prob-cred-set", str(1-q)])
		cmd.extend(["--prior-k0", str(probk0)])
		cmd.extend(["--prior-k"])
		cmd.extend(["--log"])
		process_out = subprocess.run(cmd)
		# Read log file (postprocessing due to FINEMAP format)
		logfile = f"{file_prefix_chain}.log_sss"
		ncausals = []
		margprobs = []
		with open(logfile, 'r') as thefile:
			post_pr_flag = False
			for i, line in enumerate(thefile):
				if "Post-Pr(# of causal SNPs is k)" in line:
					post_pr_flag = True
				if post_pr_flag and "->" in line:
					line = line.replace(" ", '').replace("(", '')
					line = line.replace(")", '').split("->")
					ncausal = int(line[0])
					margprob = float(line[-1])
					ncausals.append(ncausal)
					margprobs.append(margprob)
		margprobs = pd.Series(
			index=ncausals, 
			data=margprobs
		)

		# Check which bf files are available and pick the one
		# with the highest marginal probability
		bf_files = glob.glob(f"{file_prefix_chain}.bf*")
		nc_cands = [int(x.split(f".bf")[-1]) for x in bf_files]
		margcands = margprobs[nc_cands]
		selected_nc = margcands.index[margcands.argmax()]
		# Extra default credible sets
		credfile = f"{file_prefix_chain}.cred{int(selected_nc)}"
		creddf = pd.read_csv(credfile, delimiter=' ', skiprows=5)
		default_credsets = []
		for col in creddf.columns:
			if 'cred' in col:
				flags = creddf[col].notnull()
				credset = creddf.loc[flags, col].copy()
				credset = credset.str.split("rs").apply(lambda x: int(x[-1]))
				default_credsets.append(set(credset))

		# Extract configuration output
		configfiles.append(f"{file_prefix_chain}.config")


	# Create candidate groups
	cand_groups = create_groups.finemap_groups(
			configfile=configfiles,
			X=X,
			q=q,
			**kwargs
	)

	# Delete unnecessary files
	if remove_data:
		for chain in range(finemap_chains):
			file_prefix_chain = file_prefix + f"chain{chain}"
			to_delete = glob.glob(f"{file_prefix_chain}.*")
			for fname in to_delete:
				os.remove(fname)
			os.remove(file_prefix_chain) # master file
	return default_credsets, cand_groups 