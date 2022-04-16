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

def create_finemap_data(X, y, pi1, max_nsignal, file_prefix):
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
	zdata['maf'] = np.zeros(p) + 0.5
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
	probs = stats.binom(p, pi1).pmf(np.arange(max_nsignal))
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
	n_iter=100000,
	n_config=50000,
	**kwargs, # kwargs for all_cand_groups
):
	# Create data
	n, p = X.shape
	create_finemap_data(
		X=X,
		y=y,
		pi1=pi1,
		file_prefix=file_prefix,
		max_nsignal=max_nsignal,
	)
	# Construct command and run 
	# First, locate finemap
	finemap_executable = "finemap/finemap_v1.4.1_x86_64"
	for i in range(3):
		if os.path.exists(finemap_executable):
			break
		else:
			finemap_executable = "../" + finemap_executable
		if i == 2:
			raise ValueError("Could not find FINEMAP source")

	# Example: dap-g -d_z sim.1.zval.dat -d_ld sim.1.LD.dat -t 4 -o output.zval -l log.zval
	cmd = [finemap_executable]
	cmd.extend(["--sss"])
	cmd.extend(["--in-files", file_prefix])
	#cmd.extend(["--n-causal-snps", str(max_nsignal)])
	cmd.extend(["--n-iter", str(n_iter)])
	cmd.extend(["--n-configs-top", str(n_config)])
	cmd.extend(["--corr-config", str(0.99)])
	cmd.extend(["--prob-cred-set", str(1-q)])
	cmd.extend(["--prior-k"])
	cmd.extend(["--log"])
	process_out = subprocess.run(cmd)
	# Read log file (postprocessing due to FINEMAP format)
	logfile = f"{file_prefix}.log_sss"
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
	bf_files = glob.glob(f"{file_prefix}.bf*")
	nc_cands = [int(x.split(f".bf")[-1]) for x in bf_files]
	margcands = margprobs[nc_cands]
	selected_nc = margcands.index[margcands.argmax()]
	# Extra default credible sets
	credfile = f"{file_prefix}.cred{int(selected_nc)}"
	creddf = pd.read_csv(credfile, delimiter=' ', skiprows=5)
	default_credsets = []
	for col in creddf.columns:
		if 'cred' in col:
			flags = creddf[col].notnull()
			credset = creddf.loc[flags, col].copy()
			credset = credset.str.split("rs").apply(lambda x: int(x[-1]))
			default_credsets.append(set(credset))

	# Extract configuration output
	configfile = f"{file_prefix}.config"
	cand_groups = create_groups.finemap_groups(
			configfile=configfile,
			X=X,
			q=q,
			**kwargs
	)

	# Delete unnecessary files
	to_delete = glob.glob(f"{file_prefix}*")
	for fname in to_delete:
		os.remove(fname)
	return default_credsets, cand_groups 