import os
import sys
import numpy as np
import pandas as pd
import subprocess

def create_dap_data(X, y, file_prefix):
	n, p = X.shape
	ld = np.corrcoef(X.T)
	# Note E[X] = 0 
	correst = np.dot(X.T, y) / n
	ses = np.std(X.T * y, axis=1) / np.sqrt(n)
	zvals = correst / ses
	# for output
	data = pd.DataFrame()
	data['snp_name_i'] = [f"rs{x}" for x in range(p)]
	data['z_i'] = zvals
	data.to_csv(
		file_prefix+'zval.dat',
		sep=' ',
		index=False,
		header=False
	)
	np.savetxt(
		fname=file_prefix+"LD.dat",
		X=ld,
		delimiter=' ',
	)
	with open(file_prefix + "N_syy.dat", 'w') as file:
		file.write(f"N = {n} \n")
		file.write(f"Syy = {np.power(y, 2).sum()} \n")
	return 0

def run_dap(X, y, file_prefix, q, pi1=None, threads=1):
	# Create data
	create_dap_data(X=X, y=y, file_prefix=file_prefix)
	# Construct command and run 
	# Example: dap-g -d_z sim.1.zval.dat -d_ld sim.1.LD.dat -t 4 -o output.zval -l log.zval
	outfile = file_prefix + "output.zval"
	logfile = file_prefix + "log.zval"
	cmd = [f"../dap/dap_src/dap-g"]
	cmd.extend(["-d_z", f"{file_prefix}zval.dat"])
	cmd.extend(["-d_ld", f"{file_prefix}LD.dat"])
	cmd.extend(["-d_n", str(X.shape[0])])
	cmd.extend(["-d_syy", str(np.power(y, 2).sum())])
	cmd.extend(["-t", str(threads)])
	cmd.extend(["-o", outfile])
	cmd.extend(["-l", logfile])
	if pi1 is not None:
		cmd.extend(["-pi1", str(pi1)])
	process_out = subprocess.run(cmd)
	# Read file (postprocessing due to dap-G format)
	indiv_pips = []
	cluster_pips = dict()
	models = []
	with open(outfile, 'r') as thefile:
		for i, line in enumerate(thefile):
			if "((" in line:
				splitline = line.split()
				feature = int(splitline[1].split("rs")[-1])
				pip = float(splitline[2])
				cluster = int(splitline[4])
				indiv_pips.append([feature, pip, cluster])
			if "[" in line:
				splitline = line.split()
				pip = float(splitline[1])
				if 'NULL' in line:
					models.append(dict(pip=pip, features=[]))
					continue
				features = [x for x in splitline if x[0] == '[']
				features = [x.split('rs')[-1].split(']')[0] for x in features]
				try:
					features = [int(x) for x in features]
				except Exception as error:
					print(line)
					print(features)
					raise error
				models.append(dict(pip=pip, features=features))
			if "{" in line:
				splitline = line.split()
				cid = int(splitline[0].split("{")[-1].split("}")[0])
				pip = float(splitline[2])
				cluster_pips[cid] = pip

	indiv_pips = pd.DataFrame(
		indiv_pips, columns=['feature', 'pip', 'cluster']
	)
	cluster_ids = indiv_pips['cluster'].unique()
	rej_clusters = [

	]
	for cid in cluster_ids:
		if cid == -1:
			continue
		if cluster_pips[cid] >= 1 - q:
			rej_clusters.append(
				indiv_pips.loc[indiv_pips['cluster'] == cid, 'feature'].tolist()
			)
	return rej_clusters, indiv_pips, models