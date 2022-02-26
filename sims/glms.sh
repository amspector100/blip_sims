#!/bin/sh
#SBATCH -p janson #Partition to submit to
#SBATCH -N 1 #Number of nodes
#SBATCH -n 32 #Number of cores
#SBATCH -t 7-00:00 #Runtime in (D-HH:MM)
#SBATCH --mem-per-cpu=4000 #Memory per cpu in MB (see also --mem)
#SBATCH -o ../output/glm_pk5hd_%j.out #File to which standard out will be written
#SBATCH -e ../output/glm_pk5hd_%j.err #File to which standard err will be written
#SBATCH --mail-type=ALL #Type of email notification- BEGIN,END,FAIL,ALL

NREPS=128
NPROCESSES=32

VP_ARGS="
        --reps $NREPS
        --num_processes $NPROCESSES
        --p [500,1000,1500,2000,3000,5000,7500,10000]
        --coeff_size 1
        --kappa [0.1]
        --method [ar1]
        --y_dist [gaussian]
	--nsample [1000,10000]
	--well_specified [True]
        --max_corr 0.9999
        --k [2]
        --sparsity [0.05]
	--run_crt [0]
	--run_dap [0]
"

# All kappa: [0.1,0.2,0.35,0.5,0.75,1,1.25,1.5,1.75,2.0]
# We do not run kappa > 1 for the sparser settings
# All sparsity: [0.01,0.05,0.1,0.2]
# y_dist [gaussian,probit]
# run dap only for sparsity == 0.01

HIGH_ARGS_COMMON="
        --reps $NREPS
        --num_processes $NPROCESSES
        --p 1000
	--coeff_size 0.5
        --kappa [1]
        --covmethod [ark]
        --y_dist [gaussian]
	--max_corr 0.99
	--k [5]
	--nsample 5000
	--chains 10
	--bsize 5
	--screen 0
"

HIGH_P1="${HIGH_ARGS_COMMON}
	--sparsity [0.01]
	--run_dap [0]	
	--run_crt [1]
	--levels 8
"

HIGH_P2="${HIGH_ARGS_COMMON}
	--sparsity [0.1,0.2]
	--run_dap [0]
	--run_crt [1]
	--levels 8
"

# Bin setting

# Original setting: coef_size [1] kappa [0.1,0.5,1.25,2,3.5,5,6.5]

# In originalsetting
# Nsample: 5000
# chains: 10

BIN_ARGS="
	--reps $NREPS
	--num_processes $NPROCESSES
	--p 400
	--coeff_size [1]
	--kappa [0.1,1.25]
	--method ar1
	--y_dist [probit]
	--well_specified [True,False]
	--k [5]
	--chains 10
	--max_corr 0.99
	--nsample 5000
	--sparsity [0.01]
	--screen [0]
	--run_crt [1]
	--run_dap [1]
	--levels 8
"

# Install python/anaconda, activate base environment
module purge
module load Gurobi/9.1.2
module load Anaconda3/5.0.1-fasrc02
module load GCC/8.2.0-2.31.1 GSL/2.5 # for dap
source activate adaptexper1

# Load R 4.0.2
module load R/4.0.2-fasrc01

# Load packages
export R_LIBS_USER=$HOME/apps/R_4.0.2:$R_LIBS_USERi

#python3.9 glms.py $HIGH_P1
python3.9 glms.py $BIN_ARGS
#python3.9 glms.py $VP_ARGS

module purge



