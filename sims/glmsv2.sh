#!/bin/sh
#SBATCH -p janson #Partition to submit to
#SBATCH -N 1 #Number of nodes
#SBATCH -n 32 #Number of cores
#SBATCH -t 7-00:00 #Runtime in (D-HH:MM)
#SBATCH --mem-per-cpu=4000 #Memory per cpu in MB (see also --mem)
#SBATCH -o ../output/glmv2_test_%j.out #File to which standard out will be written
#SBATCH -e ../output/glmv2_test_%j.err #File to which standard err will be written
#SBATCH --mail-type=ALL #Type of email notification- BEGIN,END,FAIL,ALL

NREPS=256
NPROCESSES=32

TEST_ARGS="
	--reps $NREPS
	--num_processes $NPROCESSES
	--p 300
	--kappa [0.1,0.3,0.5,0.7,0.9]
	--y_dist gaussian
	--nsample [2000]
	--well_specified [True]
	--max_corr 0.9999
	--sparsity [0.05]
	--run_dap [0]
	--run_susie [1]
	--max_size 1
	--k 5
"

VP_ARGS="
        --reps $NREPS
        --num_processes $NPROCESSES
        --p [500,1000,1500,2000,3000,5000,7500,10000]
        --coeff_size 1
        --kappa [0.5]
        --method [ar1]
        --y_dist [gaussian]
	--nsample [1000,10000]
	--well_specified [True]
        --max_corr 0.9999
        --k [2]
        --sparsity [0.05]
	--run_dap [0]
"

# All kappa: start0.1end1numvals9 and then [1.125, 1.25, 1.375,1.5,1.625,1.75,1.875,2]
# All sparsity: [0.01,0.05,0.1,0.2]
# y_dist [gaussian,probit]
# run dap only for sparsity == 0.01

HIGH_ARGS="
        --reps $NREPS
        --num_processes $NPROCESSES
        --p 1000
	--coeff_size 0.5
        --kappa start0.1end1numvals9 
        --method [ar1]
        --y_dist [gaussian]
	--max_corr 0.9999
	--k [5]
	--sparsity [0.01]
	--run_dap [1]
	--nsample 5000
"

# Bin low-dim setting
BIN_ARGS="
	--reps $NREPS
	--num_processes $NPROCESSES
	--p 400
	--coeff_size [1]
	--kappa [0.5,2,3.5,5]
	--method ar1
	--y_dist [probit]
	--well_specified [True,False]
	--k [1]
	--max_corr 0.9999
	--sparsity [0.01]
	--run_dap [1]
	--nsample 3000
"

# Bin high-dim setting
BIN_HD_ARGS="
        --reps $NREPS
        --num_processes $NPROCESSES
        --p 400
        --coeff_size [7.5,10]
        --kappa [0.5]
        --method ar1
        --y_dist [probit]
        --well_specified [True,False]
        --k [1]
        --sparsity [0.01,0.03,0.05]
        --run_dap [1]
        --nsample 2500
"



LOW_ARGS="
	--reps $NREPS
	--num_processes $NPROCESSES
	--p 400
	--coeff_size 0.01
	--kappa start2end20numvals10
	--method [ar1]
	--y_dist [gaussian,binomial]
	--max_corr 0.9999
	--k [1,3,5]
	--sparsity [0.01,0.03,0.05,0.1]
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

python3.9 glmsv2.py $TEST_ARGS
module purge



