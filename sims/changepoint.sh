#!/bin/sh
#SBATCH -p janson #Partition to submit to
#SBATCH -N 1 #Number of nodes
#SBATCH -n 32 #Number of cores
#SBATCH -t 7-00:00 #Runtime in (D-HH:MM)
#SBATCH --mem-per-cpu=4000 #Memory per cpu in MB (see also --mem)
#SBATCH -o ../output/cp2_%j.out #File to which standard out will be written
#SBATCH -e ../output/cp2_%j.err #File to which standard err will be written
#SBATCH --mail-type=ALL #Type of email notification- BEGIN,END,FAIL,ALL

NREPS=256
NPROCESSES=32

CP_ARGS="
        --reps $NREPS
        --num_processes $NPROCESSES
        --p [200]
        --coeff_size [1,5,9,13,17]
	--nsample [20000]
	--well_specified [False]
        --sparsity [0.01,0.05,0.1]
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

python3.9 changepoint.py $CP_ARGS

module purge



