# example usage
sbatch --export=ALL,job_name=my_run,data=/path/to/dataset.npz,name=my_test,ntrain=8000,nvalid=2000,nepochs=50000 ./submit-job.sh

