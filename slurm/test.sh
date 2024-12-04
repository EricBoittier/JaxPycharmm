sbatch --export=ALL,job_name=my_run,data=/pchem-data/meuwly/boittier/home/dataset_aaa.npz,name=aaa,natoms=34,ntrain=10000,nvalid=2500,nepochs=50000 ./submit-job.sh
