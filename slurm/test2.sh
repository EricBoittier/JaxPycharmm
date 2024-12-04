sbatch --export=ALL,job_name=my_run,data="/pchem-data/meuwly/boittier/home/jaxeq/notebooks/ala-esp-dip-0.npz",name=adp,natoms=37,ntrain=5000,nvalid=2000,nepochs=50000 ./submit-job.sh
