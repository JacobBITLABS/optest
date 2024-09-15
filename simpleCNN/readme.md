## Steps to run on Leonardo
1. Prepare the environment: Ensure the appropriate PyTorch module with GPU support is available on Leonardo (usually via module load pytorch). (Handled in .slurm)
2. Submit the job: Submit the job using sbatch:
```
sbatch test_cifar10_gpu_job.slurm
```
3. Monitor job status: You can monitor your job with:
```
squeue -u your_username
```
4. Check the output: When the job is done, the training logs will be saved in: `cifar10_test.out`
