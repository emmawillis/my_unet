import submitit

# from utils.analyze_dataset import compute_mean_std
from train import train

executor = submitit.AutoExecutor(folder="logs", slurm_max_num_timeout=10)
executor.update_parameters(
    slurm_gres='gpu:a40:1', 
    cpus_per_task=16,
    slurm_time=120,  # Increase time limit to 2 hours (in minutes)
    stderr_to_stdout=True,
    slurm_name="emma test",
    slurm_setup=[
        "module load anaconda/3.9",
        "module load pytorch2.1-cuda11.8-python3.9",
        "source activate my_unet_env"
    ])

job = executor.submit(train)
print(job.job_id)

output = job.result()  # waits for completion and returns output
print("done. output: ", output)