import submitit
import torch

from utils.analyze_dataset import compute_mean_std
from train import train

device1 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device outside submitit: ", device1)

executor = submitit.AutoExecutor(folder="logs", slurm_max_num_timeout=10)
executor.update_parameters(
    slurm_gres='gpu:a40:1', 
    cpus_per_task=16,
    stderr_to_stdout=True,
    slurm_name="test"
)

job = executor.submit(compute_mean_std)  # will compute add(5, 7)
print(job.job_id)  # ID of your job

output = job.result()  # waits for completion and returns output
print("done. output: ", output)