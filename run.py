import submitit
from dataloader import ProstateMRISegmentationDataset
import unet

def add(a, b):
    u = unet.UNetEncoder(3)
    x = ProstateMRISegmentationDataset('dataset_split/train/images', 'dataset_split/train/masks')
    return a + b

executor = submitit.AutoExecutor(folder="logs", slurm_max_num_timeout=10)
executor.update_parameters(
    slurm_gres='gpu:a40:1', 
    cpus_per_task=16,
    stderr_to_stdout=True,
    slurm_name="test"
)

job = executor.submit(add, 5, 7)  # will compute add(5, 7)
print(job.job_id)  # ID of your job

output = job.result()  # waits for completion and returns output
print("done - got", output)