import os
import shutil
import random
import glob

def split_data_by_patient(image_dir, mask_dir, output_dir, test_size=0.2, random_seed=42):
    """
    Splits dataset into train/test folders based on patient ID.
    
    Args:
        image_dir (str): Path to the folder containing image files.
        mask_dir (str): Path to the folder containing mask files.
        output_dir (str): Root folder to save split dataset.
        test_size (float): Proportion of patients to allocate to the test set.
        random_seed (int): Random seed for reproducibility.
    """

    # Extract patient IDs from image filenames
    image_files = os.listdir(image_dir)
    patient_ids = set(f.split("-")[0] for f in image_files)  # Extract patient ID

    print("count of patients", len(patient_ids))
    
    # Split patient IDs into train and test sets
    random.seed(random_seed)
    patient_ids = list(patient_ids)
    random.shuffle(patient_ids)
    split_index = int(len(patient_ids) * (1 - test_size))

    train_patients = set(patient_ids[:split_index])
    test_patients = set(patient_ids[split_index:])

    # Define train/test output directories
    train_img_out = os.path.join(output_dir, "train/images")
    train_mask_out = os.path.join(output_dir, "train/masks")
    test_img_out = os.path.join(output_dir, "test/images")
    test_mask_out = os.path.join(output_dir, "test/masks")

    os.makedirs(train_img_out, exist_ok=True)
    os.makedirs(train_mask_out, exist_ok=True)
    os.makedirs(test_img_out, exist_ok=True)
    os.makedirs(test_mask_out, exist_ok=True)

    # Move files to train/test folders
    for filename in image_files:
        patient_id = filename.split("-")[0]
        slice_id = filename.split("_")[0]  # Extract slice ID for matching masks
        src_img = os.path.join(image_dir, filename)

        # Get all corresponding masks (ignoring label postfix)
        mask_pattern = os.path.join(mask_dir, f"{slice_id}_MRI_prostate_*.png")
        mask_files = glob.glob(mask_pattern)

        if patient_id in train_patients:
            shutil.move(src_img, os.path.join(train_img_out, filename))
            for mask in mask_files:
                shutil.move(mask, os.path.join(train_mask_out, os.path.basename(mask)))
        else:
            shutil.move(src_img, os.path.join(test_img_out, filename))
            for mask in mask_files:
                shutil.move(mask, os.path.join(test_mask_out, os.path.basename(mask)))

    print(f"Dataset split complete! {len(train_patients)} patients in train, {len(test_patients)} in test.")

# Example usage:
split_data_by_patient("pngs/train", "pngs/train_mask", "dataset_split", test_size=0.2)
