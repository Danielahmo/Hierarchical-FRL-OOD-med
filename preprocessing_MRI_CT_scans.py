

import os
import nibabel as nib
import numpy as np
import torch
import torchio as tio
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def save_numpy_array(data, file_path):
    """
    A simple helper function to save a numpy array.
    This runs in a separate CPU process.
    """
    try:
        np.save(file_path, data)
        return True, file_path
    except Exception as e:
        return False, f"Failed to save {file_path}: {e}"

def save_mri_volumes_gpu(input_dir, output_dir):
    """
    Main function to orchestrate the GPU-accelerated processing.
    GPU processing is sequential, while file saving is parallel.
    """
    # 1. Set up GPU device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("Warning: CUDA not available. Running on CPU, which will be slow.")
    else:
        print(f"Using device: {device}")

    # 2. Get list of files
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    file_names = [f for f in os.listdir(input_dir) if f.lower().endswith(('.nii', '.nii.gz'))]
    if not file_names:
        print("No NIfTI files found in the input directory.")
        return

    # 3. Define the Torchio transform
    transform = tio.Compose([
        tio.Resize((64, 64, 64)),
    ])
    
    print("Starting processing...")
    
    # Use a ProcessPoolExecutor to save files in parallel (CPU-bound task)
    num_cpu_workers = os.cpu_count() or 4
    with ProcessPoolExecutor(max_workers=num_cpu_workers) as executor:
        
        # This list will hold the Future objects for saving tasks
        saving_futures = []
        
        # This loop runs sequentially on the main process with GPU
        for fname in tqdm(file_names, desc="Processing MRI volumes on GPU"):
            try:
                # Load the NIfTI file as a Torchio Subject
                subject = tio.Subject(mri=tio.ScalarImage(os.path.join(input_dir, fname)))
                
                # Apply transforms on the main process (on GPU if available)
                transformed_subject = transform(subject)

                # Get the processed data back to CPU and as a NumPy array
                vol = transformed_subject['mri'].data.cpu().numpy()
                
                # Squeeze the single channel dimension if present
                if vol.shape[0] == 1:
                    vol = np.squeeze(vol, axis=0)

                base_name = os.path.splitext(os.path.basename(fname))[0]
                mri_filename = f"{base_name}.npy"
                mri_path = os.path.join(output_dir, mri_filename)
                
                # Submit the saving task to the CPU worker pool
                saving_future = executor.submit(save_numpy_array, vol, mri_path)
                saving_futures.append(saving_future)
                
            except Exception as e:
                print(f"Error processing {fname}: {e}")
                
        # Wait for all saving tasks to complete and report progress
        for future in tqdm(as_completed(saving_futures), total=len(saving_futures), desc="Saving files to disk"):
            success, result = future.result()
            if not success:
                print(result)
                
    print("Processing and saving complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    opt = parser.parse_args()

    input_dir = opt.input_dir
    output_dir = opt.output_dir
    save_mri_volumes_gpu(input_dir, output_dir)