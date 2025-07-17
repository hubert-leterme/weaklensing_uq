
import h5py
import numpy as np
import torch
import torchvision.transforms as T
from torch.nn.functional import unfold
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt

def main():
    # # Prepare CosmoSLICS Dataset: Train/Test Split and Augmentation
    #
    # This script performs the following steps:
    # 1. Loads the original CosmoSLICS maps (500 maps, 20 realizations for 25 cosmologies).
    # 2. Extracts maps from the 2nd cosmology (maps 20-39) to create the first test set (`Test Set Cosmology 2`).
    # 3. From the remaining 480 maps, randomly selects 40 maps to create a second test set (`Test Set Random`).
    # 4. The remaining 440 maps form the training set.
    # 5. Saves the raw images for both test sets to new HDF5 files.
    # 6. Extracts patches from the training set images.
    # 7. Applies data augmentation to the training patches.
    # 8. Saves the augmented training patches to a new HDF5 file.

    # ## 1. Configuration Parameters
    INPUT_FILE_PATH = "/path/to/cosmoslics_maps_029arcmin.hdf5"
    TEST_COSMOLOGY_2_DATA_PATH = "/path/to/cosmoslics_maps_029arcmin_test_cosmology2.hdf5"
    TEST_RANDOM_REALIZATIONS_DATA_PATH = "/path/to/cosmoslics_maps_029arcmin_test_random.hdf5"
    AUGMENTED_TRAIN_DATA_PATH = "/path/to/cosmoslics_maps_029arcmin_train_small_augmented_384patch.hdf5"

    PATCH_SIZE = 384
    STRIDE = 192
    NUM_MAPS_FOR_RANDOM_TEST_SET = 40
    RANDOM_SEED = 42

    # Ensure output directories exist
    os.makedirs(os.path.dirname(TEST_COSMOLOGY_2_DATA_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(TEST_RANDOM_REALIZATIONS_DATA_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(AUGMENTED_TRAIN_DATA_PATH), exist_ok=True)

    # ## 2. Define Patch Extraction Function
    def extract_patches(image_tensor, patch_size, stride):
        """Extract patches using PyTorch's efficient unfold operation."""
        # Ensure image_tensor is 2D (H, W)
        if len(image_tensor.shape) != 2:
            raise ValueError(f"Input image_tensor must be 2D (H, W), got shape {image_tensor.shape}")
        
        # Add batch and channel dimensions for unfold: (1, 1, H, W)
        image_tensor_unsqueezed = image_tensor.unsqueeze(0).unsqueeze(0)
        
        # Use unfold to extract patches
        patches = unfold(image_tensor_unsqueezed, kernel_size=patch_size, stride=stride)
        
        # Reshape to get individual patches: (num_patches, patch_size, patch_size)
        # patches will have shape (1, patch_size*patch_size, num_patches_in_image)
        # We want (num_patches_in_image, patch_size, patch_size)
        patches = patches.squeeze(0).permute(1, 0).reshape(-1, patch_size, patch_size)
        return patches

    # ## 3. Load Original Data
    try:
        with h5py.File(INPUT_FILE_PATH, 'r') as f:
            # Assuming the dataset key is 'maps' as in your previous notebook
            # Or adjust if it's different (e.g., 'kappa')
            dataset_key = 'maps' # Default key, try to find if not present
            if dataset_key not in f:
                if 'kappa' in f:
                    dataset_key = 'kappa'
                else:
                    available_keys = list(f.keys())
                    raise KeyError(f"'{dataset_key}' and 'kappa' not found in HDF5 file. Available keys: {available_keys}")
            all_maps = torch.tensor(f[dataset_key][:], dtype=torch.float32)
        print(f"Loaded {all_maps.shape[0]} maps, each of shape {all_maps.shape[1:]}")
    except Exception as e:
        print(f"Error loading data from {INPUT_FILE_PATH}: {e}")
        # You might want to stop execution here or handle it appropriately
        all_maps = None

    # ## 4. Split Data into Training and Test Sets
    #
    # The splitting strategy is as follows:
    # 1.  **Test Set 1 (Cosmology 2):** Maps with indices 20-39 (20 maps). These correspond to the 2nd cosmology.
    # 2.  **Remaining Pool for Training/Test Set 2:** Maps with indices 0-19 and 40-499 (480 maps).
    # 3.  **Test Set 2 (Random Realizations):** Randomly select `NUM_MAPS_FOR_RANDOM_TEST_SET` (e.g., 40) maps from the 'Remaining Pool'.
    # 4.  **Training Set:** The maps left in the 'Remaining Pool' after Test Set 2 is extracted (e.g., 480 - 40 = 440 maps).
    if all_maps is not None:
        num_total_maps = all_maps.shape[0]
        all_indices = np.arange(num_total_maps)

        # 1. Test Set 1 (Cosmology 2): Indices 20 to 39
        cosmology_2_test_indices = np.arange(20, 40)
        test_maps_cosmology_2 = all_maps[cosmology_2_test_indices]
        print(f"Test Set 1 (Cosmology 2): {test_maps_cosmology_2.shape[0]} maps from indices 20-39.")

        # 2. Remaining Pool for Training/Test Set 2
        remaining_pool_indices = np.setdiff1d(all_indices, cosmology_2_test_indices)
        print(f"Maps remaining after extracting Cosmology 2: {len(remaining_pool_indices)}")

        # 3. Test Set 2 (Random Realizations)
        if len(remaining_pool_indices) < NUM_MAPS_FOR_RANDOM_TEST_SET:
            raise ValueError(f"Not enough maps in the remaining pool ({len(remaining_pool_indices)}) to select {NUM_MAPS_FOR_RANDOM_TEST_SET} for Test Set 2.")
        
        np.random.seed(RANDOM_SEED) # for reproducibility of this random selection
        random_test_indices = np.random.choice(remaining_pool_indices, size=NUM_MAPS_FOR_RANDOM_TEST_SET, replace=False)
        test_maps_random = all_maps[random_test_indices]
        print(f"Test Set 2 (Random Realizations): {test_maps_random.shape[0]} maps randomly selected.")

        # 4. Training Set
        train_indices = np.setdiff1d(remaining_pool_indices, random_test_indices)
        train_maps = all_maps[train_indices]
        print(f"Training Set: {train_maps.shape[0]} maps.")

        # Sanity check
        total_selected = len(test_maps_cosmology_2) + len(test_maps_random) + len(train_maps)
        print(f"Total maps accounted for: {total_selected} (should be {num_total_maps})")
        assert total_selected == num_total_maps, "Mismatch in map counts after splitting!"
    else:
        print("Skipping train/test split due to data loading failure.")
        test_maps_cosmology_2, test_maps_random, train_maps = None, None, None

    # ## 5. Save Test Sets (Raw Images)
    if test_maps_cosmology_2 is not None:
        try:
            with h5py.File(TEST_COSMOLOGY_2_DATA_PATH, 'w') as f:
                f.create_dataset('maps', data=test_maps_cosmology_2.numpy()) # Or 'kappa'
            print(f"Saved {test_maps_cosmology_2.shape[0]} raw test images (Cosmology 2) to {TEST_COSMOLOGY_2_DATA_PATH}")
        except Exception as e:
            print(f"Error saving Test Set 1 (Cosmology 2) data to {TEST_COSMOLOGY_2_DATA_PATH}: {e}")
    else:
        print("Skipping saving Test Set 1 (Cosmology 2) due to data splitting failure.")

    if test_maps_random is not None:
        try:
            with h5py.File(TEST_RANDOM_REALIZATIONS_DATA_PATH, 'w') as f:
                f.create_dataset('maps', data=test_maps_random.numpy()) # Or 'kappa'
            print(f"Saved {test_maps_random.shape[0]} raw test images (Random Realizations) to {TEST_RANDOM_REALIZATIONS_DATA_PATH}")
        except Exception as e:
            print(f"Error saving Test Set 2 (Random Realizations) data to {TEST_RANDOM_REALIZATIONS_DATA_PATH}: {e}")
    else:
        print("Skipping saving Test Set 2 (Random Realizations) due to data splitting failure.")

    # The test images for both sets are saved in their original, un-augmented, and un-patched format. This ensures that model evaluation is based on data that reflects the true, original distribution expected at inference time, covering both a specific out-of-distribution cosmology and random samples from the training cosmologies.

    # ## 6. Define Augmentation Transforms for Training Data
    augmentation_transforms = [
        # Original
        T.Lambda(lambda x: x),
        # Rotations (ensure expand=False if patch size should be maintained)
        T.RandomRotation((90, 90), expand=False),
        T.RandomRotation((180, 180), expand=False),
        T.RandomRotation((270, 270), expand=False),
        # Flips
        T.RandomHorizontalFlip(p=1.0),
        T.RandomVerticalFlip(p=1.0),
        # Combined transforms
        T.Compose([T.RandomHorizontalFlip(p=1.0), T.RandomRotation((90, 90), expand=False)]),
        T.Compose([T.RandomHorizontalFlip(p=1.0), T.RandomRotation((180, 180), expand=False)]),
    ]
    print(f"Defined {len(augmentation_transforms)} augmentation transforms.")

    # ## 7. Process and Augment Training Data
    if train_maps is not None:
        all_augmented_patches = []
        torch.manual_seed(RANDOM_SEED) # For reproducibility of transforms if they have internal randomness
        np.random.seed(RANDOM_SEED)
        
        print(f"Starting augmentation for {train_maps.shape[0]} training maps...")
        for i, map_tensor in enumerate(train_maps):
            # map_tensor is expected to be 2D (H,W) by extract_patches
            patches_from_map = extract_patches(map_tensor, PATCH_SIZE, STRIDE)
            
            for patch_idx, patch in enumerate(patches_from_map):
                # patch is a 2D tensor (PATCH_SIZE, PATCH_SIZE)
                # Add a channel dimension for torchvision transforms: (1, PATCH_SIZE, PATCH_SIZE)
                patch_for_transform = patch.unsqueeze(0)
                for transform_idx, transform in enumerate(augmentation_transforms):
                    # Apply transform. Most torchvision transforms handle 2D (H,W) tensors.
                    augmented_patch = transform(patch_for_transform)
                    all_augmented_patches.append(augmented_patch.numpy())
            
            if (i + 1) % 10 == 0 or (i+1) == len(train_maps): # Print progress every 10 maps
                 print(f"Processed map {i+1}/{len(train_maps)}: extracted {patches_from_map.shape[0]} patches, applied {len(augmentation_transforms)} augmentations to each.")
        
        if all_augmented_patches:
            all_augmented_patches_np = np.array(all_augmented_patches, dtype=np.float32)
            np.random.shuffle(all_augmented_patches_np) # Shuffle all augmented patches together
            print(f"\nTotal augmented training patches: {all_augmented_patches_np.shape[0]}")
            print(f"Shape of augmented training dataset: {all_augmented_patches_np.shape}")
        else:
            print("\nNo augmented patches were generated. Check input data and patch extraction.")
            all_augmented_patches_np = np.array([]) # Empty array for saving
    else:
        print("Skipping augmentation due to data splitting or loading failure.")
        all_augmented_patches_np = np.array([])

    if all_augmented_patches_np.ndim == 4 and all_augmented_patches_np.shape[1] == 1:
        all_augmented_patches_np = all_augmented_patches_np.squeeze(axis=1)
        print(f"Squeezed augmented training dataset to shape: {all_augmented_patches_np.shape}")

    if 'all_augmented_patches_np' in locals() and all_augmented_patches_np.size > 0:
      print(all_augmented_patches_np.shape)
    else:
      print("Augmented training data not available or empty.")

    # ## 8. Save Augmented Training Set
    if all_augmented_patches_np.size > 0 :
        try:
            with h5py.File(AUGMENTED_TRAIN_DATA_PATH, 'w') as f:
                f.create_dataset('kappa', data=all_augmented_patches_np)
            print(f"Saved {all_augmented_patches_np.shape[0]} augmented training patches to {AUGMENTED_TRAIN_DATA_PATH}")
        except Exception as e:
            print(f"Error saving augmented training data to {AUGMENTED_TRAIN_DATA_PATH}: {e}")
    elif train_maps is not None: # Only print if data splitting was successful but no patches made
        print(f"No augmented training patches to save.")



if __name__ == '__main__':
    main()
