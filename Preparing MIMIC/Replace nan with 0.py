import os
import csv
import tarfile
import tempfile
import shutil
import multiprocessing
from multiprocessing import Pool, cpu_count

# --- 1. EXTRACTION FUNCTION ---
def extract_dataset(tar_path, extract_to_dir):
    print(f"Starting extraction of {tar_path}...")
    print("This may take a significant amount of time for 800,000 files.")
    
    if not os.path.exists(extract_to_dir):
        os.makedirs(extract_to_dir, exist_ok=True)
        
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_to_dir)
        
    print("Extraction complete!\n")

# --- 2. WORKER FUNCTION (LOG & FILL) ---
def process_and_fill_file(file_path):
    try:
        file_name = os.path.basename(file_path)

        min_empty = float('inf')
        max_empty = -1
        has_empty_cells = False
        row_count = 0

        # Create a temporary file in the SAME directory
        file_dir = os.path.dirname(file_path)
        temp_fd, temp_path = tempfile.mkstemp(dir=file_dir, text=True)

        with open(file_path, 'r', newline='', encoding='utf-8') as csv_in, \
             os.fdopen(temp_fd, 'w', newline='', encoding='utf-8') as csv_out:
            
            reader = csv.reader(csv_in)
            writer = csv.writer(csv_out)

            for row in reader:
                if not row: continue
                row_count += 1

                # Count empty strings in this specific row
                empty_in_row = row.count("")

                if empty_in_row > 0:
                    has_empty_cells = True
                    # FILLING STEP: Replace all "" with "0"
                    row = ["0" if cell == "" else cell for cell in row]

                # Update Min/Max tracking
                if empty_in_row < min_empty:
                    min_empty = empty_in_row
                if empty_in_row > max_empty:
                    max_empty = empty_in_row

                # Write the (potentially corrected) row to the temp file
                writer.writerow(row)

        # --- FORMAT RESULTS & CLEANUP ---
        if row_count == 0:
            os.remove(temp_path) 
            return (file_name, "0; 0", "NO")

        if min_empty == float('inf'): min_empty = 0
        if max_empty == -1: max_empty = 0

        empty_range_str = f"{min_empty}; {max_empty}"
        has_empty_str = "YES" if has_empty_cells else "NO"

        # If we found and fixed empty cells, replace original with the fixed temp file
        if has_empty_cells:
            shutil.move(temp_path, file_path)
        else:
            # If no empty cells were found, just delete the temp file
            os.remove(temp_path) 

        return (file_name, empty_range_str, has_empty_str)

    except Exception:
        # Emergency cleanup of temp file if something crashes
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        return (os.path.basename(file_path), "ERROR", "ERROR")

# --- 3. MAIN CONTROLLER ---
def main():
    # --- EXACT PATHS PROVIDED ---
    tar_archive = r"/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/Data/MIMIC_IV_ECG_CSV_MICROVOLTS_v3.tar.gz"
    
    target_folder = r"/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/Data/Unzipped"
    
    log_dir = r"/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/Preparing MIMIC"
    log_output = os.path.join(log_dir, "ecg_filled_log.csv")

    # Ensure the log directory exists before trying to save a file there
    os.makedirs(log_dir, exist_ok=True)

    # STEP A: Unzip the files
    if os.path.exists(tar_archive):
        # We only extract if the target folder doesn't exist or is empty to prevent doing it twice
        if not os.path.exists(target_folder) or not os.listdir(target_folder):
            extract_dataset(tar_archive, target_folder)
        else:
            print(f"Files already seem to exist in {target_folder}. Skipping extraction.")
    else:
        print(f"CRITICAL ERROR: Archive still not found at {tar_archive}.")
        return

    # STEP B: Collect all file paths
    all_files = []
    print(f"Scanning directory: {target_folder}...")
    
    for root, _, files in os.walk(target_folder):
        for file in files:
            if file.lower().endswith('.csv'):
                all_files.append(os.path.join(root, file))

    total_files = len(all_files)
    if total_files == 0:
        print("No CSV files found after extraction. Exiting.")
        return

    print(f"Found {total_files} files. Starting parallel processing on {cpu_count()} cores...")

    # STEP C: Process, Fill, and Log
    with open(log_output, 'w', newline='', encoding='utf-8') as log_out:
        writer = csv.writer(log_out)
        writer.writerow(['File Name', 'Empty Cells (Min; Max)', 'Has Empty Cells (And Was Filled)'])

        with Pool(processes=cpu_count()) as pool:
            results = pool.imap_unordered(process_and_fill_file, all_files, chunksize=50)

            count = 0
            for result in results:
                writer.writerow(result)
                count += 1
                
                # Progress update every 100 files
                if count % 100 == 0 or count == total_files:
                    print(f"Processed and cleaned {count}/{total_files} files...", end='\r')

    print(f"\nProcessing complete. Log saved to: {log_output}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()