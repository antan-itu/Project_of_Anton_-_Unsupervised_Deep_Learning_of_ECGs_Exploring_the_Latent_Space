import os
import csv
from tqdm import tqdm

def delete_flagged_files():
    # --- CONFIGURATION ---
    # Path to the folder containing your ECG files
    # target_folder = r"C:\Users\anton\Downloads\MIMIC_IV_ECG_CSV_MICROVOLTS_v3\MIMIC_IV_ECG_CSV_MICROVOLTS_v3\files"
    files_directory = r"C:\Users\anton\Downloads\Test"

    # Path to the log file we created in the previous step
    log_file_path = "ecg_consistency_log.csv"

    # SAFETY: Set this to False to actually delete files
    # True = Simulate deletion
    # False = Execute deletion
    DRY_RUN = False

    print(f"Reading log file: {log_file_path}...")

    files_to_delete = []

    # 1. Read the log file and find files to delete
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Check if file is flagged with YES in "Has Empty Cells"
                if row['Has Empty Cells'].strip().upper() == 'YES':
                    files_to_delete.append(row['File Name'])
    except FileNotFoundError:
        print(f"Error: Could not find log file '{log_file_path}'. Please run the scan script first.")
        return

    count = len(files_to_delete)

    if count == 0:
        print("No files found to delete.")
        return

    print(f"Found {count} files flagged for deletion.")

    # 2. Confirmation before deletion (if not Dry Run)
    if not DRY_RUN:
        print(f"WARNING: You are about to permanently delete {count} files.")
        confirm = input(f"Type 'YES' to confirm deletion: ")
        if confirm != "YES":
            print("Deletion cancelled.")
            return

    # 3. Execute (or simulate) deletion
    deleted_count = 0
    errors = 0

    print(f"Starting deletion (Mode: {'DRY RUN / SIMULATION' if DRY_RUN else 'LIVE / DESTRUCTIVE'})...")

    for file_name in tqdm(files_to_delete, unit="file"):
        full_path = os.path.join(files_directory, file_name)

        try:
            if os.path.exists(full_path):
                if not DRY_RUN:
                    os.remove(full_path)  # The actual deletion happens here
                deleted_count += 1
            else:
                # File already gone or path incorrect
                pass

        except Exception as e:
            if not DRY_RUN:
                print(f"Could not delete {file_name}: {e}")
            errors += 1

    # 4. Status Report
    print("-" * 30)
    if DRY_RUN:
        print(f"--- DRY RUN COMPLETE ---")
        print(f"The script WOULD have deleted: {deleted_count} files.")
        print(f"To perform actual deletion to: DRY_RUN = False")
    else:
        print(f"--- DELETION COMPLETE ---")
        print(f"Successfully deleted: {deleted_count} files.")
        if errors > 0:
            print(f"Errors encountered: {errors}")


if __name__ == "__main__":
    delete_flagged_files()