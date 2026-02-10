import os
import csv
import multiprocessing
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def process_file(file_path):
    try:
        file_name = os.path.basename(file_path)

        min_empty = float('inf')
        max_empty = -1
        has_empty_cells = False
        row_count = 0

        with open(file_path, 'r', newline='', encoding='utf-8') as csv_in:
            reader = csv.reader(csv_in)

            for row in reader:
                if not row: continue
                row_count += 1

                # Count empty strings in this specific row
                empty_in_row = row.count("")

                if empty_in_row > 0:
                    has_empty_cells = True

                # Update Min/Max tracking
                if empty_in_row < min_empty:
                    min_empty = empty_in_row
                if empty_in_row > max_empty:
                    max_empty = empty_in_row

        # --- FORMAT RESULTS ---

        # Handle empty files
        if row_count == 0:
            return (file_name, "0; 0", "NO")

        # Format Empty Cells Range
        empty_range_str = f"{min_empty}; {max_empty}"

        # Format Boolean Flag
        has_empty_str = "YES" if has_empty_cells else "NO"

        return (file_name, empty_range_str, has_empty_str)

    except Exception:
        return (os.path.basename(file_path), "ERROR", "ERROR")


def main():
    # --- CONFIGURATION ---
    # target_folder = r"C:\Users\anton\Downloads\MIMIC_IV_ECG_CSV_MICROVOLTS_v3\MIMIC_IV_ECG_CSV_MICROVOLTS_v3\files"
    target_folder = r"C:\Users\anton\Downloads\Test"
    log_output = "ecg_consistency_log.csv"
    # 1. Collect all file paths
    all_files = []
    print(f"Scanning directory: {target_folder}...")
    for root, _, files in os.walk(target_folder):
        for file in files:
            if file.lower().endswith('.csv'):
                all_files.append(os.path.join(root, file))

    total_files = len(all_files)
    print(f"Found {total_files} files. Starting parallel processing on {cpu_count()} cores...")

    # 2. Write Headers and Process
    with open(log_output, 'w', newline='', encoding='utf-8') as log_out:
        writer = csv.writer(log_out)
        writer.writerow(['File Name', 'Empty Cells (Min; Max)', 'Has Empty Cells'])

        with Pool(processes=cpu_count()) as pool:
            # chunksize=50 helps keep the CPU fed with work
            results = pool.imap_unordered(process_file, all_files, chunksize=50)

            for result in tqdm(results, total=total_files, unit="file"):
                writer.writerow(result)

    print(f"\nProcessing complete. Log saved to: {log_output}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()