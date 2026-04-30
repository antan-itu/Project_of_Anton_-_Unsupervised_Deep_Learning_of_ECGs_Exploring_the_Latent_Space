import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Configurations ---
H5_PATH = "/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/data/MIMIC_IV_ECG_HDF5/mimic_iv_train.h5"
OUTPUT_DIR = "/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/exploration_and_preparation/logs/report_normal_and_af"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_shortened_ecgs():
    print("Opening HDF5 file and scanning labels...")
    
    with h5py.File(H5_PATH, 'r') as f:
        # 1. Extract Ground Truth Labels
        gt_group = f['GT']
        report_cols = [key for key in gt_group.keys() if key.startswith('report_')]
        df_gt_dict = {col: [val.decode('utf-8').strip().upper() for val in gt_group[col][:]] for col in report_cols}
        df_gt = pd.DataFrame(df_gt_dict)
        
        # 2. Define our target rhythms
        sinus_mask = (df_gt['report_0'] == "SINUS RHYTHM") & (df_gt['report_1'] == "")
        afib_mask = (df_gt['report_0'].isin(["ATRIAL FIBRILLATION", "ATRIAL FIBRILLATION."])) & (df_gt['report_1'] == "")
        
        sinus_indices = df_gt.index[sinus_mask].tolist()
        afib_indices = df_gt.index[afib_mask].tolist()
        
        # 3. Recreate the exact random state to grab your specific targets
        np.random.seed(42) 
        chosen_sinus_indices = np.random.choice(sinus_indices, size=10, replace=False)
        chosen_afib_indices = np.random.choice(afib_indices, size=10, replace=False)
        
        # In Python, arrays are 0-indexed, so Example 6 is index 5, and Example 5 is index 4.
        target_nsr_idx = chosen_sinus_indices[5]
        target_afib_idx = chosen_afib_indices[4]
        
        # 4. Creating a time axis
        time_axis = np.linspace(0, 3, 1500) 
        
        def save_plot_with_highlights(waveform, title, filename, is_nsr=False):
            plt.figure(figsize=(16, 7))
            plt.plot(time_axis, waveform, color='black', linewidth=2.5)
            plt.title(title, fontsize=30, fontweight='bold', pad=15)
            plt.xlabel("Time (Seconds)", fontsize=25)
            plt.ylabel("Amplitude (mV)", fontsize=25)
            plt.tick_params(axis='both', which='major', labelsize=20)
            
            # --- ADDING THE COLORS ---
            # We only want to highlight one heartbeat to avoid cluttering the plot.
            # IMPORTANT: You must change these X-axis numbers based on your specific chosen plot!
            if is_nsr:
                # Example: Highlighting a heartbeat that occurs around the 1.0 - 1.5 second mark
                plt.axvspan(1.07, 1.21, color='#1f77b4', alpha=0.3, label='P-Wave')       # Blue
                plt.axvspan(1.27, 1.37, color='#d62728', alpha=0.3, label='QRS Complex')  # Red
                plt.axvspan(1.50, 1.71, color='#2ca02c', alpha=0.3, label='T-Wave')       # Green
                plt.legend(loc='upper right', fontsize=25, framealpha=1.0)
            
            # Standard ECG grid styling
            plt.grid(which='major', color='#dddddd', linewidth=1.2)
            plt.grid(which='minor', color='#eeeeee', linewidth=0.5)
            plt.minorticks_on()
            
            plt.xlim(0, 3) 
            plt.tight_layout()
            
            full_path = os.path.join(OUTPUT_DIR, filename)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: {full_path}")

        # 5. Extract just the first 1500 steps
        print("\nExtracting the 5-second windows...")
        nsr_waveform = f['rhythm_filtered'][target_nsr_idx, :1500, 0]
        afib_waveform = f['rhythm_filtered'][target_afib_idx, :1500, 0]
        
        # 6. Plot and Save
        save_plot_with_highlights(nsr_waveform, "Normal Sinus Rhythm (Lead I)", "Report_NSR_Lead_I_5_sec.png", is_nsr=True)
        save_plot_with_highlights(afib_waveform, "Atrial Fibrillation (Lead I)", "Report_AFib_Lead_I_5_sec.png")

if __name__ == "__main__":
    extract_shortened_ecgs()