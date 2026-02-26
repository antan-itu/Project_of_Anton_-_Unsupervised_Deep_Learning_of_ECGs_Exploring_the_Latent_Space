import h5py

h5_path = r"/home/akokholm/mnt/SUN-BMI-EC-MIMIC-ECG/ECG.hdf5"

with h5py.File(h5_path, 'r') as f:
    print("--- ROOT LEVEL ---")
    print("Root keys:", list(f.keys()))
    print("\n--- DEEP DIVE (First 20 items) ---")
    
    count = 0
    def visit_func(name, obj):
        global count
        if count < 20:
            print(f"\nPath: {name}")
            print(f"Type: {'Group (Folder)' if isinstance(obj, h5py.Group) else 'Dataset (Data)'}")
            
            # Print attributes if they exist
            if obj.attrs:
                for k, v in obj.attrs.items():
                    print(f"  Attr [{k}]: {v}")
            else:
                print("  (No attributes)")
            count += 1
            
    f.visititems(visit_func)