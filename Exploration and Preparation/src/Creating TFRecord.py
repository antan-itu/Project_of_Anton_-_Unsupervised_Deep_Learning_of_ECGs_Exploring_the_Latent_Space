import os
import glob
import numpy as np
import tensorflow as tf
from multiprocessing import Pool, cpu_count

DATA_DIR = "/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/" \
"Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/Data/MIMIC-IV_Subset/Test"

OUTPUT_DIR = "/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/" \
"Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/Data/MIMIC-IV_Subset/"

EXPECTED_SHAPE = (5000, 12)


def serialize_example(ecg_bytes):
    feature = {
        "ecg": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[ecg_bytes])
        )
    }
    example = tf.train.Example(
        features=tf.train.Features(feature=feature)
    )
    return example.SerializeToString()


def process_files(args):
    file_list, shard_id = args
    output_file = os.path.join(OUTPUT_DIR, f"subset_{shard_id:03d}.tfrecord")

    success = 0
    skipped = 0

    with tf.io.TFRecordWriter(output_file) as writer:
        for file in file_list:
            try:
                data = np.loadtxt(file, delimiter=",", dtype=np.float32)

                if data.shape != EXPECTED_SHAPE:
                    skipped += 1
                    continue

                mean = data.mean()
                std = data.std()
                if std > 0:
                    data = (data - mean) / std

                ecg_bytes = data.tobytes()
                example = serialize_example(ecg_bytes)
                writer.write(example)

                success += 1

            except Exception:
                skipped += 1

    return success, skipped


def chunkify(lst, n_chunks):
    """Split list into n roughly equal chunks"""
    return [lst[i::n_chunks] for i in range(n_chunks)]


def convert_parallel(file_pattern, num_workers=None):
    if num_workers is None:
        num_workers = cpu_count()

    print(f"Using {num_workers} worker processes")

    # Get file list once
    csv_files = glob.glob(file_pattern)
    print(f"Found {len(csv_files)} CSV files")

    # Split into chunks
    chunks = chunkify(csv_files, num_workers)

    with Pool(num_workers) as pool:
        results = pool.map(process_files, [(chunks[i], i) for i in range(num_workers)])

    total_success = sum(r[0] for r in results)
    total_skipped = sum(r[1] for r in results)

    print(f"\nDone!")
    print(f"Saved: {total_success}")
    print(f"Skipped: {total_skipped}")
    print(f"Created {num_workers} TFRecord shards")


if __name__ == "__main__":
    convert_parallel(os.path.join(DATA_DIR, "*.csv"))
