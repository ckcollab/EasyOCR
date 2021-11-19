import csv
import glob
import os

from trdg.generators import GeneratorFromRandom


output_dirs = ["all_data/", "all_data/en_val"]

for output_dir in output_dirs:
    generator = GeneratorFromRandom(
        fonts=glob.glob("fonts/*"),
        count=1000,
    )

    with open(os.path.join(output_dir, "labels.csv"), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # write csv header
        writer.writerow(["filename", "words"])
        for i, (img, label) in enumerate(generator):
            img.save(f"{output_dir}/{i}.jpg")
            writer.writerow([f"{i}.jpg", label])
