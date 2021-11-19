import csv
import glob
import os
from random import shuffle

import numpy as np

from trdg.generators import GeneratorFromStrings


# Get a bunch of real d2 text..
armors = np.loadtxt('text/Armor.txt', dtype="U", delimiter='\t', skiprows=1, usecols=0)
runes = np.loadtxt('text/Runes.txt', dtype="U", delimiter='\t', skiprows=1, usecols=1, encoding='iso-8859-1')
rune_combos = np.loadtxt('text/Runes.txt', dtype="U", delimiter='\t', skiprows=1, usecols=13, encoding='iso-8859-1')
rune_combos = [f"'{r}'" for r in rune_combos if r != '']  # in game they are surrounded 'like this'
set_items = np.loadtxt('text/SetItems.txt', dtype="U", delimiter='\t', skiprows=1, usecols=1, encoding='iso-8859-1')
sets = np.loadtxt('text/Sets.txt', dtype="U", delimiter='\t', skiprows=1, usecols=1, encoding='iso-8859-1')
unique_items = np.loadtxt('text/UniqueItems.txt', dtype="U", delimiter='\t', skiprows=1, usecols=0, encoding='iso-8859-1')
unique_item_types = np.loadtxt('text/UniqueItems.txt', dtype="U", delimiter='\t', skiprows=1, usecols=9, encoding='iso-8859-1')
weapons = np.loadtxt('text/Weapons.txt', dtype="U", delimiter='\t', skiprows=1, usecols=0, encoding='iso-8859-1')

numbers = np.loadtxt('text/numbers.txt', dtype="U", delimiter='\t', usecols=0, encoding='iso-8859-1')

# Try some new stuff on..
merged = np.loadtxt('text/Merged.txt', dtype="U", delimiter='\t', skiprows=0, usecols=1, encoding='iso-8859-1')

# and a lot more strings...
strings = np.loadtxt('text/strings.txt', dtype="U", delimiter='\t', usecols=1, encoding='iso-8859-1')

combined = np.concatenate((armors, runes, rune_combos, set_items, sets, unique_items, unique_item_types, weapons, numbers, merged, strings))
#combined = np.concatenate((merged,))
# combined = np.concatenate((strings,))

# strip any dupes + empties + things longer than 32 chars + shuffle them
combined = list(set([c for c in combined if c != '' and len(c) < 50]))
shuffle(combined)

# Generate a shitload of images
outputs = [
    # Lighter text
    ("all_data/en_train_filtered", 20000, "#b5b5b5", 2),
    ("all_data/en_train_filtered", 20000, "#b5b5b5", 0),
    ("all_data/en_val", 4200, "#b5b5b5", 2),
    ("all_data/en_val", 4200, "#b5b5b5", 0),

    # Darker text
    ("all_data/en_train_filtered", 20000, "#282828", 2),
    ("all_data/en_train_filtered", 20000, "#282828", 0),
    ("all_data/en_val", 4200, "#282828", 2),
    ("all_data/en_val", 4200, "#282828", 0),
]

for size in (32, 48):
    for output_dir, count, text_color, background_type in outputs:
        generator = GeneratorFromStrings(
            combined,
            fonts=glob.glob("fonts/*"),
            count=count,
            # text_color="#c2c2c2",  # light af
            # text_color="#b5b5b5",  # normal socketed/ethereal in d2
            text_color=text_color,
            # background_type=2,
            background_type=background_type,
            space_width=1.5,
            size=size
        )

        os.makedirs(output_dir, exist_ok=True)

        # Start counter at one or # of files
        if len(os.listdir(output_dir)) > 0:
            start_counter = len(os.listdir(output_dir))
        else:
            start_counter = 1

        # check and store a bool for whether labels file existed, so we can write headers later if needed (only when file didn't exist)
        labels_file_existed = os.path.exists(os.path.join(output_dir, "labels.csv"))

        with open(os.path.join(output_dir, "labels.csv"), 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # write csv header, if file didn't already exist
            if not labels_file_existed:
                writer.writerow(["filename", "words"])

            for i, (img, label) in enumerate(generator):
                print(img, label)
                img.save(f"{output_dir}/generated_10_26_{start_counter + i}.jpg")
                writer.writerow([f"generated_10_26_{start_counter + i}.jpg", label])
