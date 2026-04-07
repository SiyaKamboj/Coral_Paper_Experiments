from .defaultExtractors import DefaultExtractor
from datasets import ClassLabel, Sequence, Audio
from .. import AudioDataset
import os
from datasets import Dataset, concatenate_datasets
import datasets
import pandas as pd
import soundfile as sf
import random
import fnmatch
import sys
from collections import Counter, defaultdict


def parse_config(config_path):
    metadata = {}
    with open(config_path, "r") as f:
        for line in f:
            if ":" in line:
                key, val = line.split(":", 1)
                if ((key.strip()== "Device ID") or (key.strip() == "Sample rate (Hz)")):
                    metadata[key.strip()] = val.strip()
    return metadata


def extract_features(wav, label, site, dataset):
    if label==0:
        oneHotEncodedLabel = [1,0] #Non_Degraded_Reef
    elif (label==1):
    #else:
        oneHotEncodedLabel = [0,1] #Degraded_Reef
    else: #if label is 2 (neither)
    #     oneHotEncodedLabel = [0,0,1] #Unknown
        return #should not happen
        
    try:
        with sf.SoundFile(wav, "r") as audio_file:
            sample_rate = audio_file.samplerate
    except Exception as e:
        print(f"Exception reading {wav}: {e}")
        return
        
    return {
        "sample_rate": sample_rate,
        "labels": oneHotEncodedLabel,
        "filepath": str(wav),
        "audio": str(wav),
        "audio_in": {"array": str(wav), "sampling_rate": sample_rate},
        "site": site,
        "dataset": dataset
    }


class MultiCoralReef(DefaultExtractor):
    def __init__(self):
        super().__init__("CoralReef")

    def __call__(self, audio_path, sampling=False, data_percentage=100):
        # Constants
        if data_percentage <= 0 or data_percentage > 100:
            raise ValueError("data_percentage must be in the range (0, 100].")

        sampling_fraction = data_percentage / 100.0

        # Organize into buckets
        buckets = {
            ('PaolaMexico', 0): [],
            ('PaolaMexico', 1): [],
            ('PaolaCostaRica', 0): [],
            ('PaolaCostaRica', 1): [],
            ('Williams_et_al_2024', 0): [],
            ('Williams_et_al_2024', 1): [],
            #('sandy area_no corals' , 2) : []
        }

        supported_exts = (".wav", ".flac", ".mp3", ".ogg", ".m4a")

        for root, dirs, files in os.walk(audio_path):
            for file in files:
                if not file.lower().endswith(supported_exts):
                    continue

                file_path = os.path.join(root, file)

                # verify the file is decodable before doing any further processing so that invalid files are not possibly computed in grand total. slows down preprocessing but valuable step
                try:
                    with sf.SoundFile(file_path, "r"):
                        pass
                except Exception as exc:
                    print(f"Skipping invalid audio {file_path}: {type(exc).__name__}: {exc}")
                    continue

                # Detect dataset
                if "PaolaMexico" in file_path:
                    dataset = "PaolaMexico"
                elif "PaolaCostaRica" in file_path:
                    dataset="PaolaCostaRica"
                elif "Williams_et_al_2024" in file_path:
                    dataset = "Williams_et_al_2024"
                # elif "sandy area_no corals" in file_path:
                #     dataset="sandy area_no corals"
                else:
                    continue  # skip others (there shouldn't be otheres)

                # Detect label
                if "Non_Degraded_Reef" in file_path:
                    label = 0
                    site = "Non_Degraded_Reef"
                elif "Degraded_Reef" in file_path:
                    label = 1
                    site = "Degraded_Reef"
                else:
                    label=2
                    site = "Neither" #useful for sandy_area_no_corals

                buckets[(dataset, label)].append((file_path, label, site, dataset))
                
        # Define how many spectrograms each dataset contributes per file
        dataset_multipliers = {
            'Williams_et_al_2024': 1,
            'PaolaMexico': 1,
            'PaolaCostaRica': 1,
            #"sandy area_no corals" : 1
        }

        random.seed(42)

        # Apply the requested percentage to each dataset/label bucket before balancing.
        for bucket_key, items in buckets.items():
            if not items or sampling_fraction == 1.0:
                continue

            num_to_keep = max(1, int(len(items) * sampling_fraction))
            num_to_keep = min(num_to_keep, len(items))
            buckets[bucket_key] = random.sample(items, num_to_keep)
            print(f"keeping {num_to_keep}/{len(items)} from {bucket_key} based on data_percentage={data_percentage}")

        #does not do equal split when it comes to lin et al. 
        min_size=sys.maxsize
        for (dataset, label), items in buckets.items():
            size = len(items) * dataset_multipliers[dataset] #number of spectogrgams that the current dataset + label contributetes
            if (size >0 and size<min_size):
                min_size= size
                datasetOfMinSize= dataset

        if min_size == sys.maxsize:
            raise ValueError("No audio samples remained after applying data_percentage.")

        print(f"min size is {min_size} from dataset {datasetOfMinSize}")
        
        sampled = []
        for (dataset, label), items in buckets.items():
            #due to multiplicative factors, it may not be a perfect split 
            numToSample = min (int(min_size / dataset_multipliers[dataset]), len(items))
            sampled += random.sample(items, numToSample)
            print(f"sampling {numToSample} from ({dataset}, {label})")

        
        # Step 7: Feature extraction
        all_data = []
        for file_path, label, site, dataset in sampled:
            try:
                curr_data = extract_features(file_path, label, site, dataset)
                if curr_data is not None:
                    all_data.append(curr_data)
            except Exception as e:
                print(f"Skipping {file_path} due to {type(e).__name__}: {e}")
                continue

        # # Summary
        print(f"Loaded: {len(all_data)} samples")


        ds = Dataset.from_list(all_data)
        class_list = ["Non_Degraded_Reef" , "Degraded_Reef"]
        #class_list = ["Non_Degraded_Reef" , "Degraded_Reef", "Neither"]

        # Hold Williams + Costa Rica for training/validation and Paola Mexico for testing
        paola_mexico_ds = ds.filter(lambda x: x["dataset"] == "PaolaMexico")
        non_paola_ds = ds.filter(lambda x: x["dataset"] != "PaolaMexico")
        if len(paola_mexico_ds) == 0:
            raise ValueError("No PaolaMexico samples found for testing.")

        # Keep a small portion of \"Neither\" (non-Paola) examples for the test set so all classes are evaluated
        neither_ds = non_paola_ds.filter(lambda x: x["site"] == "Neither")
        non_paola_without_neither = non_paola_ds.filter(lambda x: x["site"] != "Neither")
        test_neither_ratio = 0.1 #PUT 10% OF NEIther INTO TEST SPLIT SO THAT U GET REPRESENTATION FROM ALL CLASSES so that u can use metrics like recall, accuracy, etc
        test_neither_ds = None
        train_neither_ds = None
        if len(neither_ds) > 0:
            num_neither_test = max(1, int(len(neither_ds) * test_neither_ratio))
            num_neither_test = min(num_neither_test, len(neither_ds))
            indices = list(range(len(neither_ds)))
            test_indices = sorted(random.sample(indices, num_neither_test))
            remaining_indices = sorted(set(indices) - set(test_indices))
            test_neither_ds = neither_ds.select(test_indices)
            if remaining_indices:
                train_neither_ds = neither_ds.select(remaining_indices)
        else:
            print("Warning: No 'Neither' samples available outside PaolaMexico; test split will not include class 2.")

        train_components = []
        if len(non_paola_without_neither) > 0:
            train_components.append(non_paola_without_neither)
        if train_neither_ds is not None and len(train_neither_ds) > 0:
            train_components.append(train_neither_ds)
        if not train_components:
            raise ValueError("No Williams_et_al_2024 or PaolaCostaRica samples found for training.")
        train_valid_ds = train_components[0] if len(train_components) == 1 else concatenate_datasets(train_components)

        test_components = [paola_mexico_ds]
        if test_neither_ds is not None and len(test_neither_ds) > 0:
            test_components.append(test_neither_ds)
        paola_mexico_test = test_components[0] if len(test_components) == 1 else concatenate_datasets(test_components)

        # Split remaining datasets into train/validation
        split_ds = train_valid_ds.train_test_split(test_size=0.3, seed=42)
        mutlilabel_class_label = Sequence(ClassLabel(names=class_list))

        datasets_dict = {
            "train": split_ds["train"],
            "valid": split_ds["test"],
            "test": paola_mexico_test
        }

        for split_name, split_dataset in datasets_dict.items():
            datasets_dict[split_name] = split_dataset.cast_column("labels", mutlilabel_class_label)
            datasets_dict[split_name] = datasets_dict[split_name].cast_column("audio", Audio())
            # summarize dataset/label distribution per split
            ds_labels = datasets_dict[split_name]["dataset"]
            site_labels = datasets_dict[split_name]["site"]
            summary = defaultdict(Counter)
            for dataset_name, site_name in zip(ds_labels, site_labels):
                summary[dataset_name][site_name] += 1
            print(f"{split_name} split distribution:")
            for dataset_name, label_counts in summary.items():
                for site_name, count in label_counts.items():
                    print(f"  {dataset_name} - {site_name}: {count}")

        return AudioDataset(
                    datasets_dict,
                    "null"
                )
