# %%
# %load_ext autoreload
# %reload_ext autoreload
# %autoreload 2
######TODO: CHANGE MULTICORALREEF EXTRACTOR TO ONLY HAVE 2 CLASSES NOT 1 
# %%
from pyha_analyzer import PyhaTrainer, PyhaTrainingArguments, extractors
#from pyha_analyzer import extractors

# %%
#you have 1148 williams nondegraded files that are getting broken into (60 seconds/5 seconds)= 12 spectograms each so a total of 13776 spectograms. Therefore, you can get 13776 Paola nondegraded files
#you have 300 williams degraded files that are getting broken into (60 seconds/5 seconds)= 12 spectograms each so a total of 3600 spectograms. Therefore, you can get (13776-3600) Paola degraded files so that degraded and nondegraded are equal
# coralreef_extractor = extractors.MultiCoralReef()
# coral_ads = coralreef_extractor("/home/s.kamboj.400/mount/files/")
coralreef_extractor = extractors.MultiCoralReef()
coral_ads = coralreef_extractor("/home/s.kamboj.400/mount/files/", 100)
# coral_ads

# %%
from pathlib import Path
import pandas as pd

export_rows = []
for split_name in ["train", "valid", "test"]:
    split_ds = coral_ads[split_name].remove_columns(["audio"])
    for sample in split_ds:
        export_rows.append({
            "split": split_name,
            "sample_rate": sample["sample_rate"],
            "filepath": sample["filepath"],
            "site": sample["site"],
            "dataset": sample["dataset"],
            "labels": ",".join(map(str, sample["labels"])),
            "audio_path": sample["filepath"],
        })

coral_ads_csv = pd.DataFrame(export_rows)
csv_path = Path("coral_ads_100_percent.csv")
coral_ads_csv.to_csv(csv_path, index=False)
csv_path.resolve()

# %%
from collections import Counter
print(Counter([sample['labels'].index(1) for sample in coral_ads["test"]]))


# %%
#from pyha_analyzer.preprocessors import MelSpectrogramPreprocessors
#converts williams to many spectograms & paola to only one spectogram
from pyha_analyzer.preprocessors import MelSpectrogramPreprocessors

# TODO: allow for normalization system

# preprocessor acts as a function for processing
# class allows us to configure parameters and whatnot
preprocessor = MelSpectrogramPreprocessors(duration=5, class_list=coral_ads["train"].features["labels"].feature.names)

coral_ads["train"].set_transform(preprocessor)
coral_ads["valid"].set_transform(preprocessor)
coral_ads["test"].set_transform(preprocessor)
coral_ads["train"][[0, 1]]["audio"][0].mean()

# %%
coral_ads["test"][0]

# %%
from pyha_analyzer.models import EfficentNet
#model = EfficentNet(num_classes=len(coral_ads["train"].features["ebird_code"].names))
model = EfficentNet(num_classes=2)

# %%
import torch
from pyha_analyzer import constants

torch.cuda.empty_cache()
args = PyhaTrainingArguments(
    working_dir="working_dir",
    run_name= constants.DEFAULT_RUN_NAME,
    project_name=constants.DEFAULT_PROJECT_NAME
)
args.num_train_epochs = 1
args.eval_steps = 20


trainer = PyhaTrainer(
    model=model,
    dataset=coral_ads,
    training_args=args,
)
trainer.train()

# %%
#Uncomment the following lines to save the git commit hash and model state_dict as pt file
import subprocess
import torch

git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()

# with open("training_info.txt", "w") as f:
#     f.write(f"Git commit: {git_hash}\n")

#save model in .pt file associated with git hash. This is useful for reproducibility, so that we can always refer back to the exact code used for training.
model_save_path = f"100_percent_{git_hash[:7]}.pt"
torch.save(trainer.model.state_dict(), model_save_path)

# %%
print(trainer.evaluate(eval_dataset=coral_ads["test"], metric_key_prefix="Soundscape"))

