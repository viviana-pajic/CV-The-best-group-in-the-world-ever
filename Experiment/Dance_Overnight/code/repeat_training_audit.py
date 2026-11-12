"""Repeat the chosen architecture with two seeds; development-only robustness audit."""
from train_support import *

if __name__=='__main__':
    dest=OUT/'seed_repeatability';dest.mkdir(exist_ok=True)
    write(dest/'protocol.json',{'scope':'Repeat the selected expanded architecture with seeds29 and43. Identical original969-crop manifest, augmentation, optimizer and80-epoch/early-stop schedule. Development-only robustness audit; no automatic replacement of the frozen confirmation candidate. Evaluate best_joint checkpoints at fixed0.25 threshold with ten crop/flip views. Average point and reportability for a three-seed ensemble, without threshold search.','seeds':[17,29,43],'checkpoint':'best_joint.pt','mode':'crop5flip','threshold':.25})
    for seed in [29,43]:
        train(f'v9_repeat_wide_s{seed}',OUT/'data/development_ending.json',seed=seed,attention=True,wide=True,ignore_uncertain=True,weight_floor=30)
