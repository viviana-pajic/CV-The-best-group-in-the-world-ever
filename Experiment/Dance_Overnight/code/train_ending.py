"""Ending-data ablation. Earlier frozen choices and evaluations remain unchanged."""
from train_support import *

if __name__=='__main__':
    manifest=OUT/'data/development_ending.json'
    for name,seed,condition,wide in [('attention_s17',17,False,False),('attention_s29',29,False,False),('mask_attention_s17',17,True,False),('attention_wide_s17',17,False,True)]:
        train('v6_'+name,manifest,seed,condition=condition,attention=True,wide=wide,ignore_uncertain=True,weight_floor=30)
