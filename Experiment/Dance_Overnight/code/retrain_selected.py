"""Train the selected support architecture again under a new run name."""
import argparse
from train_support import *

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',default='replication_support_s17');p.add_argument('--seed',type=int,default=17);p.add_argument('--variant',choices=['established','expanded'],default='established');p.add_argument('--manifest',type=Path);a=p.parse_args();assert not (OUT/'runs'/a.name).exists(),'Choose a new run name'
    manifest=a.manifest or OUT/'data'/('development_ending.json' if a.variant=='expanded' else 'development_expanded.json')
    train(a.name,manifest,seed=a.seed,attention=True,wide=a.variant=='expanded',ignore_uncertain=True,weight_floor=30)
