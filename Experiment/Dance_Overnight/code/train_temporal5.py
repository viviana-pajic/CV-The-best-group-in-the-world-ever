"""A wider temporal window, with the same matched training setup."""
from train_temporal import *
if __name__=='__main__':
    while not (OUT/'runs/v8_temporal2_s17/complete.json').exists():time.sleep(5)
    train_temporal('v8_temporal5_s17',5)
