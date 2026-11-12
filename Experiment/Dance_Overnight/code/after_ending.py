"""Run development evaluation when the sequential training queue finishes."""
import time
from pathlib import Path
from evaluate_ending import main
OUT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    while not (OUT/'runs/v6_attention_wide_s17/complete.json').exists():time.sleep(5)
    main()
