"""Assess the synthetic-occlusion models on real, unmasked development images."""
from evaluate_ending import main,OUT
import time
if __name__=='__main__':
    while not (OUT/'runs/v7_occlusion_p0.6_s17/complete.json').exists():time.sleep(5)
    main(['v5_attention_s17','v6_attention_wide_s17','v7_occlusion_p0.3_s17','v7_occlusion_p0.6_s17'],'occlusion_candidates')
