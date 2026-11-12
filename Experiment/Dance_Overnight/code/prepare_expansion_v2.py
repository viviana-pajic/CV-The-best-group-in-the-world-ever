"""Additional blind support-label sheets for the larger-data comparison."""
from pathlib import Path
import prepare_expansion as p
p.OUT=p.ROOT/'Dance_Overnight/data/expansion_v2';p.OUT.mkdir(parents=True,exist_ok=True)
p.TRAIN=[3304,3318,3346,3479,3493,3521,3829,3843,3871,4129,4143,4171,4165,4179,4193,4207,4221,4235,4329,4343,4371,5329,5343,5371]
p.VAL=[]
p.main()
(p.OUT/'box_edits.json').write_text('{}')
(p.OUT/'manual_points.txt').write_text('# Additional fresh assistant visual support labels; no model-derived coordinates.\n# F=feet, K=knees, S=seat, H=hidden, U=uncertain/airborne, D=duplicate/invalid.\n')
