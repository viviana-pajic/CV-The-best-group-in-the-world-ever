"""Common validation comparison; uncertain labels cannot be called negatives."""
import copy,argparse
from candidate_evaluate import OUT,read,write,oracle_scores

def main():
    p=argparse.ArgumentParser();p.add_argument('--pattern',default='v*_all_scores.json');args=p.parse_args()
    manifest=read(OUT/'data/development_v2.json');clean=copy.deepcopy(manifest)
    for d in clean.values():
        for k in d['val']:d['labels'][str(k)]=[o for o in d['labels'][str(k)] if o['contact']!='uncertain']
    summary=[]
    for path in sorted((OUT/'candidates').glob(args.pattern)):
        for row in read(path):
            cache=OUT/'candidates'/f"{row['run']}_{row['checkpoint'].removesuffix('.pt')}_{row['mode']}_validation.json"
            if not cache.exists():continue
            pred=read(cache);r=copy.deepcopy(row);r['legacy_validation']=r.pop('validation');r['validation']=oracle_scores(pred,clean,r['decoder']);r['evaluation_scope']='Common9-frame validation, 104 visible and50 hidden contacts. Fifteen ambiguous contacts excluded from correctness/F1, with their outputs retained in legacy by_type. All are development data.';summary.append(r)
    summary.sort(key=lambda r:(r['validation']['selected']['joint_f1'],r['validation']['selected']['correct']),reverse=True)
    write(OUT/'candidates/clean_ranking.json',summary)
    for r in summary[:15]:print(r['name'],r['validation']['selected'])

if __name__=='__main__':main()
