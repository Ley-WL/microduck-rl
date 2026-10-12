import json,concurrent.futures,numpy as np
from optimize_walk_slip import OUT,TRAIN,HOLD,evaluate,init
if __name__=='__main__':
 r=json.loads((OUT/'heldout.json').read_text());search=json.loads((OUT/'search.json').read_text());p=np.array(r['candidate']['params'])
 candidates=[(p*.5).tolist(),(p*.75).tolist(),[.03794700122617162,-.0011737232051485524,-.03983266718602354,.06459944959272484,.0044388603531432275]]
 cases=[dict(c,seconds=30.,seed=83) for c in TRAIN]+[dict(c,seconds=30.,seed=109) for c in HOLD[:8]]
 base=json.loads((OUT/'long-holdout.json').read_text())['baseline'];rows=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=3,initializer=init) as pool:
  for result in pool.map(evaluate,[(p,cases) for p in candidates]):
   result['lostBaselineCases']=[i for i,(b,c) in enumerate(zip(base['cases'],result['cases'])) if b['success'] and not c['success']]
   rows.append(result);print('CANDIDATE',result['params'],result['successes'],'lost',result['lostBaselineCases'],flush=True)
 (OUT/'second-screen.json').write_text(json.dumps({'baseline':base,'candidates':rows},indent=2))
