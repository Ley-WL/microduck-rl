import json,concurrent.futures
from optimize_walk_slip import OUT,TRAIN,HOLD,evaluate,init
if __name__=='__main__':
 report=json.loads((OUT/'heldout.json').read_text());cases=[dict(c,seconds=30.,seed=83) for c in TRAIN]+[dict(c,seconds=30.,seed=109) for c in HOLD[:8]]
 with concurrent.futures.ProcessPoolExecutor(max_workers=3,initializer=init) as pool: rows=list(pool.map(evaluate,[([0]*5,cases),(report['candidate']['params'],cases)]))
 lost=[i for i,(a,b) in enumerate(zip(rows[0]['cases'],rows[1]['cases'])) if a['success'] and not b['success']]
 result={'baseline':rows[0],'candidate':rows[1],'lostBaselineCases':lost,'total':len(cases)}
 (OUT/'long-holdout.json').write_text(json.dumps(result,indent=2));print(json.dumps({'baseline':rows[0]['successes'],'candidate':rows[1]['successes'],'total':len(cases),'lost':lost}),flush=True)
