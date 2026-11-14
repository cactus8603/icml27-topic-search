import json,collections,re,os
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for y in ['26','25']:
  r=json.load(open(os.path.join(ROOT,'dl','icml%s.json'%y)))['results']
  orals={x['name'].strip().lower() for x in r if x['eventtype']=='Oral'}
  seen={x['name'].strip().lower():x for x in r if x['eventtype']=='Poster'}
  if y=='26':
    none=[k for k,x in seen.items() if not x['topic']]
    print('None-topic sample', none[:8], sum(k.startswith('position') for k in none))
  kws={'LLM/language model':r'\bllms?\b|language model','reasoning':r'reason','RL (incl RLHF/RLVR)':r'reinforcement|\brl\b|rlhf|rlvr|policy optim|grpo','diffusion/flow':r'diffusion|flow match','agent':r'\bagent','transformer/attention':r'transformer|attention','in-context':r'in-context','scaling law':r'scaling','theory words':r'provabl|theor|bound|convergence','interpretab/mechanistic':r'interpretab|mechanis|circuit|sparse autoenc|probing','safety/align':r'safety|align|jailbreak','graph':r'graph','time series':r'time series|time-series','protein/molecule':r'protein|molec|drug'}
  N=len(seen); print('==',y,N)
  for n,p in kws.items():
    m=[k for k in seen if re.search(p,k)]
    o=[k for k in m if k in orals]
    print(f'{n:28s} {len(m):5d} {100*len(m)/N:5.1f}% orals {len(o)}')
