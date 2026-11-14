import json,collections,sys,os
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for y in ['26','25']:
  r=json.load(open(os.path.join(ROOT,'dl','icml%s.json'%y)))['results']
  orals={x['name'].strip().lower() for x in r if x['eventtype']=='Oral'}
  posters=[x for x in r if x['eventtype']=='Poster']
  seen={}
  for x in posters: seen[x['name'].strip().lower()]=x
  print('==',y,'unique posters',len(seen),'orals',len(orals), 'orals w/o poster', len(orals-set(seen)))
  top=collections.Counter(); tot=collections.Counter(); orc=collections.Counter(); spc=collections.Counter()
  sub=collections.Counter(); subo=collections.Counter(); subs=collections.Counter()
  for k,x in seen.items():
    t=x['topic'] or 'None'; a=t.split('->')[0]
    tot[a]+=1; sub[t]+=1
    if k in orals: orc[a]+=1; subo[t]+=1
    elif 'spotlight' in (x['decision'] or '').lower(): spc[a]+=1; subs[t]+=1
  N=sum(tot.values())
  print('decisions', collections.Counter(x['decision'] for x in seen.values()))
  for a,n in tot.most_common():
    print(f'{a:45s} {n:5d} {100*n/N:5.1f}%  oral {orc[a]:3d} ({100*orc[a]/n:4.1f}%) spot {spc[a]:3d} ({100*spc[a]/n:4.1f}%)  o+s {100*(orc[a]+spc[a])/n:4.1f}%')
  print('--- subtopics top 30')
  for t,n in sub.most_common(30):
    print(f'{t:70s} {n:5d} {100*n/N:4.1f}% oral {subo[t]:3d} spot {subs[t]:3d} o+s {100*(subo[t]+subs[t])/n:4.1f}%')
