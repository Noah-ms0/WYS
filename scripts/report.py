import re,glob,os,subprocess,time
D=sorted(glob.glob('runs/*15000'),key=os.path.getmtime)[-1]
t=open(D+'/train.log').read()
b=re.split(r'Learning iteration (\d+)/15000',t)
rows=[]
for i in range(1,len(b),2):
    def g(k):
        m=re.search(k+r':\s+(-?[\d.]+)',b[i+1]); return float(m.group(1)) if m else None
    el=re.search(r'Time elapsed:\s+([\d:]+)',b[i+1]); eta=re.search(r'ETA:\s+([\d:]+)',b[i+1])
    if g('Iteration time') and eta: rows.append((int(b[i]),g('Iteration time'),g('Curriculum/terrain_levels'),g('Mean episode length'),g('Mean reward'),el[1],eta[1]))
print(time.strftime('%F %T'),'last',rows[-1])
n=[r[1] for r in rows[-500:] if r[1]<20]; print('avg iter',round(sum(n)/len(n),2))
rf=[(r[0],r[1]) for r in rows if r[1]>20]; print('refits',len(rf),rf[-5:])
marks=[x for x in (300,500,600,1000,1500,2000,3000,4000,5000,6000,7500,10000,12500) if x<rows[-1][0]]+[rows[-1][0]]
for m in marks:
    w=[x for x in rows if m-50<x[0]<=m]
    print(m,'terrain',round(sum(x[2] for x in w)/len(w),3),'eplen',round(sum(x[3] for x in w)/len(w)),'rew',round(sum(x[4] for x in w)/len(w),2))
print('errors',len(re.findall(r'Traceback|OutOfMemory',t)),'fpm saved',len(re.findall(r'\[FPM\] saved',t)))
v=re.findall(r'\[FPM\] epoch 25/25\s+train ([\d.]+)\s+val ([\d.]+)',t); print('fpm val last',v[-3:])
r=[l.split(', ') for l in open(D+'/gpu.csv') if l.count(',')==5]
print('mem now',r[-1][1],'peak',max(int(x[1]) for x in r),'temp now/max',r[-1][4],max(int(x[4]) for x in r),'W',r[-1][3],'util',r[-1][2])
print('running',subprocess.run(['pgrep','-f','[r]un_name long4096'],capture_output=True).returncode==0)
print(sorted(f for f in os.listdir(sorted(glob.glob('src/logs/rsl_rl/g1_fpm/*long4096_seed1'))[-1]) if f.startswith('model_')))
