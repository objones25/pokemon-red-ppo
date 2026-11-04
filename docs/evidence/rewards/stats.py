import math
def tcdf(t,df,n=4000):
    # two-sided p via Simpson integration of the t pdf
    c=math.gamma((df+1)/2)/(math.sqrt(df*math.pi)*math.gamma(df/2))
    f=lambda x:c*(1+x*x/df)**(-(df+1)/2)
    h=abs(t)/n; s=f(0)+f(abs(t))
    for i in range(1,n): s+=f(i*h)*(4 if i%2 else 2)
    return max(0,1-2*s*h/3)
def tcrit(df):
    lo,hi=1,30
    for _ in range(50):
        m=(lo+hi)/2
        if tcdf(m,df)>.05: lo=m
        else: hi=m
    return lo
def w(name,a,sa,b,sb,n=5):
    d=a-b; se=math.sqrt((sa*sa+sb*sb)/n); df=(sa*sa/n+sb*sb/n)**2/((sa*sa/n)**2/(n-1)+(sb*sb/n)**2/(n-1)); t=d/se; p=tcdf(t,df); c=tcrit(df)*se
    ov=max(a-sa,b-sb)<=min(a+sa,b+sb)
    print(f"{name:42s} {a}±{sa} vs {b}±{sb} d={d:+.2f} se={se:.3f} t={t:+.2f} df={df:.1f} p={p:.3f} CI[{d-c:+.2f},{d+c:+.2f}] SDoverlap={ov}")
print('LEVEL: -Rlvl vs Baseline')
for r in [('milestones',4.60,.7,4.19,.2),('Brock',.95,.05,.99,.05),('Cerulean',.79,.1,.85,.1),('Misty',.31,.3,.27,.3),('Bill',.32,.4,.11,.2),('events',57.73,11.4,51.78,2.1),('party size',1.58,.5,3.48,.3),('entropy',1.24,.2,2.61,.1),('max lvl',27.45,2.8,27.52,3.2),('coords',4560,1015,4275,1406),('brock steps',6117,1758,5587,1235),('cer steps',21352,3898,25299,4950)]: w(*r)
w('-Rlvl vs Fast milestones',4.60,.7,4.40,.8)
print('HEAL: minus-heal vs with-heal')
for s,a,sa,b,sb in [('Squirtle',3.94,.2,4.19,.2),('Fast',3.77,.5,4.40,.8),('Charmander',3.73,.2,3.03,1.6),('Bulbasaur',3.18,.3,2.89,.1),('ChooseStarter',3.40,.4,3.43,1.7)]: w('milestones '+s,a,sa,b,sb)
for s,a,sa,b,sb in [('Squirtle',43.21,2.0,51.78,2.1),('Fast',41.79,5.3,57.21,5.4),('Charmander',45.29,7.2,46.52,19.7),('Bulbasaur',36.55,2.6,30.32,2.2),('Choose',41.49,3.0,48.42,17.7)]: w('events '+s,a,sa,b,sb)
for s,a,sa,b,sb in [('Squirtle',5497,503,5587,1235),('Fast',6531,1185,3753,990),('Charmander',11694,1439,7029,1132),('Bulbasaur',13433,950,9929,841),('Choose',13871,2991,6485,578)]: w('steps to Brock '+s,a,sa,b,sb)
for s,a,sa,b,sb in [('Squirtle',.99,.05,.99,.05),('Fast',.97,.05,.98,.05),('Charmander',.94,.05,.69,.4),('Bulbasaur',.97,.05,.95,.05),('Choose',.97,.05,.79,.4)]: w('Brock '+s,a,sa,b,sb)
for s,a,sa,b,sb in [('Squirtle',.91,.1,.85,.1),('Fast',.73,.3,.93,.1),('Charmander',.88,.1,.53,.4),('Bulbasaur',.26,.2,0,.05),('Choose',.45,.3,.75,.4)]: w('Cerulean '+s,a,sa,b,sb)
for s,a,sa,b,sb in [('Squirtle',.07,.1,.27,.3),('Fast',.12,.2,.33,.4),('Choose',.01,.05,.29,.4)]: w('Misty '+s,a,sa,b,sb)
