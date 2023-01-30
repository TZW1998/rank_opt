import numpy as np
import matplotlib.pyplot as plt

dim=10
m=10
x = np.random.rand(dim,1) + 2
x = x/np.linalg.norm(x)
reps = 100000
re_x = []
topk=10
for _ in range(reps):
    Z = np.random.randn(m,dim)
    Z /= np.linalg.norm(Z,axis=1).reshape(m,-1)
    Z = np.concatenate([Z,np.zeros((1,dim))],0)
    ry = np.matmul(Z,x).flatten()
    y_rank = np.argsort(ry)[:topk]
    
    
    Zy = []
    y = []
    for ii, x0 in enumerate(y_rank[:-1]):
            for jj, x1 in enumerate(y_rank[(ii+1):]):
                now_z = Z[x1]-Z[x0]
                Zy.append(now_z)
                y.append(1)
                
    if len(Z) > len(y_rank):
        for x0, qx0 in enumerate(Z):
            if x0 not in y_rank:
                for top_x0 in y_rank:
                    now_z = qx0 - Z[top_x0]
                    Zy.append(now_z)
                    y.append(1)
    
    y=np.array(y)
    Zy=np.array(Zy)
    
    now_rex = np.matmul(y.reshape(1,-1),Zy).flatten()
    re_x.append(now_rex/np.linalg.norm(now_rex))
    #re_x.append(Solve1BitCS(y,Z,m,dim,10000))
#print(np.linalg.norm(x.flatten()-re_x/reps))

mean_re_x = np.zeros(dim)
results_rank = []
for n in range(1,reps+1):
    mean_re_x += re_x[n-1] 
    if n in [1,10,100,1000,10000,100000]:
        test_x = mean_re_x/n
        test_x /= np.linalg.norm(test_x)
        results_rank.append(np.linalg.norm(x.flatten()-test_x))
        
#plt.plot(results,label="w/o rank")
plt.plot(results_rank,label="top-{}".format(topk))
plt.plot(np.zeros_like(results_rank))
plt.title("m=10")
plt.xlabel("n")
plt.legend()
plt.ylabel(r"$||x-\sum_{i=1}^n \frac{\hat x_i}{n}||$")
plt.xticks([0,1,2,3,4,5],[1,10,100,1000,10000,100000])