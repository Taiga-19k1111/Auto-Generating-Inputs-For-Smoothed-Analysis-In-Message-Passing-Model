import numpy as np

def gen_worstcase(n):
    G = np.ones([n,n])*-1
    for i in range(int((2*n+1)/3)):
        for j in range(i+1,n):
            if i%2 == 0:
                if j == i+1 and i != ((2*n+1)/3)-1:
                    G[i][j] = 2**((n-4-3*(i/2))/3)
                elif j == ((i/2)+(2*n+1)/3):
                    G[i][j] = 0
                elif j == ((i/2)+(2*n+1)/3)-1 and i != 0:
                    G[i][j] = 0
            else:
                if j == i+1:
                    G[i][j] = 0
            G[j][i] = G[i][j]
    return G

print(gen_worstcase(13))