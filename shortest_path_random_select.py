import numpy as np

from gen_SSSP_worstcase import gen_worstcase

def gen_initial_graph_state(G, n):
    post = [[] for _ in range(n*n)]
    dist = [10**6 for _ in range(n)]
    start_node = 0
    for i in range(n):
        if G[start_node][i] != -1:
            post[start_node*n+i].append(int(G[start_node][i]))
    dist[start_node] = 0
    return post, dist

def decide_max_message(n, post):
    max_message = -1
    ind = -1
    for i in range(n*n):
        if post[i] != []:
            message = post[i][0]
            if message > max_message:
                max_message = message
                ind = i
    return ind

def decide_message(n, post):
    candidate = []
    for i in range(n*n):
        if post[i] != []:
            candidate.append(i)
    ind = np.random.choice(np.array(candidate))
    return ind

def update_state(n, G, dist, post, DM):
    ind = DM(n, post)
    message = post[ind].pop(0)
    receive = ind%n
    if dist[receive] > message:
        dist[receive] = message
        for i in range(n):
            if G[receive][i] != -1:
                post[receive*n+i].append(dist[receive]+G[receive][i])
    return post, dist

max_count = -1
counts = []
rep = 5000
N =  25
np.random.seed(1234)
epsilon = 1
# print(epsilon)
for _ in range(rep):
    G = gen_worstcase(N)
    post, dist = gen_initial_graph_state(G,N)
    check = True
    count = 0
    while check:
        rnd = np.random.uniform(0,1,1)
        if epsilon < rnd:
            DM = decide_max_message
        else:
            DM = decide_message
        post, dist = update_state(N, G, dist, post, DM)
        count += 1
        check = False
        for p in post:
            if p != []:
                check = True
                break
    if count > max_count:
        max_count = count
    counts.append(count)
print(sum(counts)/rep)
print(max_count)