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

def decide_message(n, post):
    max_message = -1
    ind = -1
    for i in range(n*n):
        if post[i] != []:
            message = post[i][0]
            if message > max_message:
                max_message = message
                ind = i
    return ind

def update_state(n, G, dist, post):
    ind = decide_message(n, post)
    message = post[ind].pop(0)
    receive = ind%n
    if dist[receive] > message:
        dist[receive] = message
        for i in range(n):
            if G[receive][i] != -1:
                post[receive*n+i].append(dist[receive]+G[receive][i])
    return post, dist

N = 25
G = gen_worstcase(N)
post, dist = gen_initial_graph_state(G,N)
check = True
count = 0
while check:
    post, dist = update_state(N, G, dist, post)
    count += 1
    check = False
    for p in post:
        if p != []:
            check = True
            break

print(count)