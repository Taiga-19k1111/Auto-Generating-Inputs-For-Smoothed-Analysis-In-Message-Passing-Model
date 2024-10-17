#include <bits/stdc++.h>

#define REP(i,n) FOR(i,0,n)

using namespace std;

int main(){

    cin >> n >> m;

    G.assign(n, vector<int>(0));
    REP(i, m){
        int a, b;
        cin >> a >> b;
        G[a].pb(b);
        G[b].pb(a);
    }


}