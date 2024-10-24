#include <iostream>
#include <random>
#include <vector>
#include <unistd.h>
#include <chrono>
#include <algorithm>
#include <bits/stdc++.h>
#include <fstream>
      
#define FOR(i,a,b) for(ll i = (a); i < (ll)(b); i++)
#define REP(i,n) FOR(i,0,n)
#define YYS(x,arr) for(auto& x:arr)

#define pb emplace_back

using namespace std;

using ll = long long int;

int n;
int send,receive;
vector<vector<float>> G;
vector<vector<float>> post;
vector<float> dist;

int main(){
    // ifstream in("input.txt");
    // cin.rdbuf(in.rdbuf());

    cin >> n >> send >> receive;
    G.assign(n, vector<float>(n));
    post.assign(n, vector<float>(n));
    dist.assign(n,0);
    REP(i, n){
        REP(j, n){
            cin >> G[i][j];
        }
    }
    REP(i, n){
        REP(j, n){
            cin >> post[i][j];
        }
    }
    REP(i, n){
        int a;
        cin >> dist[i];
    }

    int m = post[send][receive];
    if(m < dist[receive]){
        cout << receive << endl;
        REP(i, n){
            cout << i << " " << m+G[receive][i] << endl;
        }
    }
    return 0;
}