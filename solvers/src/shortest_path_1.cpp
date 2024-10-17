#include<bits/stdc++.h>
#include<math.h>

#define FOR(i,a,b) for(ll i = (a); i < (ll)(b); i++)
#define REP(i,n) FOR(i,0,n)
#define YYS(x,arr) for(auto& x:arr)

#define pb emplace_back

using namespace std;

using ll = long long int;

ll cnt;

class ShortestPath1{
    int N;
    int M;
    vector<vector<int>> G;
    vector<float> dist;
    vector<vector<int>> post;
    int next;
    int send;

    void init(int n, int m){
        N = n;
        M = m;
        G.assign(n, vector<int>(-1));
        dist.assign(n, vector<float>(INFINITY));
        post.assign(2*m, vector<int>());
        next = -1;
        send = 0;
    }

    void add_edge(int a, int b, int w){
        G[a][b] = w;
        G[b][a] = w;
    }

    void send_message(){
        mes = post[send][next].back() //宛先へ送信するメッセージ
        post[send][next].pop_back()
        if(dist[next] > mes){ //宛先で更新が起こる場合
            dist[next] = mes; //更新
            REP(i, G[next].size()){
                neighber = G[next][i] //隣接ノード
                post[next][neighbor].pb(mes+G[send][neighber]) //隣接ノードに向けてメッセージを送信
            }
        }
    }

    void decide_post(){
        mx = -1
        REP(i, post.size()){
            if(post[i].empty()){continue;} //ポストが空
            if(post[i].back() > mx){ //最も値が大きいメッセージを次に送信
                mx = post[i].back()
                next = i
            }
        }
    }

    bool check_post(){ //全ポストが空かを判定
        int count = 0;
        REP(i, post.size()){
            if(post[i].empty()){count += 1;}
        }
        if(count == post.size()){
            return true;
        }else{
            return false;
        }
    }
}

ShortestPath1 sp;
int n, m;

int main(){
    cin >> n >> m;
    sp.init(n, m);
    REP(i, m){
        int a, b, w;
        cin >> a >> b >> w;
        sp.add_edge(a,b,w)
    }
    REP(i, sp.G.size()){
        int tmp = G[0][i];
        if(tmp >= 0){
            sp.post[0].push_back(tmp);
        }
    }
    while(true){
        if(sp.check_post()){break;}
        sp.decide_post();
        sp.send_message();
        cnt += 1:
    }
    cout << cnt << endl;
    return 0;
}