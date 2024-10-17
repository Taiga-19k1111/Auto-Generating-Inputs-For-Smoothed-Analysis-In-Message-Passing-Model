#include <iostream>
#include <random>
#include <vector>
#include <unistd.h>
#include <chrono>
#include <algorithm>
using namespace std;
using ll = long long int;

#define rep(i, n) for (int i = 0; i < n; i++)
#define pb emplace_back

ll cnt = 0;

unsigned seed = chrono::system_clock::now().time_since_epoch().count();
default_random_engine gen(seed);

void display(const vector<int> &v) {
    rep(i, v.size()) cout << v[i] << " ";
    cout << endl;
}
// 昇順チェッカー
void ascendingOrderChecker(const vector<int> &v) {
    bool flag = true;
    rep(i, v.size() - 1) flag &= v[i] <= v[i + 1];
    if (flag) {
        cout << "Correctly sorted." << endl;
    } else
        cout << "Not sorted correctly." << endl;
}
void swap(int *a, int *b) {
    int *t = a;
    a = b;
    b = t;
}

int random(int low, int high){
    uniform_int_distribution<> dist(low, high);
    return dist(gen);
}


void QuickSort(vector<int> &v, int left, int right) {
    if (left >= right) return;
    int pi = random(left,right);
    cout << pi << endl;
    int pivot = v[pi];
    int i = left-1;
    for (int j = left; j < right+1; j++){
        if (j == pi){
            continue;
        }
        cnt++;
        if (v[j] <= pivot){
            i++;
            if (i == pi){
                i++;
            }
            swap(v[i],v[j]);
        }
    }
    if (i < pi){
        i++;
    }
    swap(v[i],v[pi]);
    QuickSort(v, left, i-1);
    QuickSort(v, i+1, right);
}

int n,m;

int main() {
    cin >> n >> m;
    vector<int> S(n);
    rep(i, n) {
        int a;
        cin >> a;
        S[i] = a;
    }
    QuickSort(S, 0, S.size()-1);
    display(S);
    cout << cnt << endl;
    sleep(10);
    return 0;
}
