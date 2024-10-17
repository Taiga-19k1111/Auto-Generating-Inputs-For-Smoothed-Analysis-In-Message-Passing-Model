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

ll cnt;

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
void QuickSort(vector<int> &v, int left, int right) {
    if (left >= right) return;
    int pivot = v[left];
    int i = left;
    for (int j = left+1; j < right+1; j++){
        cnt++;
        if (v[j] <= pivot){
            i++;
            swap(v[i],v[j]);
        }
    }
    swap(v[i],v[left]);
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
    cout << cnt << endl;
    return 0;
}
