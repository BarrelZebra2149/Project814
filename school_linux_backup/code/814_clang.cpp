#include <iostream>
#include <fstream>
#include <string>
#include <array>
#include <random>
#include <chrono>
#include <vector>
#include <algorithm>
#include <iomanip>
#include "dlas.hpp"

using Grid  = std::array<std::array<int, 14>, 8>;
using Score = long long;

static constexpr int       COUNT_LIMIT      = 10000;
static constexpr long long FORMABLE_SCALE   = COUNT_LIMIT;
static constexpr int       RESTART_INTERVAL = 100;
static constexpr long long ITERS            = 1000;
static constexpr int       PATH_STACK_SIZE  = 65536;
static constexpr double    MUTATE_PROB      = 0.50;

std::mt19937 rng;   

std::mt19937 seed_mt19937() {
    std::random_device rd{};
    std::seed_seq ss{
        static_cast<std::seed_seq::result_type>(
            std::chrono::steady_clock::now().time_since_epoch().count()
        ),
        rd(), rd(), rd(), rd(), rd(), rd(), rd()
    };
    return std::mt19937{ss};
}

// ====================== Grid File I/O ======================
Grid load_grid(const std::string& filename) {
    Grid g{};
    std::ifstream fin(filename);
    if (!fin) std::exit(1);
    std::string line;
    int row = 0;
    while (std::getline(fin, line) && row < 8) {
        if (line.empty()) continue;
        for (int col = 0; col < 14; ++col) g[row][col] = line[col] - '0';
        ++row;
    }
    return g;
}


void save_grid(const std::string& filename, const Grid& g) {
    std::ofstream fout(filename);
    if (!fout) std::exit(1);
    for (const auto& row : g) {
        for (int v : row) fout << v;
        fout << '\n';
    }
}

// ====================== Score helpers ======================
inline int score_current(Score s) { return (int)((-s) / FORMABLE_SCALE); }

// ====================== Pathfinding ======================
struct Node { int r, c, idx; };

inline int reverse_int(int n) {
    int rev = 0;
    while (n > 0) { rev = rev * 10 + (n % 10); n /= 10; }
    return rev;
}

inline int get_digits(int n, int* digits) {
    int temp[7], len = 0;
    while (n > 0) { temp[len++] = n % 10; n /= 10; }
    for (int i = 0; i < len; ++i) digits[i] = temp[len - 1 - i];
    return len;
}

bool has_path_fast(const Grid& grid, const int* digits, int digit_len) {
    static const int dr[] = {-1,-1,-1,0,0,1,1,1};
    static const int dc[] = {-1,0,1,-1,1,-1,0,1};
    static Node stack[PATH_STACK_SIZE];
    int head = 0;

    for (int r = 0; r < 8; ++r)
        for (int c = 0; c < 14; ++c)
            if (grid[r][c] == digits[0]) {
                if (digit_len == 1) return true;
                stack[head++] = {r, c, 0};
            }

    while (head > 0) {
        Node curr = stack[--head];
        int next = digits[curr.idx + 1];
        for (int i = 0; i < 8; ++i) {
            int nr = curr.r + dr[i], nc = curr.c + dc[i];
            if (nr >= 0 && nr < 8 && nc >= 0 && nc < 14 && grid[nr][nc] == next) {
                if (curr.idx + 1 == digit_len - 1) return true;
                if (head < PATH_STACK_SIZE)
                    stack[head++] = {nr, nc, curr.idx + 1};
            }
        }
    }
    return false;
}

// ====================== Evaluation ======================
Score evaluate(const Grid& g) {
    bool found[COUNT_LIMIT]{};
    int current_score = 0;
    int digits[7];
    bool all_formable = true;

    for (int n = 1; n < COUNT_LIMIT; ++n) {
        int rev_n = reverse_int(n);
        if (found[n] || (n % 10 != 0 && rev_n < COUNT_LIMIT && found[rev_n])) continue;

        int len = get_digits(n, digits);
        if (has_path_fast(g, digits, len)) {
            found[n] = true;
            if (rev_n < COUNT_LIMIT) found[rev_n] = true;
        } else {
            current_score = n - 1;
            all_formable = false;
            break;
        }
    }
    if (all_formable) current_score = COUNT_LIMIT - 1;

    int formable = std::max(0, std::min(COUNT_LIMIT, current_score) - 1000 + 1);
    for (int num = std::max(1000, current_score + 1); num < COUNT_LIMIT; ++num) {
        int rev_num = reverse_int(num);
        if (found[num] || (num % 10 != 0 && rev_num < COUNT_LIMIT && found[rev_num])) {
            formable++; continue;
        }
        int len = get_digits(num, digits);
        if (has_path_fast(g, digits, len)) {
            formable++;
            found[num] = true;
            if (rev_num < COUNT_LIMIT) found[rev_num] = true;
        }
    }
    return -((long long)current_score * FORMABLE_SCALE + formable);
}

// ====================== Mutations ======================
void directional_spread_mutation(Grid& g) {
    static const int dr[] = {-1,-1,-1,0,0,1,1,1};
    static const int dc[] = {-1,0,1,-1,1,-1,0,1};
    int num_ops = 1;
    for (int i = 0; i < num_ops; ++i) {
        int tr, tc;
        if (rng() % 2) {
            int idx = rng() % 40;
            if (idx < 14) { tr = 0; tc = idx; }
            else if (idx < 28) { tr = 7; tc = idx - 14; }
            else if (idx < 34) { tr = idx - 28 + 1; tc = 0; }
            else { tr = idx - 34 + 1; tc = 13; }
        } else {
            tr = (rng() % 6) + 1; tc = (rng() % 13) + 1;
        }
        if (rng() % 2) {
            std::vector<int> valid;
            for (int d = 0; d < 8; ++d) {
                int nr = tr + dr[d], nc = tc + dc[d];
                if (nr >= 0 && nr < 8 && nc >= 0 && nc < 14) valid.push_back(d);
            }
            if (!valid.empty()) {
                int d = valid[rng() % valid.size()];
                g[tr][tc] = g[tr + dr[d]][tc + dc[d]];
            }
        } else {
            g[tr][tc] = rng() % 10;
        }
    }
}

void cyclic_remapping_mutation(Grid& g) {
    int k = (rng() % 2 == 0) ? (rng() % 3 + 1) : (rng() % 4 + 4);
    std::vector<int> pool = {0,1,2,3,4,5,6,7,8,9};
    std::shuffle(pool.begin(), pool.end(), rng);
    std::vector<int> selected(pool.begin(), pool.begin() + k);
    std::vector<int> perm(k);
    bool use_cross = (k <= 2) || (3 <= k && k < 6 && rng() % 2);
    if (use_cross) {
        std::vector<int> remaining;
        for (int i = 0; i < 10; ++i) {
            bool used = false;
            for (int s : selected) if (s == i) { used = true; break; }
            if (!used) remaining.push_back(i);
        }
        std::shuffle(remaining.begin(), remaining.end(), rng);
        for (int i = 0; i < k; ++i) perm[i] = remaining[i];
    } else {
        int shift = rng() % (k - 1) + 1;
        for (int i = 0; i < k; ++i) perm[i] = selected[(i + shift) % k];
    }
    int remap[10];
    for (int i = 0; i < 10; ++i) remap[i] = i;
    for (int i = 0; i < k; ++i) remap[selected[i]] = perm[i];
    for (auto& row : g) for (int& v : row) v = remap[v];
}

void adjacent_swap_mutation(Grid& g) {
    static const int dr[] = {-1,-1,-1,0,0,1,1,1};
    static const int dc[] = {-1,0,1,-1,1,-1,0,1};
    int tr = rng() % 8;
    int tc = rng() % 14;
    std::vector<std::pair<int,int>> valid;
    for (int i = 0; i < 8; ++i) {
        int nr = tr + dr[i], nc = tc + dc[i];
        if (nr >= 0 && nr < 8 && nc >= 0 && nc < 14) valid.emplace_back(nr, nc);
    }
    if (!valid.empty()) {
        auto [nr, nc] = valid[rng() % valid.size()];
        std::swap(g[tr][tc], g[nr][nc]);
    }
}

// ====================== MAIN ======================
int main() {
    const std::string grid_file   = "../data/grid.txt";
    const std::string result_file = "../data/result.txt";

    rng = seed_mt19937();                     
    Grid best_grid = load_grid(grid_file);
    Score ans      = evaluate(best_grid);

    constexpr int MAX_STAGNANT_RESTARTS = 1;
    int stagnation_restarts = 0;

    auto mutate_fn = [&](Grid& g) {
        if (std::uniform_real_distribution<double>(0.0, 1.0)(rng) >= MUTATE_PROB) return;
        double r = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
        if (r < 0.33)      directional_spread_mutation(g);
        else if (r < 0.66) cyclic_remapping_mutation(g);
        else               adjacent_swap_mutation(g);
    };

    while (true) {
        rng = seed_mt19937();                 
        Grid restart_grid = best_grid;
        Score restart_best = ans;
        bool improved_this_restart = false;

        for (int epoch = 1; epoch <= RESTART_INTERVAL; ++epoch) {
            auto [board, res] = dlas<Grid, Score>(evaluate, mutate_fn, restart_grid, ITERS);

            if (res < ans) {
                ans = res;
                best_grid = board;
                improved_this_restart = true;
                stagnation_restarts = 0;
                save_grid(result_file, best_grid);
            }

            if (res < restart_best) {
                restart_best = res;
                restart_grid = board;
            }
        }

        if (!improved_this_restart) {
            stagnation_restarts++;
        } else {
            stagnation_restarts = 0;
        }

        if (stagnation_restarts >= MAX_STAGNANT_RESTARTS) {
            break;
        }
    }

    return 0;
}