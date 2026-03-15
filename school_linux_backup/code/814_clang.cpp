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
static constexpr int       RESTART_INTERVAL = 50;
static constexpr long long ITERS            = 1000;

// Stack size for has_path_fast.
// Worst case: all 112 cells match digit[0], each has 8 neighbors = 896 pushes
// at depth 0 alone. For a 5-digit number with dense matches this grows fast.
// 4096 covers all realistic cases safely.
static constexpr int PATH_STACK_SIZE = 4096;
static constexpr double MUTATE_PROB = 0.50;
std::mt19937 rng(std::random_device{}());
std::mt19937 seed_mt19937() {
    std::random_device rd{};

    // learncpp.com이 추천하는 정확한 seed_seq 구성
    std::seed_seq ss{
        static_cast<std::seed_seq::result_type>(
            std::chrono::steady_clock::now().time_since_epoch().count()
        ),
        rd(), rd(), rd(), rd(), rd(), rd(), rd()   // 7회 random_device 호출
    };

    return std::mt19937{ss};
}


// ====================== Fast I/O ======================
void fastIO() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);
}

// ====================== Grid File I/O ======================
Grid load_grid(const std::string& filename) {
    Grid g{};
    std::ifstream fin(filename);
    if (!fin) { std::cerr << "Error: cannot open '" << filename << "'\n"; std::exit(1); }
    std::string line;
    int row = 0;
    while (std::getline(fin, line) && row < 8) {
        if (line.empty()) continue;
        if ((int)line.size() < 14) {
            std::cerr << "Error: row " << row << " has fewer than 14 chars.\n"; std::exit(1);
        }
        for (int col = 0; col < 14; ++col) g[row][col] = line[col] - '0';
        ++row;
    }
    if (row != 8) { std::cerr << "Error: expected 8 rows, got " << row << ".\n"; std::exit(1); }
    return g;
}

Grid load_best_or_initial(const std::string& result_file,
                          const std::string& grid_file) {
    std::ifstream test(result_file);
    if (test.good()) {
        std::cout << "Resuming from '" << result_file << "'\n";
        return load_grid(result_file);
    }
    std::cout << "No result file found, starting from '" << grid_file << "'\n";
    return load_grid(grid_file);
}

void save_grid(const std::string& filename, const Grid& g) {
    std::ofstream fout(filename);
    if (!fout) { std::cerr << "Error: cannot write '" << filename << "'\n"; std::exit(1); }
    for (const auto& row : g) {
        for (int v : row) fout << v;
        fout << '\n';
    }
}

void print_grid(const Grid& g) {
    for (const auto& row : g) {
        for (int c = 0; c < 14; ++c) std::cout << row[c];
        std::cout << '\n';
    }
}

// ====================== Score decomposition ======================
inline int score_current(Score s)  { return (int)((-s) / FORMABLE_SCALE); }
inline int score_formable(Score s) { return (int)((-s) % FORMABLE_SCALE); }

// ====================== Pathfinding Helpers ======================
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

// ====================== Pathfinding Helpers ======================
bool has_path_fast(const Grid& grid, const int* digits, int digit_len) {
    static const int dr[] = {-1,-1,-1, 0, 0, 1, 1, 1};
    static const int dc[] = {-1, 0, 1,-1, 1,-1, 0, 1};

    // FIX 2: 스택 사이즈 제한을 없애기 위해 정적 배열 사이즈를 극단적으로 늘림 (메모리 재할당 오버헤드 방지)
    static Node stack[65536];
    int head = 0;

    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 14; ++c) {
            if (grid[r][c] == digits[0]) {
                if (digit_len == 1) return true;
                stack[head++] = {r, c, 0};
            }
        }
    }

    while (head > 0) {
        Node curr = stack[--head];
        int next = digits[curr.idx + 1];

        for (int i = 0; i < 8; ++i) {
            int nr = curr.r + dr[i], nc = curr.c + dc[i];
            if (nr >= 0 && nr < 8 && nc >= 0 && nc < 14 && grid[nr][nc] == next) {
                if (curr.idx + 1 == digit_len - 1) return true;

                // 오버플로우 방지 (65536은 어떠한 꼬임 경로에서도 뚫리지 않는 안전한 크기입니다)
                if (head < 65536) {
                    stack[head++] = {nr, nc, curr.idx + 1};
                }
            }
        }
    }
    return false;
}

// ... (기존 include 및 상수 그대로)

// ====================== Evaluation ======================
Score evaluate(const Grid& g) {
    // FIX: found를 반드시 매번 초기화 (static 제거 + 명시적 zero)
    bool found[COUNT_LIMIT]{};
    std::fill(found, found + COUNT_LIMIT, false);  // 안전하게 0으로 채움

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
            formable++; found[num] = true;
            if (rev_num < COUNT_LIMIT) found[rev_num] = true;
        }
    }

    // 점수가 낮을수록 좋음 (더 많은 숫자 = 더 작은 음수)
    return -((long long)current_score * FORMABLE_SCALE + formable);
}

// ====================== Mutations ======================
void directional_spread_mutation(Grid& g) {
    static const int dr[] = {-1,-1,-1, 0, 0, 1, 1, 1};
    static const int dc[] = {-1, 0, 1,-1, 1,-1, 0, 1};
    int num_ops = 1;
    for (int i = 0; i < num_ops; ++i) {
        int tr, tc;
        if (rng() % 2) {
            int idx = rng() % 40;          // 40, not 44
            if      (idx < 14) { tr = 0; tc = idx; }
            else if (idx < 28) { tr = 7; tc = idx - 14; }
            else if (idx < 34) { tr = idx - 28 + 1; tc = 0; }   // rows 1-6
            else               { tr = idx - 34 + 1; tc = 13; }  // rows 1-6
        } else { tr = (rng() % 6) + 1; tc = (rng() % 13) + 1; }

        if (rng() % 2) {
            std::vector<int> valid;
            for (int d = 0; d < 8; ++d) {
                int nr = tr+dr[d], nc = tc+dc[d];
                if (nr >= 0 && nr < 8 && nc >= 0 && nc < 14) valid.push_back(d);
            }
            if (!valid.empty()) {
                int d = valid[rng() % valid.size()];
                g[tr][tc] = g[tr+dr[d]][tc+dc[d]];
            }
        } else {
            g[tr][tc] = rng() % 10;
        }
    }
}

void cyclic_remapping_mutation(Grid& g) {
    int k;
    if (std::uniform_real_distribution<>(0.0,1.0)(rng) < 0.5)
        k = std::uniform_int_distribution<>(1, 3)(rng);
    else
        k = std::uniform_int_distribution<>(4, 7)(rng);

    std::vector<int> pool = {0,1,2,3,4,5,6,7,8,9};
    std::shuffle(pool.begin(), pool.end(), rng);
    std::vector<int> selected(pool.begin(), pool.begin() + k);

    std::vector<int> perm(k);
    bool use_cross = (k <= 2) || (3 <= k < 6 && std::uniform_real_distribution<>(0.0,1.0)(rng) < 0.5);

    if (use_cross) {
        std::vector<bool> in_sel(10, false);
        for (int v : selected) in_sel[v] = true;
        std::vector<int> remaining;
        for (int i = 0; i < 10; ++i) if (!in_sel[i]) remaining.push_back(i);
        std::shuffle(remaining.begin(), remaining.end(), rng);
        for (int i = 0; i < k; ++i) perm[i] = remaining[i];
    } else {
        int shift = std::uniform_int_distribution<>(1, k-1)(rng);
        for (int i = 0; i < k; ++i) perm[i] = selected[(i + shift) % k];
    }

    int remap[10];
    for (int i = 0; i < 10; ++i) remap[i] = i;
    for (int i = 0; i < k; ++i) remap[selected[i]] = perm[i];
    for (auto& row : g) for (int& v : row) v = remap[v];
}

// ====================== Progress bar ======================
void print_progress(int restart, int epoch, Score best) {
    constexpr int BAR_WIDTH = 30;
    double pct    = (double)epoch / RESTART_INTERVAL;
    int    filled = (int)(pct * BAR_WIDTH);

    std::cout << '\r'
              << "restart=" << std::setw(5) << restart << "  "
              << '[';
    for (int i = 0; i < BAR_WIDTH; ++i) std::cout << (i < filled ? '#' : '-');
    std::cout << "] "
              << "epoch=" << std::setw(4) << epoch << '/' << RESTART_INTERVAL << "  "
              << "score=" << std::setw(6) << score_current(best)
              << "  formable=" << std::setw(6) << score_formable(best)
              << "   ";
    std::cout.flush();
}

// ====================== Main ======================
int main() {
    fastIO();

    const std::string grid_file   = "../data/grid.txt";
    const std::string result_file = "../data/result.txt";

    Grid initial_grid = load_best_or_initial(result_file, grid_file);

    std::cout << "Loaded grid:\n";
    print_grid(initial_grid);

    Score ans       = evaluate(initial_grid);
    Grid  best_grid = initial_grid;

    std::cout << "\nCOUNT_LIMIT      : " << COUNT_LIMIT
              << "\nRESTART_INTERVAL : " << RESTART_INTERVAL
              << "\nITERS            : " << ITERS
              << "\nInitial score    : " << score_current(ans)
              << "\nInitial formable : " << score_formable(ans)
              << "\n\n";



    auto mutate_fn = [&](Grid& g) {
        if (std::uniform_real_distribution<double>(0.0, 1.0)(rng) >= MUTATE_PROB) {
            return;   
        }

        if (rng() % 2 == 0) {
            directional_spread_mutation(g);
        } else {
            cyclic_remapping_mutation(g);
        }
    };

    int total_epoch = 0;

    for (int restart = 1; restart < 2; ++restart) {
        rng = seed_mt19937();

        Grid  restart_grid = best_grid;
        Score restart_best = ans;

        for (int epoch = 1; epoch <= RESTART_INTERVAL; ++epoch) {
            ++total_epoch;

            auto [board, res] = dlas<Grid, Score>(evaluate, mutate_fn, restart_grid, ITERS);

            print_progress(restart, epoch, ans);

            if (res < ans) {
                ans       = res;
                best_grid = board;

                std::cout << "\n[total_epoch " << std::setw(7) << total_epoch
                          << ", restart "      << std::setw(5) << restart
                          << ", epoch "        << std::setw(4) << epoch
                          << "] New best!"
                          << "  score="    << std::setw(6) << score_current(ans)
                          << ", formable=" << std::setw(6) << score_formable(ans) << '\n';
                print_grid(best_grid);

                save_grid(result_file, best_grid);
                std::cout << "Saved to '" << result_file << "'\n";
                std::cout.flush();
            }

            if (res < restart_best) {
                restart_best = res;
                restart_grid = board;
            }
        }

        std::cout << "\n[restart " << std::setw(5) << restart
                  << " done, total_epoch=" << std::setw(7) << total_epoch
                  << "]  best score=" << std::setw(6) << score_current(ans)
                  << ", formable="    << std::setw(6) << score_formable(ans)
                  << "\n";
        std::cout.flush();
    }

    return 0;
}

