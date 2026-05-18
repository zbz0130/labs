#include <iostream>
#include <vector>
#include <algorithm>
#include <climits>
#include <cstring>
#include <unordered_map>
#include <random>
#include "Point.h"
#include "Strategy.h"

using namespace std;

namespace {
constexpr int MAX_BOARD = 15;
constexpr int PLAYER_USER = 1;
constexpr int PLAYER_MACHINE = 2;
constexpr int WIN_SCORE = 1000000;
constexpr int SEARCH_DEPTH = 6;
constexpr unsigned long long SIDE_TO_MOVE_KEY = 0x9e3779b97f4a7c15ULL;

enum BoundType {
	BOUND_EXACT = 0,
	BOUND_LOWER = 1,
	BOUND_UPPER = 2
};

struct TTEntry {
	int depth;
	int score;
	int bestCol;
	BoundType bound;
};
}  // namespace

unsigned long long zobrist[MAX_BOARD][MAX_BOARD][3];
unordered_map<unsigned long long, TTEntry> TT;
mt19937_64 rng(114514);

bool checkWin(int** board, int M, int N, int x, int y, int player);
int evaluateWindow(int a, int b, int c, int d);
int evaluateBoard(int** board, int M, int N);
int evaluate4(int** board, int x, int y, int dx, int dy);
int getCenterDelta(int col, int piece, int N);
int getScoreDeltaForMove(int** board, int M, int N, int row, int col, int piece);
int minimax(
	int** board,
	int M,
	int N,
	int* top,
	int depth,
	int alpha,
	int beta,
	bool maximizingPlayer,
	unsigned long long hash,
	int noX,
	int noY,
	int currentScore
);
void initZobrist();
const std::vector<int>& getMoveOrder(int N);

unsigned long long getHash(int** board, int M, int N) {
	unsigned long long h = 0;
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			int p = board[i][j];
			if (p) {
				h ^= zobrist[i][j][p];
			}
		}
	}
	return h;
}

unsigned long long getTTKey(unsigned long long hash, bool maximizingPlayer) {
	return maximizingPlayer ? (hash ^ SIDE_TO_MOVE_KEY) : hash;
}

int getCenterDelta(int col, int piece, int N) {
	if (col != N / 2) {
		return 0;
	}
	if (piece == PLAYER_MACHINE) {
		return 6;
	}
	if (piece == PLAYER_USER) {
		return -6;
	}
	return 0;
}

int getScoreDeltaForMove(int** board, int M, int N, int row, int col, int piece) {
	int before = 0;
	int after = 0;

	for (int startY = col - 3; startY <= col; startY++) {
		if (startY >= 0 && startY + 3 < N) {
			before += evaluate4(board, row, startY, 0, 1);
		}
	}
	for (int startX = row - 3; startX <= row; startX++) {
		if (startX >= 0 && startX + 3 < M) {
			before += evaluate4(board, startX, col, 1, 0);
		}
	}
	for (int offset = 0; offset < 4; offset++) {
		int startX = row - offset;
		int startY = col - offset;
		if (startX >= 0 && startY >= 0 && startX + 3 < M && startY + 3 < N) {
			before += evaluate4(board, startX, startY, 1, 1);
		}
	}
	for (int offset = 0; offset < 4; offset++) {
		int startX = row - offset;
		int startY = col + offset;
		if (startX >= 0 && startY < N && startX + 3 < M && startY - 3 >= 0) {
			before += evaluate4(board, startX, startY, 1, -1);
		}
	}

	board[row][col] = piece;

	for (int startY = col - 3; startY <= col; startY++) {
		if (startY >= 0 && startY + 3 < N) {
			after += evaluate4(board, row, startY, 0, 1);
		}
	}
	for (int startX = row - 3; startX <= row; startX++) {
		if (startX >= 0 && startX + 3 < M) {
			after += evaluate4(board, startX, col, 1, 0);
		}
	}
	for (int offset = 0; offset < 4; offset++) {
		int startX = row - offset;
		int startY = col - offset;
		if (startX >= 0 && startY >= 0 && startX + 3 < M && startY + 3 < N) {
			after += evaluate4(board, startX, startY, 1, 1);
		}
	}
	for (int offset = 0; offset < 4; offset++) {
		int startX = row - offset;
		int startY = col + offset;
		if (startX >= 0 && startY < N && startX + 3 < M && startY - 3 >= 0) {
			after += evaluate4(board, startX, startY, 1, -1);
		}
	}

	return after - before + getCenterDelta(col, piece, N);
}

int playMove(int** board, int M, int N, int* top, int col, int player, int noX, int noY, int* scoreDelta = nullptr) {
	int row = top[col] - 1;
	if (scoreDelta != nullptr) {
		*scoreDelta = getScoreDeltaForMove(board, M, N, row, col, player);
	} else {
		board[row][col] = player;
	}
	top[col]--;
	if (col == noY && top[col] - 1 == noX) {
		top[col]--;
	}
	return row;
}

extern "C" __declspec(dllexport) Point* getPoint(const int M, const int N, const int* top, const int* _board,
	const int lastX, const int lastY, const int noX, const int noY) {
	/*
		不要更改这段代码
	*/
	(void)lastX;
	(void)lastY;
	int mytop[MAX_BOARD] = {};
	memcpy(mytop, top, sizeof(int) * N);
	int x = -1, y = -1;
	int** board = new int*[M];
	for (int i = 0; i < M; i++) {
		board[i] = new int[N];
		for (int j = 0; j < N; j++) {
			board[i][j] = _board[i * N + j];
		}
	}

	static bool initialized = false;
	if (!initialized) {
		initZobrist();
		initialized = true;
	}
	if (TT.size() > 300000) {
		TT.clear();
	}
	unsigned long long hash = getHash(board, M, N);

	// 自己能否直接赢
	for (int col = 0; col < N; col++) {
		if (mytop[col] <= 0) {
			continue;
		}
		int prevTop = mytop[col];
		int row = playMove(board, M, N, mytop, col, PLAYER_MACHINE, noX, noY);
		if (checkWin(board, M, N, row, col, PLAYER_MACHINE)) {
			x = row;
			y = col;
			board[row][col] = 0;
			mytop[col] = prevTop;
			break;
		}
		board[row][col] = 0;
		mytop[col] = prevTop;
	}

	// 挡住对面必胜
	if (x == -1) {
		for (int col = 0; col < N; col++) {
			if (mytop[col] <= 0) {
				continue;
			}
			int prevTop = mytop[col];
			int row = playMove(board, M, N, mytop, col, PLAYER_USER, noX, noY);
			if (checkWin(board, M, N, row, col, PLAYER_USER)) {
				x = row;
				y = col;
				board[row][col] = 0;
				mytop[col] = prevTop;
				break;
			}
			board[row][col] = 0;
			mytop[col] = prevTop;
		}
	}

	if (x != -1) {
		clearArray(M, N, board);
		return new Point(x, y);
	}
	int currentScore = evaluateBoard(board, M, N);
	int bestScore = INT_MIN;
	int bestCol = -1;
	const std::vector<int>& order = getMoveOrder(N);

	for (int col : order) {
		if (mytop[col] <= 0) {
			continue;
		}
		int prevTop = mytop[col];
		int scoreDelta = 0;
		int row = playMove(board, M, N, mytop, col, PLAYER_MACHINE, noX, noY, &scoreDelta);
		unsigned long long childHash = hash ^ zobrist[row][col][PLAYER_MACHINE];
		int score;
		if (checkWin(board, M, N, row, col, PLAYER_MACHINE)) {
			score = WIN_SCORE;
		} else {
			score = minimax(board, M, N, mytop, SEARCH_DEPTH - 1, INT_MIN, INT_MAX, false,
				childHash, noX, noY, currentScore + scoreDelta);
		}
		board[row][col] = 0;
		mytop[col] = prevTop;
		if (score > bestScore) {
			bestScore = score;
			bestCol = col;
		}
	}

	if (bestCol == -1) {
		for (int col = 0; col < N; col++) {
			if (top[col] > 0) {
				bestCol = col;
				break;
			}
		}
	}

	y = bestCol;
	x = top[bestCol] - 1;

	/*
		不要更改这段代码
	*/
	clearArray(M, N, board);
	return new Point(x, y);
}

extern "C" __declspec(dllexport) void clearPoint(Point* p) {
	delete p;
	return;
}

void clearArray(int M, int N, int** board) {
	(void)N;
	for (int i = 0; i < M; i++) {
		delete[] board[i];
	}
	delete[] board;
}

bool checkWin(int** board, int M, int N, int x, int y, int player) {
	int dx[4] = {1, 0, 1, 1};
	int dy[4] = {0, 1, 1, -1};

	for (int dir = 0; dir < 4; dir++) {
		int count = 1;

		for (int step = 1; step < 4; step++) {
			int nx = x + dx[dir] * step;
			int ny = y + dy[dir] * step;
			if (nx < 0 || nx >= M || ny < 0 || ny >= N) {
				break;
			}
			if (board[nx][ny] == player) {
				count++;
			} else {
				break;
			}
		}

		for (int step = 1; step < 4; step++) {
			int nx = x - dx[dir] * step;
			int ny = y - dy[dir] * step;
			if (nx < 0 || nx >= M || ny < 0 || ny >= N) {
				break;
			}
			if (board[nx][ny] == player) {
				count++;
			} else {
				break;
			}
		}

		if (count >= 4) {
			return true;
		}
	}

	return false;
}

int evaluateWindow(int a, int b, int c, int d) {
	int self = 0;
	int opp = 0;
	int empty = 0;
	int cells[4] = {a, b, c, d};

	for (int x : cells) {
		if (x == PLAYER_MACHINE) {
			self++;
		} else if (x == PLAYER_USER) {
			opp++;
		} else {
			empty++;
		}
	}

	if (self > 0 && opp > 0) {
		return 0;
	}

	int score = 0;
	if (self == 4) {
		score += 100000;
	} else if (self == 3 && empty == 1) {
		score += 5000;
	} else if (self == 2 && empty == 2) {
		score += 500;
	} else if (self == 1 && empty == 3) {
		score += 10;
	}

	if (opp == 4) {
		score -= 100000;
	} else if (opp == 3 && empty == 1) {
		score -= 8000;
	} else if (opp == 2 && empty == 2) {
		score -= 600;
	} else if (opp == 1 && empty == 3) {
		score -= 10;
	}

	return score;
}

int evaluateBoard(int** board, int M, int N) {
	int score = 0;
	int center = N / 2;

	for (int i = 0; i < M; i++) {
		if (board[i][center] == PLAYER_MACHINE) {
			score += 6;
		} else if (board[i][center] == PLAYER_USER) {
			score -= 6;
		}
	}

	for (int i = 0; i < M; i++) {
		for (int j = 0; j + 3 < N; j++) {
			score += evaluateWindow(board[i][j], board[i][j + 1], board[i][j + 2], board[i][j + 3]);
		}
	}

	for (int i = 0; i + 3 < M; i++) {
		for (int j = 0; j < N; j++) {
			score += evaluateWindow(board[i][j], board[i + 1][j], board[i + 2][j], board[i + 3][j]);
		}
	}

	for (int i = 0; i + 3 < M; i++) {
		for (int j = 0; j + 3 < N; j++) {
			score += evaluateWindow(board[i][j], board[i + 1][j + 1], board[i + 2][j + 2], board[i + 3][j + 3]);
		}
	}

	for (int i = 0; i + 3 < M; i++) {
		for (int j = 3; j < N; j++) {
			score += evaluateWindow(board[i][j], board[i + 1][j - 1], board[i + 2][j - 2], board[i + 3][j - 3]);
		}
	}

	return score;
}

int minimax(int** board, int M, int N, int* top, int depth, int alpha, int beta, bool maximizingPlayer,
	unsigned long long hash, int noX, int noY, int currentScore) {
	const int alphaOrig = alpha;
	const int betaOrig = beta;
	const unsigned long long ttKey = getTTKey(hash, maximizingPlayer);
	int ttBestCol = -1;

	auto it = TT.find(ttKey);
	if (it != TT.end()) {
		ttBestCol = it->second.bestCol;
		if (it->second.depth >= depth) {
			if (it->second.bound == BOUND_EXACT) {
				return it->second.score;
			}
			if (it->second.bound == BOUND_LOWER) {
				alpha = std::max(alpha, it->second.score);
			} else if (it->second.bound == BOUND_UPPER) {
				beta = std::min(beta, it->second.score);
			}
			if (alpha >= beta) {
				return it->second.score;
			}
		}
	}

	if (depth == 0) {
		return currentScore;
	}

	bool hasMove = false;
	for (int col = 0; col < N; col++) {
		if (top[col] > 0) {
			hasMove = true;
			break;
		}
	}
	if (!hasMove) {
		return 0;
	}

	const std::vector<int>& order = getMoveOrder(N);
	auto storeTT = [&](int score, int bestCol, BoundType bound) {
		TT[ttKey] = TTEntry{depth, score, bestCol, bound};
		return score;
	};

	if (maximizingPlayer) {
		int value = INT_MIN;
		int bestCol = -1;

		auto exploreMove = [&](int col) -> bool {
			if (top[col] <= 0) {
				return false;
			}
			int prevTop = top[col];
			int scoreDelta = 0;
			int row = playMove(board, M, N, top, col, PLAYER_MACHINE, noX, noY, &scoreDelta);
			unsigned long long childHash = hash ^ zobrist[row][col][PLAYER_MACHINE];
			int score;
			if (checkWin(board, M, N, row, col, PLAYER_MACHINE)) {
				score = WIN_SCORE + depth;
			} else {
				score = minimax(board, M, N, top, depth - 1, alpha, beta, false,
					childHash, noX, noY, currentScore + scoreDelta);
			}
			board[row][col] = 0;
			top[col] = prevTop;

			if (score > value) {
				value = score;
				bestCol = col;
			}
			alpha = std::max(alpha, value);
			return alpha >= beta;
		};

		bool cutoff = false;
		if (ttBestCol >= 0 && ttBestCol < N) {
			cutoff = exploreMove(ttBestCol);
		}
		if (!cutoff) {
			for (int col : order) {
				if (col == ttBestCol) {
					continue;
				}
				if (exploreMove(col)) {
					break;
				}
			}
		}

		BoundType bound = BOUND_EXACT;
		if (value <= alphaOrig) {
			bound = BOUND_UPPER;
		} else if (value >= betaOrig) {
			bound = BOUND_LOWER;
		}
		return storeTT(value, bestCol, bound);
	}

	int value = INT_MAX;
	int bestCol = -1;

	auto exploreMove = [&](int col) -> bool {
		if (top[col] <= 0) {
			return false;
		}
		int prevTop = top[col];
		int scoreDelta = 0;
		int row = playMove(board, M, N, top, col, PLAYER_USER, noX, noY, &scoreDelta);
		unsigned long long childHash = hash ^ zobrist[row][col][PLAYER_USER];
		int score;
		if (checkWin(board, M, N, row, col, PLAYER_USER)) {
			score = -WIN_SCORE - depth;
		} else {
			score = minimax(board, M, N, top, depth - 1, alpha, beta, true,
				childHash, noX, noY, currentScore + scoreDelta);
		}
		board[row][col] = 0;
		top[col] = prevTop;

		if (score < value) {
			value = score;
			bestCol = col;
		}
		beta = std::min(beta, value);
		return alpha >= beta;
	};

	bool cutoff = false;
	if (ttBestCol >= 0 && ttBestCol < N) {
		cutoff = exploreMove(ttBestCol);
	}
	if (!cutoff) {
		for (int col : order) {
			if (col == ttBestCol) {
				continue;
			}
			if (exploreMove(col)) {
				break;
			}
		}
	}

	BoundType bound = BOUND_EXACT;
	if (value <= alphaOrig) {
		bound = BOUND_UPPER;
	} else if (value >= betaOrig) {
		bound = BOUND_LOWER;
	}
	return storeTT(value, bestCol, bound);
}

const std::vector<int>& getMoveOrder(int N) {
	static std::vector<int> cached[MAX_BOARD + 1];
	std::vector<int>& order = cached[N];
	if (!order.empty()) {
		return order;
	}

	int center = N / 2;
	order.push_back(center);
	for (int d = 1; d <= N; d++) {
		if (center - d >= 0) {
			order.push_back(center - d);
		}
		if (center + d < N) {
			order.push_back(center + d);
		}
	}
	return order;
}

void initZobrist() {
	TT.reserve(1 << 19);
	for (int i = 0; i < MAX_BOARD; i++) {
		for (int j = 0; j < MAX_BOARD; j++) {
			for (int k = 0; k < 3; k++) {
				zobrist[i][j][k] = rng();
			}
		}
	}
}

int evaluate4(
    int** board,
    int x,
    int y,
    int dx,
    int dy
){
    return evaluateWindow(
        board[x][y],
        board[x+dx][y+dy],
        board[x+2*dx][y+2*dy],
        board[x+3*dx][y+3*dy]
    );
}
