#include <iostream>
#include <vector>
#include<algorithm>
#include<climits>
#include<cstring>
#include "Point.h"
#include "Strategy.h"

using namespace std;

/*
	策略函数接口,该函数被对抗平台调用,每次传入当前状态,要求输出你的落子点,该落子点必须是一个符合游戏规则的落子点,不然对抗平台会直接认为你的程序有误
	
	input:
		为了防止对对抗平台维护的数据造成更改，所有传入的参数均为const属性
		M, N : 棋盘大小 M - 行数 N - 列数 均从0开始计， 左上角为坐标原点，行用x标记，列用y标记
		top : 当前棋盘每一列列顶的实际位置. e.g. 第i列为空,则_top[i] == M, 第i列已满,则_top[i] == 0
		_board : 棋盘的一维数组表示, 为了方便使用，在该函数刚开始处，我们已经将其转化为了二维数组board
				你只需直接使用board即可，左上角为坐标原点，数组从[0][0]开始计(不是[1][1])
				board[x][y]表示第x行、第y列的点(从0开始计)
				board[x][y] == 0/1/2 分别对应(x,y)处 无落子/有用户的子/有程序的子,不可落子点处的值也为0
		lastX, lastY : 对方上一次落子的位置, 你可能不需要该参数，也可能需要的不仅仅是对方一步的
				落子位置，这时你可以在自己的程序中记录对方连续多步的落子位置，这完全取决于你自己的策略
		noX, noY : 棋盘上的不可落子点(注:其实这里给出的top已经替你处理了不可落子点，也就是说如果某一步
				所落的子的上面恰是不可落子点，那么UI工程中的代码就已经将该列的top值又进行了一次减一操作，
				所以在你的代码中也可以根本不使用noX和noY这两个参数，完全认为top数组就是当前每列的顶部即可,
				当然如果你想使用lastX,lastY参数，有可能就要同时考虑noX和noY了)
		以上参数实际上包含了当前状态(M N _top _board)以及历史信息(lastX lastY),你要做的就是在这些信息下给出尽可能明智的落子点
	output:
		你的落子点Point
*/
bool checkWin(int** board, int M, int N, int x, int y, int player);

int evaluateWindow(std::vector<int> window);

int evaluateBoard(int** board, int M, int N);

bool isValid(int col, const int* top);

int minimax(
    int** board,
    int M,
    int N,
    int* top,
    int depth,
    int alpha,
    int beta,
    bool maximizingPlayer
);

extern "C" Point* getPoint(const int M, const int N, const int* top, const int* _board, 
	const int lastX, const int lastY, const int noX, const int noY){
	/*
		不要更改这段代码
	*/
	int mytop[15] = {};
	memcpy(mytop, top, sizeof(top));
	int x = -1, y = -1;//最终将你的落子点存到x,y中
	int** board = new int*[M];
	for(int i = 0; i < M; i++){
		board[i] = new int[N];
		for(int j = 0; j < N; j++){
			board[i][j] = _board[i * N + j];
		}
	}
	//自己能否直接赢
	for(int col = 0;col < N; col++){
		if(top[col]<=0)
			continue;
		int row = top[col] - 1;
		board[row][col] = 2;
		if(checkWin(board, M, N, row, col, 2)){
			x = row;
			y = col;
			board[row][col] = 0;
			break;
		}
		board[row][col] = 0;
	}
	//挡住对面必胜
	if(x==-1){
		for(int col = 0; col<N; col++)
		{
			if(top[col]<=0)
				continue;
			int row = top[col]-1;
			board[row][col] = 1;
			if(checkWin(board, M, N, row, col, 1)){
				x = row;
				y = col;
				board[row][col] = 0;
				break;
			}
			board[row][col] = 0;
		}
	}
	if(x!=-1)
	{
		clearArray(M, N, board);
		return new Point(x, y);
	}

	int bestScore = INT_MIN;
	int bestCol = -1;
	const int SEARCH_DEPTH = 6;
	std::vector<int> order;
	int center = N/2;
	order.push_back(center);
	for(int d = 1;d<=N; d++){
		if(center - d>=0)
			order.push_back(center-d);
		if(center+d<N)
			order.push_back(center+d);
	}
	//Alpha-Beta搜索
	for(int col:order){
		if(top[col]<=0)
			continue;
		int row = top[col] - 1;
		board[row][col] = 2;
		mytop[col]--;
		int score = minimax(board, M, N, mytop, SEARCH_DEPTH - 1, INT_MIN,INT_MAX, false);
		board[row][col] = 0;
		mytop[col]++;
		if(score>bestScore){
			bestScore = score;
			bestCol = col;
		}
	}
	y = bestCol;
	x = top[bestCol]-1;
	/*
		不要更改这段代码
	*/
	clearArray(M, N, board);
	return new Point(x, y);
}


/*
	getPoint函数返回的Point指针是在本dll模块中声明的，为避免产生堆错误，应在外部调用本dll中的
	函数来释放空间，而不应该在外部直接delete
*/
extern "C"  void clearPoint(Point* p){
	delete p;
	return;
}

/*
	清除top和board数组
*/
void clearArray(int M, int N, int** board){
	for(int i = 0; i < M; i++){
		delete[] board[i];
	}
	delete[] board;
}


/*
	添加你自己的辅助函数，你可以声明自己的类、函数，添加新的.h .cpp文件来辅助实现你的想法
*/
bool checkWin(int** board, int M, int N, int x, int y, int player) {

    // 四个方向：
    // 1. 竖直
    // 2. 水平
    // 3. 主对角线
    // 4. 副对角线
    int dx[4] = {1, 0, 1, 1};
    int dy[4] = {0, 1, 1, -1};

    // 枚举四个方向
    for (int dir = 0; dir < 4; dir++) {

        int count = 1; // 包含自己

        // 正方向统计
        for (int step = 1; step < 4; step++) {

            int nx = x + dx[dir] * step;
            int ny = y + dy[dir] * step;

            // 越界
            if (nx < 0 || nx >= M || ny < 0 || ny >= N)
                break;

            // 连续相同棋子
            if (board[nx][ny] == player)
                count++;
            else
                break;
        }

        // 反方向统计
        for (int step = 1; step < 4; step++) {

            int nx = x - dx[dir] * step;
            int ny = y - dy[dir] * step;

            // 越界
            if (nx < 0 || nx >= M || ny < 0 || ny >= N)
                break;

            // 连续相同棋子
            if (board[nx][ny] == player)
                count++;
            else
                break;
        }

        // 四连
        if (count >= 4)
            return true;
    }

    return false;
}
//评分函数
int evaluateWindow(std::vector<int> window){
	int self = 0;
	int opp = 0;
	int empty = 0;
	for(int x:window){
		if(x==2)
			self++;
		else if (x==1)
			opp++;
		else 
			empty++;
	}
	int score = 0;
	//自己
	if (self == 4)
		score += 100000;
	else if(self ==3&&empty==1)
		score += 1000;
	else if(self==2&&empty==2)
		score+=100;
	if(opp ==4)
		score -= 100000;
	else if (opp==3&empty==1)
		score -=1200;
	else  if(opp==2&&empty ==2)
		score -= 120;
	return score;
}
//棋盘总评分
int evaluateBoard(int** board, int M,int N){
	int score = 0;
	int center = N/2;
	//中心列
	for(int i = 0;i<M;i++){
		if(board[i][center]==2)
			score += 6;
		else if(board[i][center] == 1)
			score -= 6;
	}
	//横向
	for(int i = 0; i<M;i++){
		for(int j = 0;j+3<N;j++){
			std::vector<int> window;
			for(int k = 0;k<4;k++)
				window.push_back(board[i][j+k]);
			score += evaluateWindow(window);
		}
	}
	//纵向
	for(int i = 0;i+3<M;i++)
	{
		for(int j = 0;j<N;j++){
			std::vector<int> window;
			for(int k = 0;k<4;k++)
				window.push_back(board[i+k][j]);
			score += evaluateWindow(window);
		}
	}
	//主对角线
	for(int i = 0;i+3<M;i++)
	{
		for(int j = 0;j+3<N;j++)
		{
			std::vector<int> window;
			for(int k = 0;k<4;k++)
				window.push_back(board[i+k][j+k]);
			score+=evaluateWindow(window);
		}
	}
	//副对角线
	for(int i = 0;i+3<M;i++){
		for(int j = 3;j<N;j++){
			std::vector<int> window;
			for(int k = 0;k<4;k++)
				window.push_back(board[i+k][j-k]);
			score += evaluateWindow(window);
		}
	}
	return score;
}
int minimax(int** board, int M,int N, int * top, int depth, int alpha, int beta, bool maximizingPlayer){
	if(depth==0){
		return evaluateBoard(board, M, N);
	}
	bool hasMove = false;
	for(int col = 0;col<N;col++){
		if(top[col]>0){
			hasMove = true;
			break;
		}
	}
	//平局 
	if(!hasMove) return 0;
	if(maximizingPlayer){
		int value = INT_MIN;
		for(int col = 0;col<N; col++){
			if(top[col]<=0)
				continue;
			int row = top[col] - 1;
			board[row][col] = 2;
			top[col]--;
			int score;
			if( checkWin(board, M, N,row,col, 2))
				score = 1000000 + depth;
			else
				score = minimax(board, M, N, top, depth - 1, alpha, beta, false);
				board[row][col] = 0;
				top[col]++;
				value = std::max(value, score);
				alpha = std::max(alpha, value);
				if(alpha >= beta)
					break;
			}
			return value;
	}
	else{
		int value = INT_MAX;
		for(int col = 0;col<N;col++)
		{
			if(top[col]<=0)
				continue;
				int row = top[col] - 1;
				board[row][col] = 1;
				top[col]--;
				int score;
				if(checkWin(board,M,N,row,col,1))
					score = -1000000 - depth;
				else
					score = minimax(board, M,N,top,depth-1,alpha,beta,true);
					board[row][col]=0;
					top[col]++;
					value = std::min(value, score);
					beta = std::min(beta, value);
					if(alpha>=beta)
						break;

		}
		return value;
	}

}