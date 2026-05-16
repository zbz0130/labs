import sys
import os
import json
import math
SMALL = 0.001
#加载数据
def load_data():
    # 获取资源文件的正确路径
    def resource_path(relative_path):
        """获取资源文件的绝对路径，适用于开发环境和PyInstaller打包后"""
        try:
            # PyInstaller创建临时文件夹将路径存储在_MEIPASS中
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")
        return os.path.join(base_path, relative_path)
    
    with open(resource_path('1_word.txt'),'r',encoding = 'utf-8') as f:
        unigram = json.load(f)
    with open(resource_path('2_word.txt'),'r',encoding = 'utf-8') as f:
        bigram = json.load(f)
    with open(resource_path('3_word.txt'),'r',encoding = 'utf-8') as f:
        trigram = json.load(f)
    #unigram_dict["qing"]["清"] = freq
    unigram_dict = {}
    #bigram_dict["ba nen"]["巴 嫩"] = freq
    bigram_dict = {}
    #trigram_dict["da xue sheng"]["大 学 生"]= freq
    trigram_dict = {}
    #统计拼音中词的总频数
    #unigram_nums["qing"] = 1000
    unigram_nums = {}
    tot_nums = 0
    for pinyin,datas in unigram.items():
        py_dict = {}
        for word,counts in zip(datas["words"], datas["counts"]):
            py_dict[word] = counts
            unigram_nums[pinyin]=unigram_nums.get(pinyin,0)+counts
            tot_nums+=counts
        unigram_dict[pinyin] = py_dict
    for pinyinpair, datas in bigram.items():
        pair_dict = {}
        for words, counts in zip(datas["words"], datas["counts"]):
            pair_dict[words] = counts
            # bigram_nums[pinyinpair]+=counts
        bigram_dict[pinyinpair] = pair_dict
    for pinyinpair, datas in trigram.items():
        tripair_dict = {}
        for words, counts in zip(datas["words"], datas["counts"]):
            tripair_dict[words] = counts
        trigram_dict[pinyinpair] = tripair_dict
    return unigram_dict, bigram_dict, trigram_dict, unigram_nums, tot_nums

#拼音->汉字
"""
P(w1w2...wk) = P(w1)P(w2|w1)P(w3|w1w2)...P(wk|w1w2w3..wk-1)
\approx P(w1)P(w2|w1)P(w3|w2)....P(wk|wk-1)
"""
def viterbi(pinyins, unigram_dict, bigram_dict, unigram_nums, tot_nums):
    LAMBDA = 0.99 #平滑概率参数
    def get_uni_prob(py,ch):
        """
        P(ch)的概率
        Args:
            py :拼音
            ch :字符
        """
        return unigram_dict.get(py,{}).get(ch,0.01)/tot_nums
    #dp[轮数][字] = 概率
    dp = [{} for _ in range(len(pinyins))]
    #回溯指针
    backptr = [{} for _ in range(len(pinyins))]
    FirstChrs = unigram_dict.get(pinyins[0],{}).keys()
    p0 = 0.0
    best_score = -float('inf')
    for chr0 in FirstChrs:
        # p0 = unigram_dict[pinyins[0]][chr0] / unigram_nums[pinyins[0]]
        p0 = unigram_dict[pinyins[0]][chr0] / tot_nums
        dp[0][chr0] = math.log(p0)
        backptr[0][chr0] = None
    for itr in range(1,len(pinyins)):
        py_prev = pinyins[itr-1]
        py_curr = pinyins[itr]
        py_pair = " ".join([py_prev,py_curr])
        candidates_prev = unigram_dict.get(py_prev,{}).keys()
        candidates_curr = unigram_dict.get(py_curr,{}).keys()
        for chr_curr in candidates_curr:
            dp[itr][chr_curr] = -float('inf')
            for chr_prev in candidates_prev:
                p_uni = get_uni_prob(py_curr, chr_curr)
                word_pair = f"{chr_prev} {chr_curr}"
                prev_count = unigram_dict.get(py_prev, {}).get(chr_prev, 0)
                if prev_count > 0:
                    bi_count = bigram_dict.get(py_pair, {}).get(word_pair, 0)
                    p_bi = LAMBDA * bi_count / prev_count + (1 - LAMBDA) * p_uni
                else:
                    p_bi = p_uni
                if p_bi <=0:
                    p_bi  = SMALL
                dp[itr][chr_curr] = max(dp[itr-1][chr_prev]+math.log(p_bi),dp[itr][chr_curr])
                if dp[itr][chr_curr] == dp[itr-1][chr_prev]+math.log(p_bi):
                    backptr[itr][chr_curr] = chr_prev
    n = len(pinyins) - 1
    best_ch = None
    for ch, score in dp[n].items():
        best_score = max(best_score, score)
        if best_score == score:
            best_ch =  ch
    result = []
    for t in range(n,-1,-1):
        result.append(best_ch)
        best_ch = backptr[t][best_ch]
    result.reverse()
    return "".join(result)


#不剪枝的三元模型
def viterbi_3gram(pinyins, uni_dict, bi_dict, tri_dict,unigram_nums, tot_nums):
    n = len(pinyins)
    if n == 0: return ""
    if n == 1: # 退化回一元处理
        return max(uni_dict.get(pinyins[0], {}), key=uni_dict.get(pinyins[0], {}).get)

    # dp[i][(w_prev, w_curr)] = max_log_prob
    dp = [{} for _ in range(n)]
    # backptr[i][(w_prev, w_curr)] = w_prev_prev
    backptr = [{} for _ in range(n)]

    # --- 1. 初始化 i=0 和 i=1 (处理前两个字) ---
    # 这部分本质上还是用二元模型，因为没有“前两个字”
    p0, p1 = pinyins[0], pinyins[1]
    for w0 in uni_dict.get(p0, {}):
        prob0 = uni_dict[p0][w0] / tot_nums
        for w1 in uni_dict.get(p1, {}):
            # 计算 P(w1|w0)
            bi_count = bi_dict.get(f"{p0} {p1}", {}).get(f"{w0} {w1}", 0)
            p_bi = 0.9 * (bi_count / uni_dict[p0][w0]) + 0.1 * (uni_dict[p1][w1]/tot_nums)
            
            dp[1][(w0, w1)] = math.log(prob0) + math.log(p_bi)
            backptr[1][(w0, w1)] = None

    # --- 2. 递推阶段 i=2 到 n-1 ---
    for i in range(2, n):
        p_pp = pinyins[i-2]
        p_p = pinyins[i-1]
        p_c = pinyins[i]
        
        # 预取三元字典
        tri_map = tri_dict.get(f"{p_pp} {p_p} {p_c}", {})
        bi_map_curr = bi_dict.get(f"{p_p} {p_c}", {})

        # 遍历当前位置的所有候选字 w_i
        for w_c in uni_dict.get(p_c, {}):
            p_uni = uni_dict[p_c][w_c] / tot_nums
            
            # 遍历上一时刻的所有状态 (w_{i-2}, w_{i-1})
            for (w_pp, w_p), prev_score in dp[i-1].items():
                # 计算 P(w_i | w_{i-2}, w_{i-1})
                # 需要用到三元、二元、一元进行平滑
                count_tri = tri_map.get(f"{w_pp} {w_p} {w_c}", 0)
                count_bi_prev = bi_dict.get(f"{p_pp} {p_p}", {}).get(f"{w_pp} {w_p}", 0)
                
                # 计算二元概率用于平滑
                count_bi_curr = bi_map_curr.get(f"{w_p} {w_c}", 0)
                p_bi = 0.9 * (count_bi_curr / uni_dict[p_p][w_p]) + 0.1 * p_uni
                
                # 三元转移概率
                if count_bi_prev > 0:
                    p_tri = 0.7 * (count_tri / count_bi_prev) + 0.2 * p_bi + 0.1 * p_uni
                else:
                    p_tri = p_bi
                
                score = prev_score + math.log(max(p_tri, 1e-20))
                
                state = (w_p, w_c)
                if state not in dp[i] or score > dp[i][state]:
                    dp[i][state] = score
                    backptr[i][state] = w_pp

    # --- 3. 回溯 ---
    last_state = max(dp[n-1], key=dp[n-1].get)
    res = [last_state[1], last_state[0]]
    curr_state = last_state
    for i in range(n-1, 1, -1):
        prev_prev = backptr[i][curr_state]
        res.append(prev_prev)
        curr_state = (prev_prev, curr_state[0])
        
    res.reverse()
    return "".join(res)

#剪枝的三元模型
def viterbi_3gram_pruning(pinyins, uni_dict, bi_dict, tri_dict, unigrams_nums, tot_nums):
    n = len(pinyins)
    if n == 0: return ""
    if n == 1: # 退化回一元处理
        return max(uni_dict.get(pinyins[0], {}), key=uni_dict.get(pinyins[0], {}).get)

    # 剪枝超参数 
    MAX_CANDIDATES = 15 # 维度1：每个拼音最多只考虑一元词频最高的前 15 个汉字
    BEAM_SIZE = 40       # 维度2：每一轮（每个时刻）最多只保留概率最大的 40 条路径状态

    # 辅助函数：获取某个拼音的 Top K 候选字
    def get_top_candidates(py):
        cands = uni_dict.get(py, {})
        if not cands: 
            return[py] #如果没有对应的字，返回拼音本身
        # 按照一元词频降序排序,截取前 MAX_CANDIDATES 个
        sorted_cands = sorted(cands.keys(), key=lambda k: cands[k], reverse=True)
        return sorted_cands[:MAX_CANDIDATES]

    # dp[i][(w_prev, w_curr)] = max_log_prob
    dp = [{} for _ in range(n)]
    # backptr[i][(w_prev, w_curr)] = w_prev_prev
    backptr =[{} for _ in range(n)]

    # --- 1. 初始化 i=0 和 i=1 (处理前两个字) ---
    p0, p1 = pinyins[0], pinyins[1]
    
    # 只取前两个拼音的 Top 候选字
    cands_0 = get_top_candidates(p0)
    cands_1 = get_top_candidates(p1)
    
    for w0 in cands_0:
        prob0 = uni_dict.get(p0, {}).get(w0, 1) / tot_nums
        for w1 in cands_1:
            bi_count = bi_dict.get(f"{p0} {p1}", {}).get(f"{w0} {w1}", 0)
            uni_w0_count = uni_dict.get(p0, {}).get(w0, 1)
            p_uni_w1 = uni_dict.get(p1, {}).get(w1, 1) / tot_nums
            
            p_bi = 0.9 * (bi_count / uni_w0_count) + 0.1 * p_uni_w1
            dp[1][(w0, w1)] = math.log(prob0) + math.log(max(p_bi, 1e-20))
            backptr[1][(w0, w1)] = None

    # 对第 1 轮的状态进行 Beam Search 剪枝
    if len(dp[1]) > BEAM_SIZE:
        sorted_dp1 = sorted(dp[1].items(), key=lambda x: x[1], reverse=True)
        dp[1] = dict(sorted_dp1[:BEAM_SIZE])

    # --- 2. 递推阶段 i=2 到 n-1 ---
    for i in range(2, n):
        p_pp = pinyins[i-2]
        p_p = pinyins[i-1]
        p_c = pinyins[i]
        
        tri_map = tri_dict.get(f"{p_pp} {p_p} {p_c}", {})
        bi_map_curr = bi_dict.get(f"{p_p} {p_c}", {})

        # 只取当前拼音的 Top 候选字
        cands_c = get_top_candidates(p_c)

        for w_c in cands_c:
            p_uni = uni_dict.get(p_c, {}).get(w_c, 1) / tot_nums
            
            # 遍历上一时刻经过剪枝保留下来的所有状态
            for (w_pp, w_p), prev_score in dp[i-1].items():
                count_tri = tri_map.get(f"{w_pp} {w_p} {w_c}", 0)
                count_bi_prev = bi_dict.get(f"{p_pp} {p_p}", {}).get(f"{w_pp} {w_p}", 0)
                count_bi_curr = bi_map_curr.get(f"{w_p} {w_c}", 0)
                
                # 平滑计算
                uni_wp_count = uni_dict.get(p_p, {}).get(w_p, 1)
                p_bi = 0.9 * (count_bi_curr / uni_wp_count) + 0.1 * p_uni
                
                if count_bi_prev > 0:
                    p_tri = 0.7 * (count_tri / count_bi_prev) + 0.2 * p_bi + 0.1 * p_uni
                else:
                    p_tri = p_bi
                
                score = prev_score + math.log(max(p_tri, 1e-20))
                
                state = (w_p, w_c)
                # 更新状态：保留最大概率
                if state not in dp[i] or score > dp[i][state]:
                    dp[i][state] = score
                    backptr[i][state] = w_pp

        # 对第 i 轮的状态进行 Beam Search 剪枝
        if len(dp[i]) > BEAM_SIZE:
            sorted_dpi = sorted(dp[i].items(), key=lambda x: x[1], reverse=True)
            dp[i] = dict(sorted_dpi[:BEAM_SIZE])

    #  回溯 
    if not dp[n-1]: return ""
    last_state = max(dp[n-1], key=dp[n-1].get)
    res = [last_state[1], last_state[0]]
    curr_state = last_state
    
    for i in range(n-1, 1, -1):
        prev_prev = backptr[i].get(curr_state)
        if prev_prev is None: break
        res.append(prev_prev)
        curr_state = (prev_prev, curr_state[0])
        
    res.reverse()
    return "".join(res)

def pinyin_3gram():
    try:
        unigram_dict, bigram_dict, trigram_dict, unigram_nums,tot_nums = load_data()
    except Exception as e:
        print(f"数据加载有问题：{e}")
        sys.exit(1)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        pinyins = line.split()
        sentence = viterbi_3gram(pinyins,unigram_dict, bigram_dict,trigram_dict,unigram_nums,tot_nums)
        print(sentence)
if __name__ == "__main__":
    pinyin_3gram()