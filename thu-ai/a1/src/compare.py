from pathlib import Path
import os
import sys

answer_path = Path("./data/answer.txt")
output_path = Path("./data/output.txt")
def compare():
    with open(answer_path, 'r', encoding='utf-8') as f_ans, \
        open(output_path,'r',encoding='utf-8') as f_out:
        ans_lines = [line.strip() for line in f_ans if line.strip()]
        out_lines = [line.strip() for line in f_out if line.strip()]
    if len(ans_lines) != len(out_lines):
        print(f"行数不匹配")
    else:
        count = len(ans_lines)
        correct_sentences = 0
        correct_words = 0
        total_words = 0
        wrong_sentences = []
        for i in range(count):
            ans_sent = ans_lines[i]
            out_sent = out_lines[i]
            if ans_sent == out_sent:
                correct_sentences += 1
            else:
                wrong_sentences.append(i)
            total_words += len(ans_sent)
            for j in range(len(ans_sent)):
                if(ans_sent[j]==out_sent[j]):
                    correct_words+=1
        sentence_accuracy = (correct_sentences/count)*100
        words_accuracy = (correct_words/total_words)*100
        print(f"句准确率{sentence_accuracy:.2f}%")
        print(f"字准确率{words_accuracy:.2f}%")
        print(wrong_sentences)