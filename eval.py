# -*- coding: utf-8 -*-
"""
@author: Yaxin Guo
@software: PyCharm
@file: eval.py
@email:202112407002@email.sxu.edu.cn
@time: 2023/3/30 15:43
"""
import sys
import os
import json


def eval_advrobust(pred_file, golden_file):
    pred_o, gold_o = [], []
    pred_p, gold_p = [], []
    pred_n, gold_n = [], []

    try:
        with open(file=pred_file, mode="r", encoding="utf-8") as fin:
            pred_data = json.load(fin)
            pred_jsons = pred_data['data']
            for data_json in pred_jsons:
                pred_o.append((data_json['id'], data_json['answer']))
                pred_p.append((data_json['id'], data_json['positive_answer']))
                pred_n.append((data_json['id'], data_json['negative_answer']))

        succ_info = "predict file load succeed..."
        print(succ_info)
    except Exception as e:
        err_info = "predict file load failed. please upload a json file. err: {}".format(e)
        print(err_info)
        exit(-1)

    try:
        with open(file=golden_file, mode="r", encoding="utf-8") as fin:
            gold_data = json.load(fin)
            gold_jsons = gold_data['data']
            for data_json in gold_jsons:
                gold_o.append((data_json['id'], data_json['answer']))
                gold_p.append((data_json['id'], data_json['positive_answer']))
                gold_n.append((data_json['id'], data_json['negative_answer']))

        succ_info = "golden file load succeed..."
        print(succ_info)
    except Exception as e:
        err_info = "golden file load failed. please upload a json file. err: {}".format(e)
        print(err_info)
        exit(-1)

    # 原始选项
    pred_res_set_o, gold_res_set_o = set(pred_o), set(gold_o)
    # 正对抗选项
    pred_res_set_p, gold_res_set_p = set(pred_p), set(gold_p)
    # 负对抗选项
    pred_res_set_n, gold_res_set_n = set(pred_n), set(gold_n)
    # 原始选项正确集合
    correct_ori = pred_res_set_o & gold_res_set_o
    # 正对抗选项正确集合
    correct_posi = pred_res_set_p & gold_res_set_p
    # 负对抗选项正确集合
    correct_nega = pred_res_set_n & gold_res_set_n

    # 去正确样例编号
    correct_ori_list = []
    for i in correct_ori:
        correct_ori_list.append(i[0])
    correct_ori_set = set(correct_ori_list)

    correct_posi_list = []
    for i in correct_posi:
        correct_posi_list.append(i[0])
    correct_posi_set = set(correct_posi_list)

    correct_nega_list = []
    for i in correct_nega:
        correct_nega_list.append(i[0])
    correct_nega_set = set(correct_nega_list)

    # 原始选项与正对抗选项正确交集
    mix_oripo = correct_ori_set & correct_posi_set

    # 原始选项与负对抗选项正确交集
    mix_orine = correct_ori_set & correct_nega_set


    # 只有原始作对的选项


    #  原始始选项和对抗选项其中之一正确预测选项
    correct_adv1 = mix_oripo | mix_orine

    # 原始选项和两个对抗选项均正确预测选项
    correct_adv2 = mix_oripo & mix_orine

    ori_acc = 1.0 * len(correct_ori_set) / len(pred_res_set_o)
    adv_acc1 = 1.0 * len(correct_adv1) / len(pred_res_set_o)
    adv_acc2 = 1.0 * len(correct_adv2) / len(pred_res_set_o)
    Score = 0.2 * ori_acc + 0.3 * adv_acc1 + 0.5 * adv_acc2

    return ori_acc, adv_acc1, adv_acc2, Score


if __name__ == '__main__':
    pred_file = sys.argv[1]
    golden_file = sys.argv[2]

    if not os.path.exists(pred_file) or not os.path.exists(golden_file):
        print(
            "predict file load failed. please upload a json file.err:predict file is not existing." if not os.path.exists(
                pred_file) else "")
        print(
            "golden file load failed. please upload a json file.err:golden file is not existing." if not os.path.exists(
                golden_file) else "")
        exit(-1)

    ori_acc, adv_acc1, adv_acc2, Score = eval_advrobust(pred_file, golden_file)

    print(json.dumps({"Acc0": ori_acc, "Acc1": adv_acc1, "Acc2": adv_acc2, "Score": Score}))

''' shell
python eval.py prediction_file test_GCRCadvrobust_private_file
eg: python eval_ccl_public.py ./submit/dev_submit.json  ./submit/dev_GCRCadvrobust.json
'''
