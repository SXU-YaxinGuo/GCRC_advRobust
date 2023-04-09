import re
words_list = ['是','的','下列','不','正确','对','原文','表述','关于','符合','理解','内容','意思','文意','分析','说法','有关','中','最','概括',
              '恰当','根据','材料','属于','一','项','，','二','认识','观点','解说','思路','选项','各项','画线','句子','相关','下面','推断','支持',
              '__','point','</point','>',"<","</",'与','和','）',"（",'联系','全文','[',']','叙述','对于','作者','文章','几个','重要','概念',
              '相符','上下文','第③','段','画','线','句','第①','第②','第④','第③','陈述','一致','说明','从','看','第三','段','说','错误',
              '第一','第二','第三','第四','自然段','依据','以上','三','二','一','以下','判断','各句','】','【','文本','语句','两项','填写','部分','适合',
              '文字','标题','是','划线','文中','链接','上述','文中','定义','合理','、','原意','加','字词','点','“','或','合理','表达','话','各','。',
              "从","看",'说明',"《","》",'准确','证明','不符','错误','涉及','结合','几个','词','所谓','根本','几个','提供','信息','所拟','为','有误']


def cut_sentence(passage):
    """
    [passage, question, option_a,label,id]
    """
    cut_flag = ["。","？","！","*"]
    judge_flag = ['“','”']
    passage_sentence_num = []
    passage_sentence_list = []
    judge = []
    passage = passage.replace("......", "*").replace("\\\n", "")

    one_sentence = ""
    for k in range(len(passage)):
        if passage[k] not in cut_flag:
            one_sentence = one_sentence + passage[k]
            if passage[k] in judge_flag:
                if passage[k] == judge_flag[0]:
                    judge.append(passage[k])
                elif passage[k] == judge_flag[1] and len(judge) != 0:
                    judge.pop()

        else:
            if len(judge) >= 1:
                if len(one_sentence) > 30:
                    one_sentence = one_sentence + passage[k]

                    passage_sentence_list.append(one_sentence.replace("\n", ""))
                    one_sentence = ""
                else:
                    one_sentence = one_sentence + passage[k]
            else:
                if passage[k] == "*":
                    one_sentence = one_sentence + "......"
                    one_sentence = one_sentence.strip()
                    passage_sentence_list.append(one_sentence.replace("\n", ""))
                    one_sentence = ""
                else:
                    if one_sentence == "”":
                        one_sentence = ""
                    else:
                        one_sentence = one_sentence + passage[k]
                        one_sentence = one_sentence.strip()
                        passage_sentence_list.append(one_sentence)
                        one_sentence = ""

    passage_sentence_num.append(len(passage_sentence_list))
    return passage_sentence_list


def get_question_result_word_count(question,ltp):
    final_question = ""
    word_list,_ = ltp.seg([question])
    for word in word_list[0]:
        if word not in words_list:
            final_question = final_question+word
    return final_question


if __name__ == '__main__':
    passage = '许多动物的某些器官感觉特别灵敏，它们能比人类提前知道一些灾害事件的发生，例如，海洋中的水母能预报风暴，老鼠能事先躲避矿井崩塌或有害气体，等等。地震往往能使一些动物的某些感觉器官受到刺激而发生异常反应。'
    sentences_list = cut_sentence(passage)
    for item in sentences_list:
        print(item)



