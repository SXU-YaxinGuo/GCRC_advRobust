# GCRC：高考语文阅读理解可解释数据集(The Dataset of Gaokao Chinese Reading Comprehension for Explainable Evaluation)

### 数据集简介
目前，在众多公开可用数据集的驱动下机器阅读理解模型取得了令人振奋的进展，但模型所具备的真实语言理解能力与人的期望相差甚远，且大多数据集提供的是“黑盒”（black-box）评价，不能诊断系统是否基于正确的推理过程获得答案。为了缓解这些问题并促进机器智能向类人智能迈进，山西大学在国家重点研发计划项目“基于大数据的类人智能关键技术与系统”的支持下，面向题目类型更丰富、更具挑战性的高考阅读理解任务做了重点研究，并尝试基于人的标准化测试对机器智能进行有效和实用的评价。我们收集近10年高考阅读理解测试题构建了一个包含5000多篇文本、8700多道选择题（约1.5万个选项）的数据集GCRC（A New MRC Dataset from Gaokao Chinese for Explainable Evaluation）。数据集标注了三种信息：句子级支持事实、干扰项（不正确选项）错误原因、回答问题所需推理能力，从中间推理、模型能力两方面进行可解释评价。相关实验表明该数据集具有更大挑战性，对于以可解释方式诊断系统局限性非常有用，有助于研究人员未来开发新的机器学习和推理方法解决这些挑战性问题。

### 论文
[GCRC: A New Challenging MRC Dataset from Gaokao Chinese for Explainable Evaluation](https://aclanthology.org/2021.findings-acl.113.pdf). ACL 2021 Findings.

### 数据集规模
训练集：6,994个问题；验证集：863个问题；
```
数据集划分  训练集	 验证集	
文章数量      3790	 683	 
问题数量	   6994	 863	 
标注SFs的问题/选项数量    2021/8084 	863/3452	
标注ERs的问题/干扰项数量    2000/3253	863/1437	
标注推理能力的问题/选项数量	- 	863/3452	
```
### 数据格式描述
每条数据包含
编号(id，以字符串形式存储)，
标题(title，以字符串形式存储)，
文章(passage，以字符串形式存储)，
问题(question，以字符串形式存储)，
选项(options，以列表形式存储，分别表示A、B、C、D四个选项的内容)，
选项支持句(evidences，以列表形式存储，分别表示A、B、C、D四个选项对应的原文的支撑句内容)，
推理能力(reasoning_ability,以列表形式存储，分别表示A、B、C、D四个选项对应的答题所需推理能力，具体推理能力（括号内为标签标号）为：细节推理（Detail understanding，DTL-R）、共指推理（Coreference resolution，CO-REF）、演绎推理（Deductive reasoning，DED）、数字推理（Mathematical reasoning，MATH）、时空推理（Temporal/spatial reasoning，TEM-SPA）、因果推理（Cause-effect comprehension，CAU-R）、归纳推理（Inductive reasoning，IND）、鉴赏分析（Appreciative analysis，APPREC）,
错误原因(error_type，以列表形式存储，分别表示判别A、B、C、D四个选项的错误原因)，
答案(answer，以字符串形式存储)，

### 数据样例
```json
{
  "id": "gcrc_4916_8172", 
  "title": "我们需要怎样的科学素养", 
  "passage": "第八次中国公民科学素养调查显示，2010年，我国具备...激励科技创新、促进创新型国家建设，我们任重道远。", 
  "question": "下列对“我们需要怎样的科学素养”的概括，不正确的一项是", 
  "options":  [
    '科学素养是一项基本公民素质，公民科学素养可以从科学知识、科学方法和科学精神三个方面来衡量。',
    '不仅需要掌握足够的科学知识、科学方法，更需要具备学习、理解、表达、参与和决策科学事务的能力。',
    '应该明白科学技术需要控制，期望科学技术解决哪些问题，希望所纳的税费使用于科学技术的哪些方面。', 
    '需要具备科学的思维和科学的精神，对科学技术能持怀疑态度，对于媒体信息具有质疑精神和过滤功能。'
  ],
  "evidences": [
    ['公民科学素养可以从三个方面衡量：科学知识、科学方法和科学精神。', '在“建设创新型国家”的语境中，科学素养作为一项基本公民素质的重要性不言而喻。'],
    ['一个具备科学素养的公民，不仅应该掌握足够的科学知识、科学方法，更需要强调科学的思维、科学的精神，理性认识科技应用到社会中可能产生的影响，进而具备学习、理解、表达、参与和决策科学事务的能力。'], 
    ['西方发达国家不仅测试公众对科学技术与社会、经济、文化等各方面关系的看法，更考察公众对科学技术是否持怀疑态度，是否认为科学技术需要控制，期望科学技术解决哪些问题，希望所纳的税费使用于科学技术的哪些方面等。'], 
    ['甚至还有国家专门测试公众对于媒体信息是否具有质疑精神和过滤功能。', '西方发达国家不仅测试公众对科学技术与社会、经济、文化等各方面关系的看法，更考察公众对科学技术是否持怀疑态度，是否认为科学技术需要控制，期望科学技术解决哪些问题，希望所纳的税费使用于科学技术的哪些方面等。']
   ],
  "reasoning_ability": ["DTL-R","DTL-R","IND","IND"],
  "error_type": ["ITQ", "", "", ""],
  "answer": "A",
}
```


### 作者列表
谭红叶，王笑月，吉宇，李茹，李晓黎，胡志伟，赵云肖，韩孝奇
### 制作单位
山西大学

### 论文引用
```bibtex
@inproceedings{tan-etal-2021-gcrc,
    title = "{GCRC}: A New Challenging {MRC} Dataset from {G}aokao {C}hinese for Explainable Evaluation",
    author = "Tan, Hongye  and
      Wang, Xiaoyue  and
      Ji, Yu  and
      Li, Ru  and
      Li, Xiaoli  and
      Hu, Zhiwei  and
      Zhao, Yunxiao  and
      Han, Xiaoqi",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.113",
    doi = "10.18653/v1/2021.findings-acl.113",
    pages = "1319--1330",
}
```