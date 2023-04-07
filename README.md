# CCL23-Eval汉语高考阅读理解对抗鲁棒评测
GCRC_roBust: Adversarial Robustness Evaluation for Chinese Gaokao Reading Comprehension   
## 1.任务简介
&emsp;&emsp;机器阅读理解模型的鲁棒性是衡量该技术能否在实际应用中大规模落地的关键[1]。随着技术的进步，现有模型已经能够在封闭测试集上取得较好的性能，但在面向开放、动态、真实环境下的推理与决策时，其鲁棒性仍表现不佳[2-5]。为了评估模型的鲁棒性，现有工作主要通过添加文本噪声[6]或对问题进行复述[7]来干扰原始题目，但这类方法的攻击方式比较单一，题目难度也相对较小，对衡量模型性能存在一定局限性。     
&emsp;&emsp;为了提升机器阅读理解模型在复杂、真实对抗环境下的鲁棒性，我们基于“CCL2022-高考语文阅读理解可解释评测”数据集GCRC[8]（The dataset of Gaokao Chinese Reading Comprehension），构建了对抗鲁棒子集GCRC_advRobust，并提出“汉语高考阅读理解对抗鲁棒评测”任务。不同于“CCL2022-高考语文阅读理解可解释评测”主要对模型的中间推理能力进行可解释性评价，本次评测设计了四种对抗攻击策略（关键词扰动、推理逻辑扰动、时空属性扰动、因果关系扰动），重点挑战模型在多种对抗攻击下的鲁棒性。     
&emsp;&emsp;本次评测设置了开放和封闭两个赛道，其中开放赛道中，参赛队伍可以使用ChatGPT、文心一言等大模型；封闭赛道中，参赛的模型参数量最多不超过1.5倍Bert-large（510M）。    
+ 组织者
  + 谭红叶（山西大学）
  + 李  茹（山西大学）
  + 张  虎（山西大学）
  + 俞  奎（合肥工业大学）
+ 联系人
  + 郭亚鑫（山西大学博士生，202112407002@email.sxu.edu.cn）
  + 孙欣伊（山西大学博士生）
  + 闫国航（山西大学硕士生）
  
&emsp;&emsp;评测任务详细内容可查看评测网站： https://github.com/SXU-YaxinGuo/GCRC_advRobust ，遇到任何问题请发邮件或在Issue中提问，欢迎大家参与。
## 2.评测数据
### 对抗攻击策略：
- 关键词扰动策略：通过词语替换或重新表述，对影响选项语义的关键词进行干扰。   
例1：    
原始选项：自然资源丰富的湿地，是人类的“衣食父母”，为人类生存发展提供了所有物资，如食物、饮水、能源等。（错误选项）      
正对抗选项：自然资源丰富的湿地，是人类的“衣食父母”，为人类生存发展提供了全部物资，如食物、饮水、能源等。      
负对抗选项：自然资源丰富的湿地，是人类的“衣食父母”，为人类生存发展提供了部分物资，如食物、饮水、能源等。      
- 时空属性扰动策略：通过改变时间或空间属性，对选项中的时空信息进行干扰。      
例2：   
原始选项：由于19世纪中叶中国文化在与西方文化的抗争中处于弱势地位，人们才提出“保存国学”“振兴国学”的口号，“国学”一词由此出现。（错误选项）     
正对抗选项：20世纪中叶中国文化在与西方文化的抗争中处于弱势地位，人们才提出“保存国学”“振兴国学”的口号。      
负对抗选项：19世纪中叶中国文化在与西方文化的抗争中处于弱势地位，20世纪初，人们才提出“保存国学”“振兴国学”的口号。       
- 因果关系扰动策略：通过更改或去除因果联系，对选项中的因果关系进行干扰。      
例3：    
原始选项：中国之所以选择和平共处五项原则，是为了在务实的基础上让外界消除误解。（错误选项）   
正对抗选项：因为中国选择了和平共处五项原则，所以在务实的基础上让外界消除误解。   
负对抗选项：中国选择和平共处五项原则，并积极在务实的基础上让外界消除误解。   
- 推理逻辑扰动策略：通过改写前提或结论，对选项的逻辑推理过程进行干扰。      
例4：          
原始选项：气味分子在属于G蛋白的嗅觉受体的作用下从化学信号转变成为电信号。（正确选项）        
正对抗选项：与属于G蛋白的嗅觉受体结合后，在它的作用下，气味分子从化学信号转变成为电信号。         
负对抗选项：气味分子与嗅觉受体结合后，气味分子便自行从化学信号转变成为电信号。        
##### 注意：    
1. 正确选项指与原文意思相符的选项；错误选项指与原文意思不符的选项；正对抗选项与原始选项正误相同；负对抗选项与原始选项正误相反。    
2. 推理逻辑扰动策略主要攻击由原文经过归纳推理或演绎推理得出结论的推理过程。    
### 数据集规模
本评测提供GCRC原始数据作为训练集，题目数为6994，提供GCRC_advRobust作为验证集与测试集。GCRC_advRobust数据集规模如表所示。     

|数据集划分|验证集|测试集|
| :----- :|:----- :|:-----: |
|问题/选项数量|336/1344|288/1152|
|关键字词扰动选项数量|504|418|
|推理逻辑扰动选项数量|619|543|
|因果关系扰动选项数量|192|172|
|时空属性扰动选项数量|29|19|

### 数据样例
验证集和测试集中的每条数据包含以下内容：编号(id）、标题(title)、文章(passage)、原始问题(question)、原始选项(options)、原始答案(answer)、正对抗选项（positive_options）、正对抗答案(positive_answer)、负对抗问题（negative_question）、负对抗选项（negative_options）、负对抗答案(negative_answer)。     
具体数据样例如下所示：     
```json
{
    "id": "gcrc_5093_8539",
    "title": "宣纸——中华文化中的瑰宝韩作荣",
    "passage": "“宣纸”作为纸张名词的出现，始于...纸张相比，其抗蠹虫蚀蛀的能力强，据检测，其生存寿命超过一千零五十年，被称之为千年寿纸。",
    "question": "根据原文提供的信息，下列推断正确的一项是",
    "options": [
          "因为明宣德年间皇室监制加工制造了宣纸，所以与宣德炉、宣德窑一样，“宣德纸”就成了今天的宣纸。",
          "用两年生的青檀长出的嫩芽的韧皮和泾县安吴地区沙田的稻草，方能制成最佳的真宣。",
          "只有泾县清醇洁净、硬度低、水温低的泉水，才能制造出适宜于中国书画的宣纸。",
          "独特的用料、特殊的工艺、纯净的泉水、制造者的聪明才智，是制造千年寿纸的保证。"
    ],
	"answer": "D",
	"positive_options": [
          "明宣德年间皇室监制加工制造了宣纸，与宣德炉、宣德窑一样，“宣德纸”成为了宣纸的别名。",
          "用泾县生在山石崎岖倾仄之间的两年生青檀长出的嫩芽的韧皮和泾县安吴地区沙田的稻草，方能制成最佳的宣纸。",
          "只有泾县清醇洁净、硬度低、水温低的泉水，才能制造出适宜于中国书画的宣纸。",
          "只需独特的用料、特殊的工艺、纯净的泉水、制造者的聪明才智，就能出制造千年寿纸。"
	],
	"positive_answer": "A",
	"negative_question": "根据原文提供的信息，下列推断错误的一项是",
	"negative_options": [
          "明宣德年间皇室监制加工制造了宣纸，与宣德炉、宣德窑一样，“宣德纸”成为了宣纸的别名。",
          "制作宣纸最佳的青檀皮是生长两年的嫩枝的韧皮，最佳的稻草是泾县安吴地区沙田的稻草。",
          "好的宣纸必须用泾县清醇洁净的泉水，水的硬度低，水温低，才能使宣纸不惹灰尘，洁白度高，并能延长纸的寿命。",
          "独特的用料、纯净的泉水、适宜的气候，制造者的聪明才智，是制造千年寿纸不可或缺的。"
    ],
	"negative_answer": "C"
}
```

##### 注意：
+ 训练集中的数据不包含对抗选项，仅包含编号(id）、标题(title)、文章(passage)、原始问题(question)、原始选项(options)、原始答案(answer)字段。
+ 在实际发布的验证集与测试集中我们将正、负对抗选项以及原始选项进行随机组合形成了对抗选项集合。
### 数据说明
&emsp;&emsp;我们根据GCRC数据集中原始题目的四个选项所涉及到的推理能力，设计相应的对抗攻击策略，为每个选项构建了一个正对抗选项和一个负对抗选项，形成了对抗鲁棒子集GCRC_advRobust。该子集的每条数据由原始题目及其正负对抗题目三者组成。其中原始题目包含文章、问题和原始选项集合；正对抗题目包含文章、问题和正对抗选项集合；负对抗题目包含文章、负对抗问题和负对抗选项集合。
评测要求参赛者输出原始题目及其对抗题目的答案。
### 任务输入与输出
参赛者须将验证集和测试集每条样例拆分成原始题目、正对抗题目和负对抗题目作为模型的输入，并得到对应的三个答案。其中原始题目字段集为<id,title,passage,question,options,answer>；正对抗题目字段集为<id,title,passage,question,positive_options,positive_answer>；负对抗题目字段集为<id,title,passage,negative_question,negative_options,negative_answer>。
##### 注意：模型输出与验证集格式保持一致。
### 基线系统
本次评测使用的阅读理解基线系统为开源的中文预训练模型MacBERT[9]。基线系统的说明将在智源指数平台上发布。
## 3.评价标准
参赛系统的最终得分由Acc_0、Acc_1、Acc_2三个指标综合决定，具体计算公式如下：    
Score=0.2\*Acc_0+0.3\*Acc_1+0.5\*Acc_2    
其中：   
Acc_0=原始题目正确预测个数/题目总数    
Acc_1=原始题目和任意一个对抗题目正确预测个数/题目总数    
Acc_2=原始选项和两个对抗题目均正确预测个数/题目总数    
指标计算脚本eval.py会随数据集一起发布。
## 4.评测赛程
### 报名方式： 
本次评测在智源指数平台上进行报名，届时将会开通相应的报名系统。队长创建队伍后，队伍成员可凭队伍借邀请码加入队伍。报名截止前未完成报名者不参与后续的评选。
### 赛程安排：
+ 报名时间：2023年4月10日-5月20日
+ 训练、验证、测试数据发布：2023年4月10日
  + 报名结束后在智源指数平台上获取评测数据
+ 提交测试结果：2023年6月1日
+ 公布测试结果：2023年6月10日
+ 中英文技术报告提交：2023年6月20日
+ 中英文技术报告反馈：2023年6月28日
+ 中英文评测论文提交：2023年7月3日
+ 公布获奖名单：2023年7月７日
+ 评测论文录用通知：2023年7月10日
+ 论文Camera Ready提交：2023年7月15日
+ 评测研讨会及颁奖：2023年8月3-5日
### 结果提交：
本次评测结果在智源指数平台上进行提交和排名，届时将会开通相应的提交与评测系统，参赛者可以在网站上注册账号并提交相应的测试文件。在参赛期间，严禁参赛团队注册其它账号多次提交。
参赛者需要提交三种文件：
1)输出结果文件：该文件是以utf-8为编码格式的json文件，其中的内容格式与验证集保持一致，结果文件格式不正确不予计算成绩。该文件命名为：GCRC_advRobust.json。     
2)模型文件：评测使用的模型，所提交模型必须真实可复现，文件命名为：model.zip。        
3)模型说明文档：该文档是docx文件，其中的内容为模型代码运行调试流程，文件命名为： GCRC_advRobust.docx。文件格式示例如下：     
GCRC_advRobust.zip   
&emsp;&emsp;GCRC_advRobust.json   
&emsp;&emsp;model.zip   
&emsp;&emsp;GCRC_advRobust.docx   
## 5.奖项设置
开放赛道和封闭赛道都将评选出如下奖项。由中国中文信息学会计算语言学专委会（CIPS-CL）为获奖队伍提供荣誉证书。

|奖项	|一等奖|	二等奖|	三等奖|
|:-----:|:----:|:----:|:----:|
|数量	|1名|	2名	|3名|
|奖励	|荣誉证书	|荣誉证书	|荣誉证书|

##### 注意事项
+ 由于版权保护问题，GCRC_advRobust数据集只免费提供给用户用于非盈利性科学研究使用，参赛人员不得将数据用于任何商业用途。如果用于商业产品，请联系谭红叶老师，联系邮箱tanhongye@sxu.edu.cn。 
+ 每名参赛选手只能参加一支队伍，一旦发现某选手以注册多个账号的方式参加多支队伍，将取消相关队伍的参赛资格。
+ 数据集的具体内容、范围、规模及格式以最终发布的真实数据集为准。针对测试集，参赛人员不允许执行任何人工标注，并禁止利用正、负对抗选项的构造规则修正答案。
+ 参赛队伍可在参赛期间随时上传测试集的预测结果，智源指数平台每天可提交5次，系统会实时更新当前最新榜单排名情况，严禁参赛团队注册其它账号多次提交。
+ 允许使用公开和选手个人/组织内部的代码、工具、外部数据（从其他渠道获得的标注数据）等，但需要保证参赛结果可以复现。
+ 在数据处理、模型训练和预测等任意阶段使用大模型的参赛队伍只能参加开放赛道的评测。
+ 开放赛道的参赛队伍必须协助评测组织者对测试结果进行验证，包含且不限于复现大模型结果、提供prompt设计策略等，否则成绩无效。
+ 算法与系统的知识产权归参赛队伍所有。要求最终结果排名前6的队伍提供算法代码与系统报告（包括方法说明、数据处理、参考文献和使用开源工具、外部数据等信息）。提交完毕将采用随机交叉检查的方法对各个队伍提交的模型进行检验，如果在排行榜上的结果无法复现，将取消获奖资格。
+ 参赛团队需保证提交作品的合规性，若出现下列或其他重大违规的情况，将取消参赛团队的参赛资格和成绩，获奖团队名单依次顺延。重大违规情况如下：
    + 使用小号、串通、剽窃他人代码等涉嫌违规、作弊行为；
    + 团队提交的材料内容不完整，或提交任何虚假信息；
    + 参赛团队无法就作品疑议进行足够信服的解释说明；    
+ 如需使用本数据集进行课题研究及论文发表，应公开声明使用了山西大学提供的数据，并进行如下引用：    
TAN H, WANG X, JI Y, et al. GCRC: A New Challenging MRC Dataset from Gaokao Chinese for Explainable Evaluation[C]//Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021. 2021: 1319-1330.      
同时发信给tanhongye@sxu.edu.cn，说明相关情况。
### 评测单位
山西大学 合肥工业大学

### 参考文献
[1] Robin Jia and Percy Liang. 2017. Adversarial Examples for Evaluating Reading Comprehension Systems[C]. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing. 2021–2031.   
[2] Zhijing Wu and Hua Xu. 2020. Improving the robustness of machine reading comprehension model with hierarchical knowledge and auxiliary unanswerability prediction[J]. Knowledge Based Systems 203 (2020), 106075.     
[3] Mantong Zhou, Minlie Huang, and Xiaoyan Zhu. 2020. Robust Reading Comprehension With Linguistic Constraints via Posterior Regularization[J]. IEEE Transactions on Audio, Speech, and Language Processing 28 (2020), 2500–2510.    
[4] Ren F, Liu Y, Li B, et al. An Understanding-Oriented Robust Machine Reading Comprehension Model[J]. ACM Transactions on Asian and Low-Resource Language Information Processing, 2022, 22(2): 1-23.   
[5] Wee Chung Gan and Hwee Tou Ng. 2019. Improving the Robustness of Question Answering Systems to Question Paraphrasing[C]. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics. 6065–6075.    
[6] Jakub Náplava, Martin Popel, Milan Straka, and Jana Straková. 2021. Understanding Model Robustness to User-generated Noisy Texts[C]. In Proceedings of the Seventh Workshop on Noisy User-generated Text (W-NUT2021). Association for Computational Linguistics, Online, 340–350. https://doi.org/10.18653/v1/2021.wnut-1.38    
[7] Hongxuan Tang, Hongyu Li, Jing Liu, Yu Hong, Hua Wu, and Haifeng Wang. 2021. DuReader_robust: A Chinese Dataset Towards Evaluating Robustness and Generalization of Machine Reading Comprehension in Real-World Applications[C]. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers). Association for Computational Linguistics, Online, 955–963. https://aclanthology.org/2021.acl-short.120     
[8] Tan H, Wang X, Ji Y, et al. GCRC: A new challenging MRC dataset from Gaokao Chinese for explainable evaluation[C]//Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021. 2021: 1319-1330.    
[9] Cui Y, Che W, Liu T, et al. Revisiting pre-trained models for Chinese natural language processing[C]//Findings of the Association for Computational Linguistics: EMNLP 2020. 2020: 657-668.    


