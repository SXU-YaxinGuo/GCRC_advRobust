# GCRC_advRobust：Adversarial Robustness Evaluation for Chinese Gaokao Reading Comprehension

### Introduction
&emsp;&emsp;The robustness of machine reading comprehension models is a key factor in determining whether this technology can be widely applied in practical settings. With the advancement of technology, existing models have achieved good performance on closed test sets, but their robustness is still poor when it comes to reasoning and decision-making in open, dynamic, and real-world environments. To evaluate the robustness of these models, existing methods mainly involve adding text noise or paraphrasing the questions to interfere with the original tasks. However, these methods have limited effectiveness as they employ relatively simple attack strategies and may only apply to questions with relatively low difficulty, which limits their ability to accurately assess model performance.     
&emsp;&emsp;To enhance the robustness of machine reading comprehension models in complex, real-world adversarial environments, we constructed an adversarial robustness subset GCRC_advRobust based on the "CCL2022-The Dataset of Gaokao Chinese Reading Comprehension for Explainable Evaluation" GCRC and proposed the task of "Adversarial Robustness Evaluation for Chinese Gaokao Reading Comprehension". Unlike the "CCL2022-The Dataset of Gaokao Chinese Reading Comprehension for Explainable Evaluation", which mainly evaluates the model's intermediate reasoning ability, this evaluation includes four types of adversarial attack strategies (keyword perturbation, reasoning logic perturbation, Temporal/spatial perturbation, and Cause-effect perturbation) and focuses on challenging the model's robustness under various adversarial attacks.     
&emsp;&emsp;This evaluation sets up two tracks: open and closed. In the open track, participating teams can use large models such as ChatGPT and ERNIE Bot. In the closed track, the maximum number of model parameters for participating models is 1.5 times that of Bert-large (510M).




### Data Size
Train：6,994 questions；Dev：336 questions；Test：288 questions；

|Splitting|Train|Dev|Test|
| :----- :| :----- :|:----- :|:-----: |
|questions/options|6994/27976|	336/1344 |288/1152|
|keyword Perturbation options|	       -|	          504|	     418|	
|logical Perturbation options	          | -|	          619	|     543|
|Temporal/spatial Perturbation options|   -	|          192	|     172|
|Cause-effect Perturbation options|	   -|29|19|


### Data Format（Dev set and Test set）
Each instance is composed of 
id (id, a string),    
title (title, a string),     
passage (passage, a string),    
question(question, a string),      
options (options, a list, representing the contents of A, B, C, and D, respectively),      
answer(answer,a string),      
positive_options(positive_options, a list, representing the contents of A, B, C, and D, respectively),     
positive_answer(positive_answer,a string),    
negative_question(negative_question, a string),     
negative_options(negative_options, a list, representing the contents of A, B, C, and D, respectively),    
negative_answer(negative_answer,a string).



### Example
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
### Input and output
&emsp;&emsp;Participants must split each sample in the Dev set and Test set into original sample, positive sample  and negative sample as the input of the model, and get three corresponding answers.
&emsp;&emsp;The original sample is <id, title, passage, question, options, answer>; the positive sample is <id, title, passage, question, positive_options, positive_answer>; the negative sample is <id, title ,passage,negative_question,negative_options,negative_answer>.       
&emsp;&emsp;Contestants are required to output the answers of original question, positive question and negative_question.     
&emsp;&emsp;The model output is consistent with the Dev set format.     

### Baseline
The reading comprehension baseline system used in this evaluation is the open source Chinese pre-training model MacBERT.

### Evaluation Code
The prediction results need to be consistent with the format of the Dev set.
Correct file name :GCRC_advrobust.json.
```shell
python eval.py dev_GCRCadvrobust_predict.json dev_GCRCadvrobust.json
```
Participants are required to submit the original_answer, positive_answer and negative_answer.

The evaluation metrics are Acc_0、Acc_1和Acc_2,
The overall evaluation metrics ：Score=0.2\*Acc_0+0.3\*Acc_1+0.5\*Acc_2，
and the output is in dictionary format.
```shell
return {"Acc_0":_, "Acc_1":_, "Acc_2":_, "Score":_}
```
###Results submitted
Participants need to submit three documents:
1) Output result file: This file is a json file encoded in utf-8, and the content format in it is consistent with the verification set. If the format of the result file is incorrect, the score will not be calculated. The file is named: GCRC_advRobust.json.     
2) Model file: the model used in the evaluation, the submitted model must be authentic and reproducible, and the file name is: model.zip.    
3) Model description document: This document is a docx file, which contains the model code running and debugging process, and the file is named: GCRC_advRobust.docx. An example file format is as follows:     
GCRC_advRobust.zip   
&emsp;&emsp;GCRC_advRobust.json   
&emsp;&emsp;model.zip   
&emsp;&emsp;GCRC_advRobust.docx  
### Institutions
Shanxi University, Hefei University of Technology    




