
1.下载hfl-chinese-macbert-large模型文件（包括bert_config.json，pytorch_model.bin,vocab.txt）放到hfl-chinese-macbert-large文件夹中,下载链接：https://huggingface.co/hfl
2.运行create_csv.py 获得训练集、验证集、测试集的.csv格式作为输入

3. 训练
python3 run_four.py \
  --task_name=gaokao \
  --do_train=true \
  --do_eval=true \
  --data_dir=./data/advRobust_data/\
  --bert_model=./bert-base-chinese/ \
  --max_seq_length=384 \
  --train_batch_size=16 \
  --learning_rate=1e-5 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs=3.0 \
  --output_dir=./data/BERT_output/

4. 测试
python3 run_four.py \
  --task_name=gaokao \
  --do_predict=True \
  --data_dir=./data/advRobust_data/\
  --bert_model=./bert-base-chinese/ \
  --max_seq_length=384 \
  --train_batch_size=16 \
  --learning_rate=1e-5 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs=3.0 \
  --output_dir=./data/BERT_output/