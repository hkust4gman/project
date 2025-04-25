# project

# Dataset
You can put the description of dataset here.

Put your code in the dataset folder.

And we want your dataset to be a **huggingface datasets format**.

# todo:
- spark, prepare the data, please put the url of the dataset or you can download and put the dataset in the ./data directory.


- fine tuning. I am going to fine tune using tokenized data. so you need to provide the tokenized embedding to me.

# Attetion:
Please save some statistic data about your **training**, **data cleaning** etc, like time cost, memory occupation etc.


# Note
- bert + one layer classification.
- Lora or full
- RONG Shuo for dpp (before 21)
- weight & bias (loss, time, num of gpu and diaplay meme)
- megatron and deepspeed (sun and wu, before 23, 24)
- final dataset save as datasets huggingface
- label: 3, 4, 5 for positive and 1, 2 for negative. (so here li should process dataset to a 0 1 label)
- use bert large uncased!
- time: 23 code finished, 26 finish experiment, 27 ppt, 28 report


# Param
learning rate: 1e-5
epoch: 1
how many batch to train(batch count): 10,000
batch size: 200 
so we need 4,000 * 200 data
eval after how many batch in rank0?: math.ceil(batch_count / world_size / 10)
fp: fp32
test: 取样 1000条, to do eval
