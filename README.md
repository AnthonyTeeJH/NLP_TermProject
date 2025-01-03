# Project description
## Motivation
* Avoid the need of generating training data manually
* Looking for "magic prompt(s)“, aside from using "lucrarea" 
![Picture1](https://github.com/user-attachments/assets/a1561328-ff26-4005-824e-b96ebba2e69e)

## Framework
Using Genetic Algorithm (GA) to optimize the auxiliary embedding vector
* Maximize the average similarity to the set of public mean prompts

## TODO
* Public mean prompt embedding vectors 需透過 sentence-t5-base 模型事先計算出來 
* 計算 'lucrarea' embedding vector 與 public mean prompt embedding vectors 之間的 similarit -> 驗證尋找與 public mean prompt embedding vectors 相似的 embedding vector 作為 auxiliary prompt 的想法是否合理。
* Chromosome representation: embedding vector length (fixed/ varied), 值域? (refer to t5-base model)
* Evaluation of each individual: 與'lucrare' embedding vector 之間的 mean sharpened cosine similarity -> p
* GA optimization completed, obtain auxiliary prompt embedding vector (genotype): 反向工程，將 auxiliary prompt embedding vector mapping to auxiliary prompt (phenotype). HOW?
* Submission, obtain LB score: 製作 submission 用的 codeblock (1st public mean prompt + auxiliary prompt)
* Experiment:
1. fitness function: maximize similarity with 'lucrarea' or maximize similarity with public mean prompts?
2. final prompt = 1st public mean prompt + auxiliary prompt? 是否有其他組合？一定是第一名 public mean prompt?
2. GA fitness function 的正確性，與所有 public mean prompt 的平均相似度？還是按照排名取權重？為什麼auxiliary prompt = 與 public mean prompt 相似的 prompt?
3. ...


