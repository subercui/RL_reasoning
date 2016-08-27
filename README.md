# RL_reasoning
Authors:Haotian Cui, Yukun Yan, Xianggen Liu

## data preparing
* preprocess.py: replace the end char "." and add "|||"
* stream.py: 
1.add "<s></s>" to the phrase
2.build the process class, then map to the index, make up the vector representation
finally,return a list of vectors.
namely, 
for facts, question, label in data_class.data_stream():#facts,question:length->2,label -> 1,such as:
     facts[0]:(shape:(5,10))
 [[ 9 21  0  2 14 20 21 16 11  5]
  [ 9 21 19  2 17 20 21  0 11  5]
  [ 9 21 18  2 10 20 21  3 11  5]
  [ 9 21  0  2 10 20 21 18 11  5]
  [ 9 21  6  2 17 20 21 18 11  5]]
     question[0]:shape:(1,13)
    print("-------------------")

## fda
