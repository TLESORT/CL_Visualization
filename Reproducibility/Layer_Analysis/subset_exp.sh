#!/bin/bash



seeds="0 1 2 3 4 5 6 7"
list_subsets="100 200 500 1000"
list_heads=" Linear CosLayer Linear_no_bias WeightNorm OriginalWeightNorm"
list_heads_WO_lr="SLDA MeanLayer MedianLayer KNN"

for seed in $seeds ;do

for subset in $list_subsets ;do
for head in $list_heads ;do
./script_subsets.sh --lr 0.1 --subset $subset --seed $seed --OutLayer $head
./script_subsets.sh --lr 0.1 --subset $subset --seed $seed --OutLayer $head --masked_out single
done #head


for head_WO_lr in $list_heads_WO_lr ;do
./script_subsets.sh --subset $subset --seed $seed --OutLayer $head_WO_lr
done #head_WO_lr

done #subset
done #seed
