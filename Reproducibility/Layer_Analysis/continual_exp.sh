#!/bin/bash

seeds="0 1 2 3 4 5 6 7"
list_heads_01=" Linear Linear_no_bias"
list_heads_001=" CosLayer WeightNorm OriginalWeightNorm"
list_heads_WO_lr="SLDA MeanLayer MedianLayer KNN"
cd ../..
for seed in $seeds ;do

for head in list_heads_01 ;do
./script_continual.sh --lr 0.1 --seed $seed --OutLayer $head
./script_continual.sh --lr 0.1 --seed $seed --OutLayer $head_masked --masked_out single
./script_continual.sh --lr 0.1 --seed $seed --OutLayer $head_masked --masked_out group
done #head

for head in list_heads_001 ;do
./script_continual.sh --lr 0.01 --seed $seed --OutLayer $head
./script_continual.sh --lr 0.01 --seed $seed --OutLayer $head_masked --masked_out single
./script_continual.sh --lr 0.01 --seed $seed --OutLayer $head_masked --masked_out group
done #head


for head_WO_lr in $list_heads_WO_lr ;do
./script_continual.sh --seed $seed --OutLayer $head_WO_lr
done #head_WO_lr

done #seed
