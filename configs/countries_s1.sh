#!/usr/bin/env bash

data_input_dir="datasets/data_preprocessed/countries_S1/"
vocab_dir="datasets/data_preprocessed/countries_S1/vocab"
total_iterations=1000
path_length=2
hidden_size=25
embedding_size=25
batch_size=256
beta=0.05
Lambda=0.05
use_entity_embeddings=1
train_entity_embeddings=1
train_relation_embeddings=1
base_output_dir="/data/xuht/minerva/output/countries_s1/"
load_model=0
model_load_dir="null"

gpu_id=3
CUDA_VISIBLE_DEVICES=$gpu_id python code/model/trainer.py --base_output_dir $base_output_dir --path_length $path_length --hidden_size $hidden_size --embedding_size $embedding_size \
    --batch_size $batch_size --beta $beta --Lambda $Lambda --use_entity_embeddings $use_entity_embeddings \
    --train_entity_embeddings $train_entity_embeddings --train_relation_embeddings $train_relation_embeddings \
    --data_input_dir $data_input_dir --vocab_dir $vocab_dir --model_load_dir $model_load_dir --load_model $load_model --total_iterations $total_iterations





