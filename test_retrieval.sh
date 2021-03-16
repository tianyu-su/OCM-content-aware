#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python -u evaluation_retrieval.py -lf logs/3 -k 0 >  retrieval_logs/test_retri_nondisjoint_k0.log 2>&1
CUDA_VISIBLE_DEVICES=1 python -u evaluation_retrieval.py -pols disjoint -lf logs/4 -k 0 >  retrieval_logs/test_retri_disjoint_k0.log 2>&1
