CUDA_VISIBLE_DEVICES=5 python -u train_bench.py | tee mine_logs/nondisjoint.log; \
CUDA_VISIBLE_DEVICES=5 python -u train_bench.py --polyvore-split disjoint | tee mine_logs/disjoint.log

