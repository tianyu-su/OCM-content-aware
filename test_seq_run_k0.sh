CUDA_VISIBLE_DEVICES=5 python -u test_fitb.py -lf logs/3 -k 0 >  mine_logs/test_fitb_nondisjoint_k0.log 2>&1
CUDA_VISIBLE_DEVICES=5 python -u test_fitb.py -pols disjoint -lf logs/4 -k 0 >  mine_logs/test_fitb_disjoint_k0.log 2>&1

CUDA_VISIBLE_DEVICES=5 python -u test_compatibility.py -lf logs/3 -k 0 >  mine_logs/test_auc_nondisjoint_k0.log 2>&1
CUDA_VISIBLE_DEVICES=5 python -u test_compatibility.py -pols disjoint -lf logs/4 -k 0 >  mine_logs/test_auc_disjoint_k0.log 2>&1


