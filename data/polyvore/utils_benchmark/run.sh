#################################   EXTRACT FEATURE #################################
python extract_features_batch.py

#################################   CREATE DATASET #################################
python create_dataset.py --polyvore-split nondisjoint --phase train
python create_dataset.py --polyvore-split nondisjoint --phase valid
python create_dataset.py --polyvore-split nondisjoint --phase test

python create_dataset.py --polyvore-split disjoint --phase train
python create_dataset.py --polyvore-split disjoint --phase valid
python create_dataset.py --polyvore-split disjoint --phase test