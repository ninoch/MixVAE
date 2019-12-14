python mixhop_trainer.py --dataset_name=$1 --adj_pows=0:20:6,1:20:6,2:20:6 --hidden_dims_csv=60,18 \
  --learn_rate=0.1 --lr_decrement_every=40 --early_stop_steps=200 \
  --input_dropout=0.5 --layer_dropout=0.9 --l2reg=5e-2 \
  --retrain > results/log/$1.txt
