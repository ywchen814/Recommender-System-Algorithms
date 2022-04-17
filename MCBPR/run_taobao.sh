python3 run.py -d 64 -lr 0.001 -reg 1e-05 1e-05 1e-05 -epochs 50 -optimizer adam -data ../Data/Taobao/train_tb.csv -test_data ../Data/Taobao/test_tb.csv -eval_results ./eval_results/Taobao/
python3 run.py -d 128 -lr 0.001 -reg 1e-05 1e-05 1e-05 -epochs 50 -optimizer adam -data ../Data/Taobao/train_tb.csv -test_data ../Data/Taobao/test_tb.csv -eval_results ./eval_results/Taobao/
