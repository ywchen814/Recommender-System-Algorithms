# Multi Channel BPR (MCBPR) Pytorch Implementation

This is a Pytorch implementation of Multi Channel BPR (MCBPR)
[Bayesian Personalized Ranking with Multi-channel User Feedback - Loni et al. (2016)](https://dl.acm.org/doi/pdf/10.1145/2959100.2959163). This model is an improved version of BPR for the usage of single behavior type to multi-behavior type.

## Run the code
A running example
```
python3 model.py -dataset "Beibei" -factor_num 64 -beta 1 -lr 0.001 -epoch 50
```
You can execute `python3 model.py -h` to see the all settings by following:
- `-factor_num` INT       latent feature dimension
- `-beta` FLOAT           share of unobserved within negative
                        feedback
- `-lr` FLOAT             learning rate
- `-lamda` FLOAT          regularization parameters
- `-rel_order` INT INT INT
                        the order of importance of interactions
                        (buy(0), pv(1), cart(2))
- `-rel_weight` float float float
                        the sample weight of buy, pv, cart
- `-neg_num` NEG_NUM     sample negative items for training
- `-seed` INT             seed for random number generators
- `-batch_size` BATCH_SIZE
                        batch size for training
- `-epoch` INT            no. of training epochs
- `-sample_mode` STR      negative item sampling modes (uniform or
                        multi_level)
- `-Ks` [KS]              compute metrics@top_k
- `-data_path` STR        path to read the dataset
- `-dataset` STR          dataset (Beibei, Taobao or rocket)
- `-results` STR          path to write results into

## Contributors
- [chenchongthu](https://github.com/chenchongthu/GHCF) (for the evaluation code)