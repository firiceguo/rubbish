## Xgboost

We use [Xgboost](https://xgboost.readthedocs.io/en/latest/get_started/) as the baseline.

### Source code

```bash
python trainxgb.py
```

### Output

```python
param = {'max_depth': 6, 'eta': 0.05, 'silent': 0, 'objective': 'multi:softmax', 'nthread': 4, 'num_class': 7}
num_round = 100
```

```bash
[13:11:24] 8036x97 matrix with 771270 entries loaded from train.txt
[13:11:24] 890x97 matrix with 85350 entries loaded from test.txt
[13:11:24] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 42 extra nodes, 0 pruned nodes, max_depth=6

...

[13:11:48] src/tree/updater_prune.cc:74: tree pruning end, 1 roots, 66 extra nodes, 0 pruned nodes, max_depth=6

RMSE = 1.391790 
Correct Rate = 0.787640
701 / 890

Process finished with exit code 0
```
