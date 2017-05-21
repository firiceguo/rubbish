## Using

```
Usage:
python train.py [test_data_path] [Option] <Settings>
    -h  --help          Get this help
    -b  --batch_size    batch size for training, must be int, default=64
    -t  --timesteps     timesteps for LSTM model, must be int in [1, 95], default=32
    -r  --test_rate     test rate for training, must be float in (0, 1), default=0.05
    -v  --val_rate      validation rate for training, must be float in (0, 1), default=0.02
```

## Environment

Note: If the environment is not match, there will be bugs.

- Python 2.7.13 |Anaconda 4.3.0 (64-bit)

- Using gpu device 0: GeForce GTX 950M (CNMeM is enabled with initial size: 70.0% of memory, CuDNN 4007)

- Theano: 0.8.0

- Keras: 2.0.4

- pygpu: 0.6.2-py2.7-linux-x86_64.egg

- cudnn: 4007

## File Structure

```
├── baseline-xgb
│   ├── README.md
│   ├── test.txt
│   ├── train.txt
│   └── trainxgb.py
├── basline-libsvm
│   ├── DS19test.libsvm
│   ├── DS19train.libsvm
│   ├── DS19train.libsvm.out
│   ├── model
│   └── output
├── dataset
│   └── DS19.libsvm
├── LICENSE
├── README.md
└── src
    ├── covData.py
    ├── features.py
    ├── mymodels.py
    ├── train.py
    └── utils.py
```
