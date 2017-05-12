## Using

'''bash
python train.py [option] <value>

Options:
    -m  --model:        lstm(default)/mlp/lstm_stack
    -b  --batch_size:   64 for default
    -e  --epoches:      5 for default
    -t  --timesteps:    config if using lstm_stack, 16 for default
    -r  --test_rate:    the percentage for test, 0.2 for default
    -v  --val_rate:     the percentage for validation, 0.1 for default
'''

## Environment

- Python 2.7.13 |Anaconda 4.3.0 (64-bit)

- Using gpu device 0: GeForce GTX 950M (CNMeM is enabled with initial size: 70.0% of memory, CuDNN 4007)

- Theano: 0.8.0

- Keras: 2.0.4

- pygpu: 0.6.2-py2.7-linux-x86_64.egg

## Bugs:

1. Can't use Pooling layers

    Keras 2.0+ don't support theano-0.8: `TypeError: pool_2d() got an unexpected keyword argument 'ws'`
