# DeepState: Selecting Massive Test Suites to Enhance the Robustness of Recurrent Neural Networks

DeepState has conducted experiments on the combination of 6 models and data sets. To illustrate the usage of the code of this project, the example of testing LSTM model trained with MNIST data set is illustrated as follows.

## Environment

python=3.6

```sh
pip install -r requirements.txt
```

## Preparing an RNN model as the test object

```sh
python3 ./RNNModels/mnist_demo/mnist_lstm.py
```

After the training is completed, the output is as follows, and the trained model will be saved in the `./RNNModels/mnist_demo/models/mnist_lstm.h5`.

```
Epoch 20/20
60000/60000 [==============================] - 11s 178us/step - loss: 0.0103 - accuracy: 0.9970 - val_loss: 0.0569 - val_accuracy: 0.9866
```

## Preparing the data set for selection

First, we need to generate the augmented data:

```sh
python3 ./gen_data/dau_mnist.py
```

For evaluating RQ1, we generate 30 different dataset for selection:

```sh
python3 ./gen_data/gen_toselect_dataset.py -dataset "mnist"   # for RQ1 & RQ2
```

For evaluation RQ3, we generate the dataset for selection and retraining and the test set for evaluation:

```sh
python3 ./gen_data/gen_toselect_retrain.py -dataset "mnist"   # for RQ3
```

## Generating the abstract model used for calculating the DeepStellar-coverage 

The coverage calculation of DeepStellar requires an abstract model to be generated in advance. This part of the code comes from [DeepStellar](https://github.com/xiaoningdu/deepstellar) 's open source code.

```sh
python3 ./abstraction_runner.py -test_obj "mnist_lstm"
```

## RQ1

```sh
python3 rq1.py -dl_model "./RNNModels/mnist_demo/models/mnist_lstm.h5" -model_type "lstm" -dataset "mnist"
```

The results will be saved in `.\exp_results\rq1`

## RQ2

```sh
python3 rq2.py -dl_model "./RNNModels/mnist_demo/models/mnist_lstm.h5" -model_type "lstm" -dataset "mnist"
```

The results will be saved in `.\exp_results\rq2`

## RQ3

```sh
python3 rq3.py -dl_model "./RNNModels/mnist_demo/models/mnist_lstm.h5" -model_type "lstm" -dataset "mnist"
```

The results will be saved in `.\exp_results\rq3`