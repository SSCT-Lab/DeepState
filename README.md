# DeepState: Selecting Massive Test Suites to Enhance the Robustness of Recurrent Neural Networks

DeepState has conducted experiments on the combination of 6 models and data sets. To illustrate the usage of the code of this project, the example of testing LSTM model trained with MNIST data set is illustrated as follows.

## Environment

python=3.6

```sh
pip install -r requirements.txt
```

## Preparing an RNN model as the test object

```sh
cd RNNModels/mnist_demo
python3 mnist_lstm.py -type "train"
```

After the training is completed, the output is as follows, and the trained model will be saved in the `./RNNModels/mnist_demo/models/mnist_lstm.h5`.

```
Epoch 20/20
54000/54000 [==============================] - 10s 188us/step - loss: 0.0112 - accuracy: 0.9963 - val_loss: 0.0548 - val_accuracy: 0.9878
```

## Preparing the data set for selection

For evaluating RQ1, we generate 30 different dataset for selection:

```sh
# generate the augmented data for selection
cd ../../gen_data/gen_test_dataset
python3 dau_mnist.py
python3 gen_toselect_dataset.py -dataset "mnist"   # for RQ1 & RQ2
```

For evaluation RQ3, we generate the dataset for selection and retraining and the test set for evaluation:

```sh
# generate the augmented data for selection and retraining
cd ../gen_retrain_dataset
python3 dau_mnist.py
python3 gen_retrain.py -dataset "mnist"   # for RQ3
```

If you feel that the steps to generate the augmented data are troublesome, or you just want to reproduce the results, you can download the data we generated [here](https://drive.google.com/drive/folders/1jjtwb44aX_yeeYktlyCpPUoDXeZ2mHYR?usp=sharing).

## Generating the abstract model used for calculating the DeepStellar-coverage 

The coverage calculation of DeepStellar requires an abstract model to be generated in advance. This part of the code comes from [DeepStellar](https://github.com/xiaoningdu/deepstellar) 's open source code.

```sh
cd ../..
python3 ./abstraction_runner.py -test_obj "mnist_lstm"
```

## RQ1

```sh
python3 rq1.py -dl_model "./RNNModels/mnist_demo/models/mnist_lstm.h5" -model_type "lstm" -dataset "mnist"
```

The results will be saved in `./exp_results/rq1` .

## RQ2

```sh
python3 rq2.py -dl_model "./RNNModels/mnist_demo/models/mnist_lstm.h5" -model_type "lstm" -dataset "mnist"
```

The results will be saved in `./exp_results/rq2` .

## RQ3

First, we need to prepare the model before retraining:

```sh
cd RNNModels/mnist_demo
python3 mnist_lstm.py -type "retrain"
```

Then, we can evaluate the RQ3:

```sh
python3 rq3.py -dl_model "./RNNModels/mnist_demo/models/mnist_lstm_ori.h5" -model_type "lstm" -dataset "mnist"
```

The results will be saved in `./exp_results/rq3` .
