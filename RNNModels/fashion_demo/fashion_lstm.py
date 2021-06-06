import keras
import os
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Input, Lambda, LSTM, Dense, Dropout
from keras.models import load_model
from keras.models import Model
import numpy as np
import os
from keras.callbacks import ModelCheckpoint
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# loss: 0.1949 - accuracy: 0.9281 - val_loss: 0.3311 - val_accuracy: 0.8855


class FashionLSTMClassifier:
    def __init__(self):
        # Classifier
        self.time_steps = 28  # timesteps to unroll
        self.n_units = 256  # hidden LSTM units
        self.n_inputs = 28  # rows of 28 pixels (a fashion img is 28x28)
        self.n_classes = 10  # mnist classes/labels (0-9)
        self.batch_size = 128  # Size of each batch
        self.n_epochs = 20

    def create_model(self):
        self.model = Sequential()
        self.model.add(LSTM(self.n_units, input_shape=(self.time_steps, self.n_inputs), name="lstm"))
        self.model.add(Dropout(0.2, name='drop'))
        self.model.add(Dense(self.n_classes, activation='softmax', name="dense"))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # self.model.summary()

    def load_hidden_state_model(self, model_path):
        """
        return the rnn model with return_sequence enabled.
        """
        input = Input(shape=(self.time_steps, self.n_inputs))
        lstm = LSTM(self.n_units, input_shape=(self.time_steps, self.n_inputs), return_sequences=True, name="lstm")(input)
        last_timestep = Lambda(lambda x: x[:, -1, :])(lstm)
        drop = Dropout(0.2, name='drop')(last_timestep)
        dense = Dense(self.n_classes, activation='softmax', name="dense")(drop)
        model = Model(inputs=input, outputs=[dense, lstm])
        model.load_weights(model_path, by_name=True)
        # self.model = model
        return model

    def reload_dense(self, model_path):
        input = Input(shape=(self.n_units,))
        drop = Dropout(0.2, name='drop')(input)
        dense = Dense(self.n_classes, activation='softmax', name="dense")(drop)
        model = Model(inputs=input, outputs=dense)
        model.load_weights(model_path, by_name=True)
        return model

    def train(self, save_path):
        self.create_model()
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

        x_train = self.input_preprocess(x_train)
        x_test = self.input_preprocess(x_test)

        y_test = keras.utils.to_categorical(y_test, num_classes=10)
        y_train = keras.utils.to_categorical(y_train, num_classes=10)

        checkpoint = ModelCheckpoint(filepath=os.path.join(save_path, "fashion_lstm.h5"),
                                     monitor='val_acc', mode='auto', save_best_only='True')
        self.model.fit(x_train, y_train, validation_data=(x_test, y_test),
                       batch_size=self.batch_size, epochs=self.n_epochs, shuffle=False, callbacks=[checkpoint])

        os.makedirs(save_path, exist_ok=True)
        self.model.save(os.path.join(save_path, "fashion_lstm.h5"))

    def retrain(self, X_selected, Y_selected, X_val, Y_val, save_path):
        self.create_model()
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        Xa_train = np.concatenate([X_selected, x_train])
        Ya_train = np.concatenate([Y_selected, y_train])

        Xa_train = self.input_preprocess(Xa_train)
        X_val = self.input_preprocess(X_val)
        Ya_train = keras.utils.to_categorical(Ya_train, num_classes=10)
        Y_val = keras.utils.to_categorical(Y_val, num_classes=10)

        checkpoint = ModelCheckpoint(filepath=save_path, monitor='val_acc', mode='auto', save_best_only='True')
        self.model.fit(Xa_train, Ya_train, validation_data=(X_val, Y_val),
                       batch_size=self.batch_size, epochs=self.n_epochs, shuffle=False, callbacks=[checkpoint])

        self.model.save(save_path)

    def retrain1(self, ori_model_path, X_selected, Y_selected, X_val, Y_val, save_path):
        self.create_model()
        self.model.load_weights(ori_model_path, by_name=True)
        Xa_train = self.input_preprocess(X_selected)
        X_val = self.input_preprocess(X_val)
        Ya_train = keras.utils.to_categorical(Y_selected, num_classes=10)
        Y_val = keras.utils.to_categorical(Y_val, num_classes=10)
        checkpoint = ModelCheckpoint(filepath=save_path, monitor='val_acc', mode='auto', save_best_only='True')
        self.model.fit(Xa_train, Ya_train, validation_data=(X_val, Y_val),
                       batch_size=self.batch_size, epochs=15, shuffle=False, callbacks=[checkpoint])

        self.model.save(save_path)

    def evaluate_retrain(self, retrain_model_path, ori_model_path, x_val, y_val):
        x_val = self.input_preprocess(x_val)
        y_val = keras.utils.to_categorical(y_val, num_classes=10)

        retrain_model = load_model(retrain_model_path)
        ori_model = load_model(ori_model_path)
        retrain_acc = retrain_model.evaluate(x_val, y_val)[1]
        ori_acc = ori_model.evaluate(x_val, y_val)[1]
        return retrain_acc - ori_acc

    def input_preprocess(self, data):
        data = data.reshape(data.shape[0], self.time_steps, self.n_inputs)
        data = data.astype('float32')
        data /= 255
        return data

    def profile_train_data(self, model, profile_save_path):
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train = self.input_preprocess(x_train)
        output = model.predict(x_train)
        cls = np.argmax(output[0], axis=1)
        correct_idx = np.where(cls == y_train)[0]
        os.makedirs(profile_save_path, exist_ok=True)
        states_correct = output[1][correct_idx]
        np.save(os.path.join(profile_save_path, "states_profile.npy"), states_correct)

    def get_state_profile(self, inputs, model):
        inputs = self.input_preprocess(inputs)
        output = model.predict(inputs)
        return output[1]


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    save_path = "./models"
    lstm_classifier = FashionLSTMClassifier()

    # train an rnn model
    lstm_classifier.create_model()
    lstm_classifier.train(save_path)

    # # Load a trained model with return_sequence enabled.
    # x_test = lstm_classifier.input_preprocess(x_test)
    # print(x_test[0].shape)
    #
    # model = lstm_classifier.load_hidden_state_model(os.path.join(save_path, "fashion_lstm.h5"))
    # print(model.predict(np.array([x_test[0]]))[0])
    # # lstm_classifier.profile_train_data(profile_path)
    #
    # m1 = load_model(os.path.join(save_path, "fashion_lstm.h5"))
    # print(m1.predict(np.array([x_test[0]])))
