from keras.layers import Input, Dense, Conv2D, Flatten, Reshape
from keras.models import Model
from keras.optimizers import RMSprop, Adam

def create_test_model(In=(10,), Out=1, learning_rate=1e-3):
    net_input = Input(shape=In, name='Input')
    x = Dense(Out, activation='sigmoid', name='Output')(net_input)
    model = Model(inputs=net_input, outputs=x)
    adam = Adam(lr=learning_rate)
    model.compile(optimizer=adam, loss='mse')
    model._make_train_function()
    return model