from load_data import data_loader
from ConvLSTM_U_net_v1 import ConvLSTM_U_net
from Dense_U_net import Dense_U_net
from tensorflow.python.client import device_lib

if __name__ == "__main__":
    print(device_lib.list_local_devices())
    n_class = 5
    x_train, x_test , x_valid, y_train, y_test, y_valid = data_loader(n_class)
    _, height, width, channel = x_train.shape
    #model = ConvLSTM_U_net(n_class=n_class, img_height=height, img_width= width, img_ch= channel)
    model = Dense_U_net(n_class=n_class, img_height=height, img_width= width, img_ch= channel)
    model.summary()
    history = model.fit(x_train, y_train, 
                    batch_size = 16, 
                    verbose=1, 
                    epochs=50, 
                    validation_data=(x_test, y_test), 
                    #class_weight=class_weights,
                    shuffle=False)
    model.save('test.hdf5')
    _, acc = model.evaluate(x_test, y_test)
    print("Accuracy is = ", (acc * 100.0), "%")