from load_data import data_loader
from ConvLSTM_U_net_v1 import ConvLSTM_U_net
from Dense_U_net import Dense_U_net
from Unet import multi_unet_model
from tensorflow.python.client import device_lib
import tensorflow as tf
import numpy as np
from tensorflow.keras.metrics import MeanIoU
import keras.backend as K
import keras
import matplotlib.pyplot as plt
def run(model, loss_func, batch_size, nepoch, nclass, x_train, x_test, y_train, y_test, class_weights):
    model.compile(optimizer='adam', loss = loss_func, metrics=['accuracy'], loss_weights=class_weights)
    history = model.fit(x_train, y_train, 
                    batch_size = batch_size, 
                    verbose=1, 
                    epochs= nepoch, 
                    validation_data=(x_test, y_test), 
                    #class_weight=class_weights,
                    shuffle=False)
    model.save('UNet_b{0:d}_{1:s}.hdf5'.format(batch_size,loss_func))
    _, acc = model.evaluate(x_test, y_test)
    
    y_pred = model.predict(x_test)
    y_pred_argmax = np.argmax(y_pred, axis=3)
    y_test_argmax = np.argmax(y_test, axis=3)
    IOU_keras = MeanIoU(num_classes=n_class)  
    IOU_keras.update_state(y_test_argmax, y_pred_argmax)
    print("Mean IoU =", IOU_keras.result().numpy())
    values = np.array(IOU_keras.get_weights()).reshape(n_class, n_class)
    class1_IoU_1 = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[0,4] + values[1,0]+ values[2,0]+ values[3,0] + values[4,0])
    class2_IoU_1 = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[1,4] + values[0,1]+ values[2,1]+ values[3,1] + values[4,1])
    class3_IoU_1 = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[2,4] + values[0,2]+ values[1,2]+ values[3,2] + values[4,2])
    class4_IoU_1 = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[3,4] + values[0,3]+ values[1,3]+ values[2,3] + values[4,3])
    class5_IoU_1 = values[4,4]/(values[4,4] + values[4,0] + values[4,1] + values[4,2] + values[4,3] + values[0,4]+ values[1,4]+ values[2,4] + values[3,4])
    IoU = [class1_IoU_1,class2_IoU_1,class3_IoU_1,class4_IoU_1,class5_IoU_1]

    print("IoU for class1 is: ", class1_IoU_1)
    print("IoU for class2 is: ", class2_IoU_1)
    print("IoU for class3 is: ", class3_IoU_1)
    print("IoU for class4 is: ", class4_IoU_1)
    print("IoU for class5 is: ", class5_IoU_1)
    #print("Dense U-Net Accuracy is = ", (dense_acc * 100.0), "%")
    print("U-Net {0:s} Accuracy is = ".format(loss_func), (acc * 100.0), "%")
    
    name = 'U-Net'
    # Save the loss and accuracy of each epoch
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure(1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Test loss')
    plt.title('Training and test loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # plt.show()
    plt.savefig('Loss of ' + name + loss_func + '.png')

    
    val_acc = history.history['val_accuracy']
    acc = history.history['accuracy']
    plt.figure(2)
    plt.plot(epochs, acc, 'y', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Test Accuracy')
    plt.title('Training and test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    # plt.show()
    plt.savefig('Accuracy of ' + name + loss_func + '.png')

    return IoU, acc, history

if __name__ == "__main__":
    batch_size = 8
    nepoch = 500
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    print(device_lib.list_local_devices())
    n_class = 5
    x_train, x_test, y_train, y_test, class_weights = data_loader(n_class)
    _, height, width, channel = x_train.shape
    print(class_weights)
    #model = ConvLSTM_U_net(n_class= n_class, img_height= height, img_width= width, img_ch= channel)
    # Dense U Net
    model = Dense_U_net(n_class=n_class, img_height=height, img_width= width, img_ch= channel)
    loss_func = 'categorical_crossentropy'
    IoU_dense_ce, acc_dense_ce, history_dense_ce = run(model, loss_func, batch_size, nepoch, n_class, x_train, x_test, y_train, y_test, class_weights)
    
    loss_func = 'Poisson'
    IoU_dense_poi, acc_dense_poi, history_dense_poi = run(model, loss_func, batch_size, nepoch, n_class, x_train, x_test, y_train, y_test, class_weights)
    # U Net
    loss_func = 'categorical_crossentropy'
    model = multi_unet_model(n_classes=n_class, IMG_HEIGHT=height, IMG_WIDTH= width, IMG_CHANNELS= channel)
    IoU_unet_ce, acc_unet_ce, history_unet_ce = run(model, loss_func, batch_size, nepoch, n_class, x_train, x_test, y_train, y_test, class_weights)

    loss_func = 'Poisson'
    model = multi_unet_model(n_classes=n_class, IMG_HEIGHT=height, IMG_WIDTH= width, IMG_CHANNELS= channel)
    IoU_unet_poi, acc_unet_poi, history_unet_poi = run(model, loss_func, batch_size, nepoch, n_class, x_train, x_test, y_train, y_test, class_weights)
    
    print("U-Net Poisson Accuracy is = ", (acc_unet_poi * 100.0), "%")
    print("IoU for class1 is: ", IoU_unet_poi[0])
    print("IoU for class2 is: ", IoU_unet_poi[1])
    print("IoU for class3 is: ", IoU_unet_poi[2])
    print("IoU for class4 is: ", IoU_unet_poi[3])
    print("IoU for class5 is: ", IoU_unet_poi[4])
    print('-'*100)
    print("U-Net categorical_crossentropy Accuracy is = ", (acc_unet_ce * 100.0), "%")
    print("IoU for class1 is: ", IoU_unet_ce[0])
    print("IoU for class2 is: ", IoU_unet_ce[1])
    print("IoU for class3 is: ", IoU_unet_ce[2])
    print("IoU for class4 is: ", IoU_unet_ce[3])
    print("IoU for class5 is: ", IoU_unet_ce[4])
    print('-'*100)
    print("Dense U-Net Poisson Accuracy is = ", (acc_dense_poi * 100.0), "%")
    print("IoU for class1 is: ", IoU_dense_poi[0])
    print("IoU for class2 is: ", IoU_dense_poi[1])
    print("IoU for class3 is: ", IoU_dense_poi[2])
    print("IoU for class4 is: ", IoU_dense_poi[3])
    print("IoU for class5 is: ", IoU_dense_poi[4])
    print('-'*100)
    print("Dense U-Net categorical_crossentropy Accuracy is = ", (acc_dense_ce * 100.0), "%")
    print("IoU for class1 is: ", IoU_dense_ce[0])
    print("IoU for class2 is: ", IoU_dense_ce[1])
    print("IoU for class3 is: ", IoU_dense_ce[2])
    print("IoU for class4 is: ", IoU_dense_ce[3])
    print("IoU for class5 is: ", IoU_dense_ce[4])

    name = 'U-Net'
    # Save the loss and accuracy of each epoch
    loss_dense_ce = history_dense_ce.history['loss']
    val_loss_dense_ce = history_dense_ce.history['val_loss']
    epochs = range(1, len(loss_dense_ce) + 1)
    loss_unet_ce = history_dense_ce.history['loss']
    val_loss_unet_ce = history_dense_ce.history['val_loss']
    loss_dense_poi = history_dense_ce.history['loss']
    val_loss_dense_poi = history_dense_ce.history['val_loss']
    loss_unet_poi = history_dense_ce.history['loss']
    val_loss_unet_poi = history_dense_ce.history['val_loss']
    plt.figure(1)
    plt.plot(epochs, loss_dense_ce, label='Dense CE Training')
    plt.plot(epochs, val_loss_dense_ce, label='Dense CE Test')
    plt.plot(epochs, loss_unet_ce, label='Unet CE Training')
    plt.plot(epochs, val_loss_unet_ce, label='Unet CE Test')
    plt.plot(epochs, loss_dense_poi, label='Dense POI Training')
    plt.plot(epochs, val_loss_dense_poi, label='Dense POI Test')
    plt.plot(epochs, loss_unet_poi, label='Unet POI Training')
    plt.plot(epochs, val_loss_unet_poi, label='Unet POI Test')
    plt.title('Summary of Training and test loss for different variation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # plt.show()
    plt.savefig('Summary of Loss.png')

    
    val_acc_dense_ce = history_dense_ce.history['val_accuracy']
    acc_dense_ce = history_dense_ce.history['accuracy']
    acc_unet_ce = history_dense_ce.history['accuracy']
    val_acc_unet_ce = history_dense_ce.history['val_accuracy']
    acc_dense_poi = history_dense_ce.history['accuracy']
    val_acc_dense_poi = history_dense_ce.history['val_accuracy']
    acc_unet_poi = history_dense_ce.history['accuracy']
    val_acc_unet_poi = history_dense_ce.history['val_accuracy']
    plt.figure(2)
    plt.plot(epochs, acc_dense_ce, label='Dense CE Training')
    plt.plot(epochs, val_acc_dense_ce, label='Dense CE Test')
    plt.plot(epochs, acc_unet_ce, label='Unet CE Training')
    plt.plot(epochs, val_acc_unet_ce, label='Unet CE Test')
    plt.plot(epochs, acc_dense_poi, label='Dense POI Training')
    plt.plot(epochs, val_acc_dense_poi, label='Dense POI Test')
    plt.plot(epochs, acc_unet_poi, label='Unet POI Training')
    plt.plot(epochs, val_acc_unet_poi, label='Unet POI Test')
    plt.title('Summary of Training and test Accuracy for different variation')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    # plt.show()
    plt.savefig('Summary of Accuracy.png')