from cv2 import mean
from load_data import data_loader
from ConvLSTM_U_net_v1 import ConvLSTM_U_net
from Dense_U_net import Dense_U_net
from Unet import multi_unet_model
from tensorflow.python.client import device_lib
from keras.metrics import MeanIoU
import numpy as np
from matplotlib import pyplot as plt


def run(model, name, x_train, x_test , x_valid, y_train, y_test, y_valid, batch,epoch, class_weights, n_class):
    print('***************************************')
    print('Segmentation with ' + name + ' model.')
    print('***************************************')
    model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['accuracy'], loss_weights=class_weights)
    model.summary()
    print("training ",name)
    history = model.fit(x_train, y_train, 
                    batch_size = batch, 
                    verbose = 1, 
                    epochs = epoch, 
                    validation_data=(x_test, y_test), 
                    #class_weight=class_weights,
                    shuffle=False)
    model.save('test_' + name + '.hdf5')
    _, acc = model.evaluate(x_test, y_test)
    Acc_test = acc
    print("Accuracy is = ", (acc * 100.0), "%")

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
    plt.savefig('Loss of ' + name + '.png')

    
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
    plt.savefig('Accuracy of ' + name + '.png')

    #Use validation set to evaluate the performance
    _, acc = model.evaluate(x_valid, y_valid)
    Acc_valid = acc
    print("Validation accuracy is = ", (acc * 100.0), "%")

    model.load_weights('test_' + name + '.hdf5') 
    y_pred=model.predict(x_valid)
    y_pred_argmax=np.argmax(y_pred, axis=3)
    y_valid_argmax=np.argmax(y_valid, axis=3)
    
    IOU_keras = MeanIoU(num_classes=n_class)  
    IOU_keras.update_state(y_valid_argmax, y_pred_argmax)
    mean_IOU = IOU_keras.result().numpy()
    print("Mean IoU =", mean_IOU)


    #To calculate I0U for each class... 
    values = np.array(IOU_keras.get_weights()).reshape(n_class, n_class)
    print(values)
    class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[0,4] + values[1,0]+ values[2,0]+ values[3,0] + values[4,0])
    class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[1,4] + values[0,1]+ values[2,1]+ values[3,1] + values[4,1])
    class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[2,4] + values[0,2]+ values[1,2]+ values[3,2] + values[4,2])
    class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[3,4] + values[0,3]+ values[1,3]+ values[2,3] + values[4,3])
    class5_IoU = values[4,4]/(values[4,4] + values[4,0] + values[4,1] + values[4,2] + values[4,3] + values[0,4]+ values[1,4]+ values[2,4] + values[3,4])
    IoU = [class1_IoU,class2_IoU,class3_IoU,class4_IoU,class5_IoU]

    print("IoU for class1 is: ", class1_IoU)
    print("IoU for class2 is: ", class2_IoU)
    print("IoU for class3 is: ", class3_IoU)
    print("IoU for class4 is: ", class4_IoU)
    print("IoU for class5 is: ", class5_IoU)

    return Acc_test, Acc_valid, mean_IOU, IoU


# if __name__ == "__main__":
#     print(device_lib.list_local_devices())
#     n_class = 5
#     x_train, x_test , x_valid, y_train, y_test, y_valid, class_weights = data_loader(n_class)
#     _, height, width, channel = x_train.shape

#     model_Unet = multi_unet_model(n_classes=n_class, IMG_HEIGHT=height, IMG_WIDTH= width, IMG_CHANNELS= channel)
#     model_Dense_Unet = Dense_U_net(n_class=n_class, img_height=height, img_width= width, img_ch= channel)
#     Acc_test_Unet, Acc_valid_Unet, mean_IOU_Unet, IOU_Unet = run(model_Unet, 'Unet', x_train, x_test , x_valid, y_train, y_test, y_valid)
#     Acc_test_Dense_Unet, Acc_valid_Dense_Unet, mean_IOU_Dense_Unet, IOU_Dense_Unet = run(model_Dense_Unet, 'Dense_Unet', x_train, x_test , x_valid, y_train, y_test, y_valid)
