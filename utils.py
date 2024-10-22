import time
import numpy as np
import random

import math
import matplotlib.pyplot as plt


def load_csv(path):
    '''
    Load the CSV form of MNIST data without any external library
    :param path: the path of the csv file
    :return:
        data: A list of list where each sub-list with 28x28 elements
              corresponding to the pixels in each image
        labels: A list containing labels of images
    '''
    data = []
    labels = []
    with open(path, 'r') as fp:
        images = fp.readlines()
        images = [img.rstrip() for img in images]

        for img in images:
            img_as_list = img.split(',')
            y = int(img_as_list[0]) # first entry as label
            x = img_as_list[1:]
            x = [int(px) / 255 for px in x]
            data.append(x)
            labels.append(y)
    return data, labels


def load_mnist_trainval():
    """
    Load MNIST training data with labels
    :return:
        train_data: A list of list containing the training data
        train_label: A list containing the labels of training data
        val_data: A list of list containing the validation data
        val_label: A list containing the labels of validation data
    """
    # Load training data
    print("Loading training data...")
    data, label = load_csv('./data/mnist_train.csv')
    assert len(data) == len(label)

    print("Training data loaded with {count} images".format(count=len(data)))
    # data = np.array(data1)
    # label = np.array(label1)
    # split training/validation data
    train_data = []
    train_label = []
    val_data = []
    val_label = []
    #############################################################################
    # TODO:                                                                     #
    #    1) Split the entire training set to training data and validation       #
    #       data. Use 80% of your data for training and 20% of your data for    #
    #       validation                                                          #
    #############################################################################

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # l = np.asarray(label)
    # one-hot encode labels
    # digits = 10
    # examples = l.shape[0]
    # label = l.reshape(1, examples)
    # label = np.eye(digits)[label.astype('int32')]
    # label1 = label.reshape(digits, examples)

    # split, reshape, shuffle
    # m = int(0.8 * len(data))
    # print(" SHAPE ",X.shape[0])
    # m_test = X.shape[0] - m
    # print(m)
    # X_train, X_test = X[:m], X[m:]
    # Y_train, Y_test = y[:,:m], y[:,m:]
    # # shuffle_index = np.random.permutation(m)
    # # X_train, Y_train = X_train[:, shuffle_index], Y_train[:, shuffle_index]
    #
    # train_data = X_train
    # train_label = Y_train
    # val_data = X_test
    # val_label = Y_test

    train_pct_index = int(0.8 * len(data))
    train_data, val_data = data[:train_pct_index], data[train_pct_index:]
    train_label, val_label = label[:train_pct_index], label[train_pct_index:]

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return train_data, train_label, val_data, val_label

def load_mnist_test():
    """
        Load MNIST testing data with labels
        :return:
            data: A list of list containing the testing data
            label: A list containing the labels of testing data
        """
    # Load training data
    print("Loading testing data...")
    data, label = load_csv('./data/mnist_test.csv')
    assert len(data) == len(label)
    print("Testing data loaded with {count} images".format(count=len(data)))

    return data, label



# def generate_batched_data(data, label, batch_size=32, shuffle=False, seed=None):
#     '''
#     Turn raw data into batched forms
#     :param data: A list of list containing the data where each inner list contains 28x28
#                  elements corresponding to pixel values in images: [[pix1, ..., pix768], ..., [pix1, ..., pix768]]
#     :param label: A list containing the labels of data
#     :param batch_size: required batch size
#     :param shuffle: Whether to shuffle the data: true for training and False for testing
#     :return:
#         batched_data: A list whose elements are batches of images.
#         batched_label: A list whose elements are batches of labels
#     '''
#     batched_data=[]
#     batched_label = []
#     data = np.asarray(data)
#     label = np.asarray(label)
#
#     permutation = np.random.permutation(data.shape[0])
#     X_train_shuffled = data[ permutation,:]
#     Y_train_shuffled = label[permutation]
#     batches = math.ceil(data.shape[0] / batch_size)
#     for j in range(batches):
#
#         begin = j * batch_size
#         end = min(begin + batch_size, data.shape[0] - 1)
#         X = X_train_shuffled[:, begin:end]
#         batched_data.append(X)
#         Y = Y_train_shuffled[begin:end]
#         batched_data.append(Y)
#
#     # if(shuffle):
#     #     random.shuffle(data)
#     #     random.shuffle(label)
#     #     data = np.asarray(data)
#     #     label = np.asarray(label)
#     # else:
#     #     data = np.asarray(data)
#     #     label = np.asarray(label)
#     #
#     # batches = math.ceil(data.shape[0] / batch_size)
#     # batched_data =  [data[i*batch_size:(i+1)*batch_size] for i in range(batches)]
#     # batched_label =  [label[i*batch_size:(i+1)*batch_size] for i in range(batches)]
#     #############################################################################
#     #                              END OF YOUR CODE                             #
#     #############################################################################
#     return batched_data, batched_label

def shuffle_list(*ls):
    l =list(zip(*ls))

    random.shuffle(l)
    return zip(*l)

def generate_batched_data(data1, label1, batch_size=32, shuffle=False, seed=None):
    '''
    Turn raw data into batched forms
    :param data: A list of list containing the data where each inner list contains 28x28
                 elements corresponding to pixel values in images: [[pix1, ..., pix768], ..., [pix1, ..., pix768]]
    :param label: A list containing the labels of data
    :param batch_size: required batch size
    :param shuffle: Whether to shuffle the data: true for training and False for testing
    :return:
        batched_data: A list whose elements are batches of images.
        batched_label: A list whose elements are batches of labels
    '''
    # print(" BATCHESIZE ",batch_size)
    if(seed):
        random.seed(seed)

    if(shuffle):
        data , label = shuffle_list(data1,label1)
        data = np.asarray(data)
        label = np.asarray(label)
    else:
        data = np.asarray(data1)
        label = np.asarray(label1)

    batches = math.ceil(data.shape[0] / batch_size)
    # print(" BATCHES ",batches)
    batched_data =  [data[i*batch_size:(i+1)*batch_size] for i in range(batches)]
    batched_label =  [label[i*batch_size:(i+1)*batch_size] for i in range(batches)]


    # print(" Baatch data ",len(batched_data))
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return batched_data, batched_label


def train(epoch, batched_train_data, batched_train_label, model, optimizer, debug=True):
    '''
    A training function that trains the model for one epoch
    :param epoch: The index of current epoch
    :param batched_train_data: A list containing batches of images
    :param batched_train_label: A list containing batches of labels
    :param model: The model to be trained
    :param optimizer: The optimizer that updates the network weights
    :return:
        epoch_loss: The average loss of current epoch
        epoch_acc: The overall accuracy of current epoch
    '''
    epoch_loss = 0.0
    hits = 0
    count_samples = 0.0
    for idx, (input, target) in enumerate(zip(batched_train_data, batched_train_label)):

        start_time = time.time()
        loss, accuracy = model.forward(input, target)

        optimizer.update(model)
        epoch_loss += loss
        hits += accuracy * input.shape[0]
        count_samples += input.shape[0]

        forward_time = time.time() - start_time
        if idx % 10 == 0 and debug:
            print(('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time:.3f} \t'
                  'Batch Loss {loss:.4f}\t'
                  'Train Accuracy ' + "{accuracy:.4f}" '\t').format(
                epoch, idx, len(batched_train_data), batch_time=forward_time,
                loss=loss, accuracy=accuracy))
    epoch_loss /= len(batched_train_data)
    epoch_acc = hits / count_samples

    if debug:
        print("* Average Accuracy of Epoch {} is: {:.4f}".format(epoch, epoch_acc))
    return epoch_loss, epoch_acc



def evaluate(batched_test_data, batched_test_label, model, debug=True):
    '''
    Evaluate the model on test data
    :param batched_test_data: A list containing batches of test images
    :param batched_test_label: A list containing batches of labels
    :param model: A pre-trained model
    :return:
        epoch_loss: The average loss of current epoch
        epoch_acc: The overall accuracy of current epoch
    '''
    epoch_loss = 0.0
    hits = 0
    count_samples = 0.0
    for idx, (input, target) in enumerate(zip(batched_test_data, batched_test_label)):

        loss, accuracy = model.forward(input, target, mode='valid')

        epoch_loss += loss
        hits += accuracy * input.shape[0]
        count_samples += input.shape[0]
        if debug:
            print(('Evaluate: [{0}/{1}]\t'
                  'Batch Accuracy ' + "{accuracy:.4f}" '\t').format(
                idx, len(batched_test_data), accuracy=accuracy))
    epoch_loss /= len(batched_test_data)
    epoch_acc = hits / count_samples

    return epoch_loss, epoch_acc


def plot_curves(train_loss_history, train_acc_history, valid_loss_history, valid_acc_history):
    '''
    Plot learning curves with matplotlib. Make sure training loss and validation loss are plot in the same figure and
    training accuracy and validation accuracy are plot in the same figure too.
    :param train_loss_history: training loss history of epochs
    :param train_acc_history: training accuracy history of epochs
    :param valid_loss_history: validation loss history of epochs
    :param valid_acc_history: validation accuracy history of epochs
    :return: None, save two figures in the current directory
    '''
    #############################################################################
    # TODO:                                                                     #
    #    1) Plot learning curves of training and validation loss                #
    #    2) Plot learning curves of training and validation accuracy            #
    #############################################################################
    plt.figure(0)
    plt.plot(train_loss_history, label='train')
    plt.plot(valid_loss_history, label='val')
    # plt.title("Classification loss"+"\n"+ " reg: "+ str(args.reg)+" learning rate: "+str(args.learning_rate)+" hidden layer size: "+str(args.hidden_size)+"\n"+" epochs: "+str(args.epochs)+" batch size: "+str(args.batch_size))
    plt.xlabel('Epoch')
    plt.ylabel('Clasification loss')
    plt.legend()
    # loss = str("results/"+"loss__reg_"+ str(args.reg)+"_lr_"+str(args.learning_rate)+"_hs_"+str(args.hidden_size)+"_epochs_"+str(args.epochs)+"_bs_"+str(args.batch_size)+".png")
    # plt.text(0,0," reg: "+ str(args.reg)+" learning rate: "+str(args.learning_rate)+" hidden layer size: "+str(args.hidden_size)+" epochs: "+str(args.epochs)+" batch size: "+str(args.batch_size))
    plt.savefig("loss",format="png")

    plt.figure(1)
    plt.plot(train_acc_history, label='train')
    plt.plot(valid_acc_history, label='val')
    # plt.title("Classification accuracy"+"\n"+ " reg: "+ str(args.reg)+" learning rate: "+str(args.learning_rate)+" hidden layer size: "+str(args.hidden_size)+"\n"+" epochs: "+str(args.epochs)+" batch size: "+str(args.batch_size))
    plt.xlabel('Epoch')
    plt.ylabel("Clasification accuracy")
    plt.legend()
    # acc = str("results/acc_reg_"+ str(args.reg)+"_lr_"+str(args.learning_rate)+"_hs_"+str(args.hidden_size)+"_epochs_"+str(args.epochs)+"_bs_"+str(args.batch_size)+".png")
    # plt.text(0,0,acc)
    plt.savefig("acc",format="png")



    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################