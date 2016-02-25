__author__ = 'chinna'

import matplotlib.pyplot as plt

c1      = 64*3*3*3          + 64
c2      = 128*64*3*3        + 128
c3      = 256*128*3*3       + 256
c4      = 256*256*3*3       + 256
fc1     = 1024*256*1*1      + 1024
fc2     = 1024*1024*1*1     + 1024
fc3     = 10*1024*1*1       + 10
g       = 10
b       = 10

total = c1 + c2 + c3 + c4 + fc1 + fc2 + fc3 + g + b
print total


l1 = 64*30*30
l2 = 128*13*13
l3 = 256*4*4
l4 = 256*2*2
l5 = 1024
l6 = 1024
l7 = 10
conv_neurons = l1 + l2 + l3 + l4
fc_neurons   = l5 + l6 + l7
print conv_neurons
print fc_neurons
print conv_neurons + fc_neurons


#q4

#run 1
valid_accuracy = [0.5190, 0.6482, 0.7052, 0.7614, 0.8220, 0.8534, 0.8822, 0.9058, 0.9236, 0.9316 ]

train_accuracy = [0.6852, 0.6912, 0.7665, 0.8212, 0.8657, 0.9009, 0.9286, 0.9484, 0.9635, 0.9754 ]

valid_ll = [0.4852, 0.6912, 0.7665, 0.8212, 0.8657, 0.9009, 0.9286, 0.9484, 0.9635, 0.9754 ]

#run2

valid_accuracy = [0.3294, 0.6694, 0.6480, 0.7870, 0.8354, 0.8372, 0.8678, 0.9074, 0.9194, 0.9354, 0.9124, 0.9466, 0.9602, 0.9702]

train_accuracy = [0.4841, 0.6887, 0.7713, 0.8253, 0.8684, 0.9036, 0.9296, 0.9489, 0.9643, 0.9743, 0.9799, 0.9844, 0.9887, 0.9932]

valid_ll = [2.1628, 0.9472, 1.1238, 0.6411, 0.5027, 0.4753, 0.3929, 0.2973, 0.2679, 0.2019, 0.2865, 0.1941, 0.1519, 0.1066]

plt.plot(valid_accuracy,label='Valid Accuracy')
plt.plot(train_accuracy,label='Train  Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.suptitle('Train-Validation Accuracy curve')
legend = plt.legend(loc='lower center', shadow=True)
plt.savefig("q4_plot_train_valid.jpeg")
plt.close()



plt.plot(valid_ll)
plt.xlabel('Epochs')
plt.ylabel('Negative liklihood')
plt.suptitle('NLL vs iterations curve')
plt.savefig("q4_plot_NLL.jpeg")
plt.close()


time = [0.141967, 0.113064, 0.126685 ]
print "time taken avg", sum(time)/len(time)
