__author__ = 'chinna'


c1      = 64*3*3*3          + 64
c2      = 128*64*3*3        + 128
c3      = 256*128*3*3       + 256
c4      = 256*256*3*3       + 256
fc1     = 1024*256*1*1      + 1024
fc2     = 1024*1024*1*1     + 1024
fc3     = 10*1024*1*1       + 10
g       = 1024
b       = 1024

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
