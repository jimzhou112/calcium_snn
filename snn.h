/*-------------------------------------------------------

Filename: snn.h

Takes in calcium image (./Calcium/input.txt) and outputs prediction label.

Model trained on the following CNN Pytorch model, then converted to an SNN using based on a converted SNN using rate-based encoding:

*******************************************
class Simplenet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 6, 3, 1)
        self.fc1 = nn.Linear(150, 23)
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
*******************************************

---------------------------------------------------------*/
#ifndef SNN_H
#define SNN_H

#define IN_SIZE 16
#define IN_CHANNELS 1
#define KERNEL_SIZE 3
#define NUM_KERNELS 6
#define NUM_LABELS 24
#define OUT_SIZE (IN_SIZE - KERNEL_SIZE + 1)
#define DATASET "Calcium"
#define MODEL "CNN"
#define FC_SIZE (OUT_SIZE * OUT_SIZE * NUM_KERNELS)

#define max(x, y) (((x) >= (y)) ? (x) : (y))

void cnn(const float normalized_input[IN_CHANNELS][IN_SIZE][IN_SIZE], const float conv_weight[NUM_KERNELS][IN_CHANNELS][KERNEL_SIZE][KERNEL_SIZE], float conv_bias[NUM_KERNELS], float conv_output[NUM_KERNELS][OUT_SIZE][OUT_SIZE], float relu_output[NUM_KERNELS][OUT_SIZE][OUT_SIZE], float fc_input[150], const float fc_weight[NUM_LABELS][150], const float fc_bias[NUM_LABELS], float output[NUM_LABELS]);

#endif
