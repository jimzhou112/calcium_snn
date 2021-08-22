/*-------------------------------------------------------

Filename: snn_test.c

Takes in calcium image (./Calcium/input.txt) and outputs prediction label.

Model trained on the following CNN Pytorch model, then converted to an SNN using based on a converted SNN using rate-based encoding:

*******************************************
class Simplenet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 6, 3, 1)
        self.relu = nn.ReLU(inplace=False)
        self.fc1 = nn.Linear(150, 23)

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
*******************************************

---------------------------------------------------------*/
#include "snn.h"
#include "normalize_input.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

int main()
{
    int i, j, k;
    FILE *ifp, *ofp;
    char file_name[100];
    char *line = NULL;
    size_t len = 0;
    const char *s = " ";
    char *token = NULL;

    struct timeval t1, t2, tr;

    // Input arrays
    static int input[IN_CHANNELS][IN_SIZE][IN_SIZE];
    static float normalized_input[IN_CHANNELS][IN_SIZE][IN_SIZE];

    // Convolution arrays
    static float conv_weight[KERNEL_SIZE][KERNEL_SIZE][IN_CHANNELS][NUM_KERNELS];
    static float conv_bias[NUM_KERNELS];

    // Fc arrays
    static float fc_weight[FC_SIZE][NUM_LABELS];
    static float fc_bias[NUM_LABELS];

    // Output
    static float output[NUM_LABELS];

    // Load inputs
    sprintf(file_name, "./%s/input.txt", DATASET);
    if (!(ifp = fopen(file_name, "r")))
    {
        printf("File input.txt cannot be opened for read.\n");
        return -1;
    }
    for (i = 0; i < IN_SIZE; i++)
    {
        for (j = 0; j < IN_SIZE; j++)
        {
            if (fscanf(ifp, "%u", &input[0][i][j]) != 1)
            {
                printf("File input.txt is not of correct length or format\n");
                return -1;
            }
        }
    }

    // for (i = 0; i < IN_SIZE; i++)
    // {
    //     for (j = 0; j < IN_SIZE; j++)
    //     {
    //         printf("%d\n", input[0][i][j]);
    //     }
    // }

    // Load conv1weights
    sprintf(file_name, "./%s/conv1weight.txt", MODEL);
    if (!(ifp = fopen(file_name, "r")))
    {
        printf("File conv1weight.txt cannot be opened for read.\n");
        return -1;
    }
    for (i = 0; i < KERNEL_SIZE; i++)
    {
        for (j = 0; j < KERNEL_SIZE; j++)
        {
            for (k = 0; k < NUM_KERNELS; k++)
            {
                if (fscanf(ifp, "%f", &conv_weight[i][j][0][k]) != 1)
                {
                    printf("File conv1weight.txt is not of correct length or format\n");
                    return -1;
                }
            }
        }
    }

    // for (i = 0; i < KERNEL_SIZE; i++)
    // {
    //     for (j = 0; j < KERNEL_SIZE; j++)
    //     {
    //         for (k = 0; k < NUM_KERNELS; k++)
    //         {
    //             printf("%f\n", conv_weight[i][j][0][k]);
    //         }
    //     }
    // }

    // Load conv1bias
    sprintf(file_name, "./%s/conv1bias.txt", MODEL);
    if (!(ifp = fopen(file_name, "r")))
    {
        printf("File conv1bias.txt cannot be opened for read.\n");
        return -1;
    }
    for (i = 0; i < NUM_KERNELS; i++)
    {
        if (fscanf(ifp, "%f", &conv_bias[i]) != 1)
        {
            printf("File conv1bias.txt is not of correct length or format\n");

            return -1;
        }
    }

    // for (i = 0; i < NUM_KERNELS; i++)
    // {
    //     printf("%f\n", conv_bias[i]);
    // }

    // Load fc1weight
    sprintf(file_name, "./%s/fc1weight.txt", MODEL);
    if (!(ifp = fopen(file_name, "r")))
    {
        printf("File fc1weight.txt cannot be opened for read.\n");
        return -1;
    }
    for (i = 0; i < FC_SIZE; i++)
    {
        for (j = 0; j < NUM_LABELS; j++)
        {
            if (fscanf(ifp, "%f", &fc_weight[i][j]) != 1)
            {
                printf("File fc1weight.txt is not of correct length or format\n");
                return -1;
            }
        }
    }

    // for (i = 0; i < FC_SIZE; i++)
    // {
    //     for (j = 0; j < NUM_LABELS; j++)
    //     {
    //         printf("%f\n", fc_weight[i][j]);
    //     }
    // }

    // Load fc1bias
    sprintf(file_name, "./%s/fc1bias.txt", MODEL);
    if (!(ifp = fopen(file_name, "r")))
    {
        printf("File fc1bias.txt cannot be opened for read.\n");
        return -1;
    }
    for (i = 0; i < NUM_LABELS; i++)
    {
        if (fscanf(ifp, "%f", &fc_bias[i]) != 1)
        {
            printf("File fc1bias.txt is not of correct length or format\n");
            return -1;
        }
    }

    // for (i = 0; i < NUM_LABELS; i++)
    // {
    //     printf("%f\n", fc_bias[i]);
    // }

    // Transforms PIL image input to normalized input for SNN
    normalize_input(input, normalized_input);

    // for (i = 0; i < IN_SIZE; i++)
    // {
    //     for (j = 0; j < IN_SIZE; j++)
    //     {
    //         printf("%f\n", normalized_input[0][i][j]);
    //     }
    // }

    // Start timer
    gettimeofday(&t1, NULL);

    simulate(normalized_input, conv_weight, conv_bias, fc_weight, fc_bias, output);

    // for (i = 0; i < 24; i++)
    // {
    //     printf("%f\n", output[i]);
    // }

    // Stop timer
    gettimeofday(&t2, NULL);
    timersub(&t1, &t2, &tr);
    printf("Inference finished. \n%.5f sec elapsed.\n",
           -tr.tv_sec - (double)tr.tv_usec / 1000000.0);

    // Load ground truth spikes
    static float correct_spikes[NUM_LABELS];

    sprintf(file_name, "./%s/spikes.txt", DATASET);
    if (!(ifp = fopen(file_name, "r")))
    {
        printf("File spikes.txt cannot be opened for read.\n");
        return -1;
    }
    for (i = 0; i < NUM_LABELS; i++)
    {
        if (fscanf(ifp, "%f", &correct_spikes[i]) != 1)
        {
            printf("File spikes.txt is not of correct length or format\n");
            return -1;
        }
    }

    // Verify correctness of program using ground truth spikes
    int num_correct = 0;
    for (i = 0; i < NUM_LABELS; i++)
    {
        if (correct_spikes[i] != output[i])
        {
            printf("Incorrect spikes at bin %d: Correct number of spikes is %.0f but program reports %.0f\n", i, correct_spikes[i], output[i]);
        }
        else
        {
            num_correct += 1;
        }
    }

    if (num_correct == NUM_LABELS)
    {
        printf(" *************** \n");
        printf("   Test Passed!  \n");
        printf(" *************** \n");
    }
    else
    {
        printf(" ****************** \n");
        printf("  Test NOT Passed!  \n");
        printf(" ****************** \n");
    }
    // // Outputs the prediction label
    // int index = 0;
    // float max = output[index];
    // for (i = 0; i < NUM_LABELS; i++)
    // {
    //     if (output[i] > max)
    //     {
    //         max = output[i];
    //         index = i;
    //     }
    // }
    // printf("%d is the prediction label.\n", index); //3
}