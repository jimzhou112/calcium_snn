/*-------------------------------------------------------

Filename: snn.c

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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "snn.h"
#include "normalize_input.h"

// Performs Convolution with 6x5x5 output
void convolution(const float input[IN_CHANNELS][IN_SIZE][IN_SIZE], const float weight[KERNEL_SIZE][KERNEL_SIZE][IN_CHANNELS][NUM_KERNELS], const float bias[NUM_KERNELS], float output[OUT_SIZE][OUT_SIZE][NUM_KERNELS])
{

    // Bias
    for (int i = 0; i < OUT_SIZE; ++i)
    {
        for (int h = 0; h < OUT_SIZE; ++h)
        {
            for (int w = 0; w < NUM_KERNELS; ++w)
                output[i][h][w] = bias[w];
        }
    }
    // for (int i = 0; i < NUM_KERNELS; ++i)
    // {
    //     for (int h = 0; h < OUT_SIZE; ++h)
    //     {
    //         for (int w = 0; w < OUT_SIZE; ++w)
    //             printf("%f\n", output[i][h][w]);
    //     }
    // }
    // for (int i = 0; i < IN_CHANNELS; ++i)
    // {
    //     for (int h = 0; h < IN_SIZE; ++h)
    //     {
    //         for (int w = 0; w < IN_SIZE; ++w)
    //             printf("%f\n", input[i][h][w]);
    //     }
    // }
    // Convolution
    for (int i = 0; i < OUT_SIZE; ++i)
    {
        for (int j = 0; j < OUT_SIZE; ++j)
        {
            for (int h = 0; h < NUM_KERNELS; ++h)
            {
                for (int w = 0; w < IN_CHANNELS; ++w)
                {
                    for (int p = 0; p < KERNEL_SIZE; ++p)
                    {
                        for (int q = 0; q < KERNEL_SIZE; ++q)
                            output[i][j][h] += weight[p][q][w][h] * input[w][i + p][j + q];
                    }
                }
            }
        }
    }
    // for (int i = 0; i < OUT_SIZE; ++i)
    // {
    //     for (int h = 0; h < OUT_SIZE; ++h)
    //     {
    //         for (int w = 0; w < NUM_KERNELS; ++w)
    //         {
    //             if (output[i][h][w] < 0)
    //             {
    //                 output[i][h][w] = 0;
    //             }
    //         }
    //     }
    // }
    // for (int i = 0; i < IN_CHANNELS; ++i)
    // {
    //     for (int h = 0; h < IN_SIZE; ++h)
    //     {
    //         for (int w = 0; w < IN_SIZE; ++w)
    //             printf("%f\n", input[i][h][w]);
    //     }
    // }
    // for (int i = 0; i < OUT_SIZE; ++i)
    // {
    //     for (int h = 0; h < OUT_SIZE; ++h)
    //     {
    //         for (int w = 0; w < NUM_KERNELS; ++w)
    //             printf("%f\n", output[i][h][w]);
    //     }
    // }
    // exit(1);
}

// Applies a linear transformation of the form output = xA^T + b
void fc(const float input[OUT_SIZE][OUT_SIZE][NUM_KERNELS], const float weight[FC_SIZE][NUM_LABELS], const float bias[NUM_LABELS], float output[NUM_LABELS])
{
    for (int i = 0; i < NUM_LABELS; i++)
    {
        output[i] = bias[i];
        // printf("%f\n", output[i]);
    }
    for (int i = 0; i < NUM_LABELS; i++)
    {
        for (int j = 0; j < FC_SIZE; j++)
        {
            // output[i] += weight[j][i] * input[(j % (OUT_SIZE * OUT_SIZE)) / OUT_SIZE][(j % (OUT_SIZE * OUT_SIZE)) % OUT_SIZE][j / OUT_SIZE / OUT_SIZE];
            output[i] += weight[j][i] * input[j / OUT_SIZE / NUM_KERNELS][(j % (OUT_SIZE * NUM_KERNELS)) / NUM_KERNELS][(j % (OUT_SIZE * NUM_KERNELS)) % NUM_KERNELS];
        }
    }
    // for (int i = 0; i < NUM_LABELS; i++)
    // {
    //     for (int j = 0; j < OUT_SIZE; j++)
    //     {
    //         for (int k = 0; k < OUT_SIZE; k++)
    //         {
    //             for (int l = 0; l < NUM_KERNELS; l++)
    //             {
    //                 // output[i] += weight[j][i] * input[(j % (OUT_SIZE * OUT_SIZE)) / OUT_SIZE][(j % (OUT_SIZE * OUT_SIZE)) % OUT_SIZE][j / OUT_SIZE / OUT_SIZE];
    //                 output[i] += weight[(j * OUT_SIZE) + k + (l * OUT_SIZE * OUT_SIZE)][i] * input[j][k][l];
    //             }
    //         }
    //     }
    // }
    // for (int i = 0; i < NUM_LABELS; i++)
    // {
    //     printf("%f\n", output[i]);
    // }
    // exit(1);
    // for (int i = 0; i < NUM_LABELS; i++)
    // {
    //     if (output[i] < 0)
    //     {
    //         output[i] = 0;
    //     }
    // }
}

void forward(const float normalized_input[IN_CHANNELS][IN_SIZE][IN_SIZE], const float conv_weight[KERNEL_SIZE][KERNEL_SIZE][IN_CHANNELS][NUM_KERNELS], float conv_bias[NUM_KERNELS], const float fc_weight[FC_SIZE][NUM_LABELS], const float fc_bias[NUM_LABELS], float curr_spikes[NUM_LABELS], float conv_membrane[OUT_SIZE][OUT_SIZE][NUM_KERNELS], float fc_membrane[NUM_LABELS], const float conv_current[OUT_SIZE][OUT_SIZE][NUM_KERNELS])
{
    int i, j, k;
    // Spike trains
    float conv_spikes[OUT_SIZE][OUT_SIZE][NUM_KERNELS];

    // Input current of neurons
    float fc_current[NUM_LABELS];

    // Initialize input current to 0
    for (i = 0; i < NUM_KERNELS; i++)
    {
        fc_current[i] = 0;
    }

    // // Integrate input current at current time step into membrane
    // for (i = 0; i < IN_SIZE; i++)
    // {

    //     for (j = 0; j < IN_SIZE; j++)
    //     {

    //         input_membrane[0][i][j] += normalized_input[0][i][j];
    //         // printf("%f\n", input_membrane[0][i][j]);
    //     }
    // }

    // // Linear activation
    // for (i = 0; i < IN_SIZE; i++)
    // {
    //     for (j = 0; j < IN_SIZE; j++)
    //     {
    //         if (input_membrane[0][i][j] >= THRESHOLD)
    //         {
    //             input_spikes[0][i][j] = 1;
    //         }
    //         else
    //         {
    //             input_spikes[0][i][j] = 0;
    //         }
    //     }
    // }

    // // Reset membranes by subtraction
    // for (i = 0; i < IN_SIZE; i++)
    // {
    //     for (j = 0; j < IN_SIZE; j++)
    //     {
    //         if (input_spikes[0][i][j] > 0)
    //         {
    //             input_membrane[0][i][j] -= THRESHOLD;
    //         }
    //         if (input_spikes[0][i][j] < 0)
    //         {
    //             input_membrane[0][i][j] += THRESHOLD;
    //         }
    //     }
    // }

    for (i = 0; i < OUT_SIZE; i++)
    {
        for (j = 0; j < OUT_SIZE; j++)
        {
            for (k = 0; k < NUM_KERNELS; k++)
            {
                conv_membrane[i][j][k] += conv_current[i][j][k];
            }
        }
    }
    // for (int i = 0; i < NUM_KERNELS; ++i)
    // {
    //     for (int h = 0; h < OUT_SIZE; ++h)
    //     {
    //         for (int w = 0; w < OUT_SIZE; ++w)
    //             printf("%f\n", conv_membrane[i][h][w]);
    //     }
    // }
    // exit(1);
    // Linear activation
    for (i = 0; i < OUT_SIZE; i++)
    {
        for (j = 0; j < OUT_SIZE; j++)
        {
            for (k = 0; k < NUM_KERNELS; k++)
            {
                if (conv_membrane[i][j][k] >= THRESHOLD)
                {
                    conv_spikes[i][j][k] = 1;
                }
                else
                {
                    conv_spikes[i][j][k] = 0;
                }
            }
        }
    }

    // Reset membranes by subtraction
    for (i = 0; i < OUT_SIZE; i++)
    {
        for (j = 0; j < OUT_SIZE; j++)
        {
            for (k = 0; k < NUM_KERNELS; k++)
            {
                if (conv_spikes[i][j][k] > 0)
                {
                    conv_membrane[i][j][k] -= THRESHOLD;
                }
                if (conv_spikes[i][j][k] < 0)
                {
                    conv_membrane[i][j][k] += THRESHOLD;
                }
            }
        }
    }
    // for (int i = 0; i < OUT_SIZE; ++i)
    // {
    //     for (int h = 0; h < OUT_SIZE; ++h)
    //     {
    //         for (int w = 0; w < NUM_KERNELS; ++w)
    //             printf("%f\n", conv_membrane[i][h][w]);
    //     }
    // }
    // exit(1);
    // Integrate current at current time step into membrane

    fc(conv_spikes, fc_weight, fc_bias, fc_current);

    for (i = 0; i < NUM_LABELS; i++)
    {
        fc_membrane[i] += fc_current[i];
        // printf("%f\n", fc_current[i]);
    }

    // Linear activation
    for (i = 0; i < NUM_LABELS; i++)
    {
        if (fc_membrane[i] >= THRESHOLD)
        {
            curr_spikes[i] = 1;
        }
        else
        {
            curr_spikes[i] = 0;
        }
        // printf("%f\n", curr_spikes[i]);
    }

    // Reset membranes by subtraction
    for (i = 0; i < NUM_LABELS; i++)
    {
        if (curr_spikes[i] > 0)
        {
            fc_membrane[i] -= THRESHOLD;
        }
        if (curr_spikes[i] < 0)
        {
            fc_membrane[i] += THRESHOLD;
        }
    }

    // for (i = 0; i < NUM_LABELS; i++)
    // {
    //     printf("%f\n", fc_membrane[i]);
    // }
}

void simulate(const float normalized_input[IN_CHANNELS][IN_SIZE][IN_SIZE], const float conv_weight[KERNEL_SIZE][KERNEL_SIZE][IN_CHANNELS][NUM_KERNELS], float conv_bias[NUM_KERNELS], const float fc_weight[FC_SIZE][NUM_LABELS], const float fc_bias[NUM_LABELS], float output[NUM_LABELS])
{
    int i, j, k;
    // Spike trains
    float output_spikes[TIMESTEPS][NUM_LABELS];

    // Membrane potential of neurons
    float input_membrane[IN_CHANNELS][IN_SIZE][IN_SIZE];
    float conv_membrane[OUT_SIZE][OUT_SIZE][NUM_KERNELS];
    float fc_membrane[NUM_LABELS];

    // Initialize membrane potentials to 0
    for (i = 0; i < IN_CHANNELS; i++)
    {
        for (j = 0; j < IN_SIZE; j++)
        {
            for (k = 0; k < IN_SIZE; k++)
            {
                input_membrane[i][j][k] = 0;
            }
        }
    }

    for (i = 0; i < OUT_SIZE; i++)
    {
        for (j = 0; j < OUT_SIZE; j++)
        {
            for (k = 0; k < NUM_KERNELS; k++)
            {
                conv_membrane[i][j][k] = 0;
            }
        }
    }

    for (i = 0; i < NUM_LABELS; i++)
    {
        fc_membrane[i] = 0;
    }

    // Input current of neurons
    float conv_current[OUT_SIZE][OUT_SIZE][NUM_KERNELS];

    // Initialize input current to 0
    for (i = 0; i < OUT_SIZE; i++)
    {
        for (j = 0; j < OUT_SIZE; j++)
        {
            for (k = 0; k < NUM_KERNELS; k++)
            {
                conv_current[i][j][k] = 0;
            }
        }
    }

    // Integrate current at current time step into membrane
    convolution(normalized_input, conv_weight, conv_bias, conv_current);

    int timestep;
    int bin;

    for (timestep = 0; timestep < TIMESTEPS; timestep++)
    {
        float curr_spikes[NUM_LABELS];
        forward(normalized_input, conv_weight, conv_bias, fc_weight, fc_bias, curr_spikes, conv_membrane, fc_membrane, conv_current);
        for (bin = 0; bin < NUM_LABELS; bin++)
        {
            // printf("%f\n", curr_spikes[bin]);
            output_spikes[timestep][bin] = curr_spikes[bin];
        }
    }

    for (timestep = 0; timestep < TIMESTEPS; timestep++)
    {
        printf("TIMESTEP %d -- ", timestep);
        for (bin = 0; bin < NUM_LABELS; bin++)
        {
            printf("%.2f, ", output_spikes[timestep][bin]);
        }
        printf("\n");
    }

    for (bin = 0; bin < NUM_LABELS; bin++)
    {
        // Sum the total number of spikes for each bin over the number of timesteps
        float sum = 0;

        for (timestep = 0; timestep < TIMESTEPS; timestep++)
        {
            // printf("%f, ", sum);
            // printf("%f\n", output_spikes[1][bin]);
            sum += output_spikes[timestep][bin];
        }
        output[bin] = sum;
    }
}
