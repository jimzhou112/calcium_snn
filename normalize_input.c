/*-------------------------------------------------------

Filename: normalize_input.c

Transforms PIL image input to normalized input for SNN. 

Convert PIL Image in the range [0, 255] to the range [0.0, 1.0].

---------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include "normalize_input.h"
#include <math.h>
#include "snn.h"

void normalize_input(const int input[IN_CHANNELS][IN_SIZE][IN_SIZE], float normalized_input[IN_CHANNELS][IN_SIZE][IN_SIZE])
{

    for (int i = 0; i < IN_SIZE; i++)
    {
        for (int j = 0; j < IN_SIZE; j++)
        {
            normalized_input[0][i][j] = (255.0 - (float)input[0][i][j]) / 255.0;
        }
    }
    // for (int i = 0; i < IN_SIZE; i++)
    // {
    //     for (int j = 0; j < IN_SIZE; j++)
    //     {
    //         printf("%f\n", normalized_input[0][i][j]);
    //     }
    // }
    // exit(1);
}
