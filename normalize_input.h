/*-------------------------------------------------------

Filename: normalize_input.h

Transforms PIL image input to normalized input for SNN. 

Convert PIL Image in the range [0, 255] to the range [0.0, 1.0].

---------------------------------------------------------*/

#ifndef NORMALIZEINPUT_H
#define NORMALIZEINPUT_H

#include "snn.h"
void normalize_input(const int input[IN_CHANNELS][IN_SIZE][IN_SIZE], float normalized_input[IN_CHANNELS][IN_SIZE][IN_SIZE]);

#endif
