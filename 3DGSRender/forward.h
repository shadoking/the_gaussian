#pragma once
#include "auxiliary.h"

void  PreRender(
	const float* xyz, 
	const float near, 
	const int N, 
	const float* viewMatrix, 
	const float* viewProjMatrix);