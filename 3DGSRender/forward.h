#pragma once
#include "auxiliary.h"

void  PreRender(
	const float* xyz, 
	const float* rotation,
	const float* scaling,

	const float near,
	const int N, 
	const float* viewMatrix, 
	const float* viewProjMatrix,
	float focalX, float focalY,
	float tanFovX, float tanFovY);