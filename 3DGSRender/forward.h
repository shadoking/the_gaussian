#pragma once
#include "auxiliary.h"

struct DepthPair {
	int idx;
	float depth;	
};

void  PreRender(
	const float* xyz, 
	const float* rotation,
	const float* scaling,

	const float near,
	const int N, 
	const int width,
	const int height,
	const float* viewMatrix, 
	const float* viewProjMatrix,
	float focalX, float focalY,
	float tanFovX, float tanFovY);