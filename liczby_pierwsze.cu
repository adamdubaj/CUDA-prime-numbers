#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <iostream>
#include <cmath>
#include <cstdio>
#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#include <device_functions.h>

//kernel ustawiajacy co x-ta pozycje w tablicy na 0 (FALSE) od x elementu
__global__ void falseFlagsMaker(unsigned int *container, unsigned int containerSize, unsigned int x) {
	unsigned int threadPos = threadIdx.x + blockIdx.x * blockDim.x;
	if (threadPos < containerSize) {
		if (threadPos == 0 || threadPos == 1) {
			container[threadPos] = 0;
		}
		if (threadPos%x == 0 && threadPos != x) {
			container[threadPos] = 0;
		}
	}
}

//kernel przygotowywujacy tablice pod operacje reduce do znalezienia kolejnej liczby do testowania
__global__ void makeArray(unsigned int *container, unsigned int *temp, unsigned int size) {
	unsigned int threadPos = threadIdx.x + blockIdx.x * blockDim.x;
	if (threadPos < size) {
		if (container[threadPos] == 0) temp[threadPos] = size;
		else if (container[threadPos] == 1) temp[threadPos] = threadPos;
	}
}

//struktura, model predykatu - potrzebny do operacji copy_if
struct is_one
{
	__host__ __device__
		bool operator()(const int x)
	{
		return x == 1;
	}
};

int main() {
	std::cout << " ----- Projekt 3: Szukanie liczb pierwszych ----- " << std::endl << std::endl;

	// podanie maksymalnej sprawdzanej liczby
	unsigned int m;
	std::cout << "Podaj liczbe, do ktorej mam szukac liczb pierwszych: ";
	std::cin >> m;

	if (m < 1) {
		std::cout << "BRAK LICZB PIERWSZYCH" << std::endl;
		return 0;
	}
	else if(m==2){
		std::cout << 2 << std::endl;
		return 0;
	}
	else {
		// wielkosc wektora flags
		unsigned const int ARRAY_SIZE = m + 1;

		// zaalokownie wektora hosta
		thrust::host_vector<unsigned int> hostFlags(ARRAY_SIZE);
		//wypelnienie wektora hosta
		for (int i = 0; i < ARRAY_SIZE; i++) {
			hostFlags[i] = 1;
		}
		//kopiowanie host -> device
		thrust::device_vector<unsigned int> deviceFlags = hostFlags;

		//zaalokowanie i inicializacja zerami wektorow pomocniczych
		thrust::device_vector<unsigned int> deviceTemp(ARRAY_SIZE);
		thrust::host_vector<unsigned int> hostTemp = deviceTemp;

		//2 jest pierwsza liczba pierwsza, 2 jest pod indeksem 2
		int index = 2;

		//przesiewamy do pierwiastka z m - Sito Erastotenesa
		int stop = ceil(sqrt(m));
		while (index <= stop) {
			falseFlagsMaker << <1, ARRAY_SIZE >> > (deviceFlags.data().get(), ARRAY_SIZE, index);
			makeArray << <1, ARRAY_SIZE >> > (deviceFlags.data().get(), deviceTemp.data().get(), ARRAY_SIZE);
			index = thrust::reduce(deviceTemp.begin() + (index + 1), deviceTemp.end(), -1, thrust::minimum<unsigned int>());
			if (index > stop) {
				hostTemp = deviceTemp;
				hostFlags = deviceFlags;
			}
		}

		//jak duza ma byc tablica wynikowa
		int amount = 0;
		for (int i = 0; i < ARRAY_SIZE; i++) {
			if (hostFlags[i] == 1) {
				amount++;
			}
		}

		//wektory wynikowe
		thrust::device_vector<unsigned int> deviceResult(amount);
		thrust::host_vector<unsigned int> hostResult;

		//copy_if
		thrust::copy_if(deviceTemp.begin(), deviceTemp.end(), deviceFlags.begin(), deviceResult.begin(), is_one());
		hostResult = deviceResult;

		//wyswietlanie wyniku
		for (int i = 0; i < hostResult.size(); i++) {
			std::cout << hostResult[i] << ", ";
		}
		std::cout << std::endl;

		return 0;
	}

	return 0;
}