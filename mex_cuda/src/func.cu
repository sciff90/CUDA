#include <mex.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <iostream>
#include <fstream>
struct prg
{
	float a, b;

	__host__ __device__ 
		prg(float _a=0.f, float _b=1.f) : a(_a), b(_b) {};
	__host__ __device__ 
		float operator()(const unsigned int n) const
		{
			thrust::default_random_engine rng;
			thrust::random::experimental::normal_distribution<float> dist(a, b);
			rng.discard(n);

			return dist(rng);
		}
};


void func(double *u,double *y,double *theta,int dimx,int dimy)
{
	const int N = dimx;
	thrust::device_vector<float> numbers(N);	    
	thrust::counting_iterator<unsigned int> index_sequence_begin(0);
	thrust::transform(index_sequence_begin,index_sequence_begin + N,numbers.begin(),prg(0.f,1.f));

	std::ofstream myfile;
	myfile.open("data/normal.dat");
	for(int i = 0; i < N; i++)
	{
		myfile << numbers[i] << std::endl;
		printf("y[%d] = %f\n",i,y[i]);
	}
	myfile.close();
	return ;
} 

