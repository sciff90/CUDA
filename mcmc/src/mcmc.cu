#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/random.h>
#include <iostream>
#include <fstream>
#include <string>
#define N 1 
#define M 128

#define Nmax 100000
#define num_samples 401
#define order 2
#define Nthreads M*N
__host__ __device__
unsigned int hash(unsigned int a)
{
	a = (a+0x7ed55d16) + (a<<12);
	a = (a^0xc761c23c) ^ (a>>19);
	a = (a+0x165667b1) + (a<<5);
	a = (a+0xd3a2646c) ^ (a<<9);
	a = (a+0xfd7046c5) + (a<<3);
	a = (a^0xb55a4f09) ^ (a>>16);
	return a;
}

__host__ __device__
unsigned int grid_map(int u1,int u2,int u3)
{
	return u1*(Nmax)*(Nthreads)+u2*(Nthreads)+u3;
}

__global__ void kernel(float * a_save,float* u,float* D) {

	//Get thread number
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;    
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	int grid_width = gridDim.x * blockDim.x;
	//get the global index 
	int global_idx = index_y * grid_width + index_x;
	
	//Set the Random generators up
	unsigned int seed_normal = hash(global_idx);
	unsigned int seed_uniform = hash(global_idx*256*256);
	thrust::default_random_engine rng_normal(seed_normal);
	thrust::default_random_engine rng_uniform(seed_uniform);
	thrust::random::experimental::normal_distribution<float> dist_norm(0,1);
	thrust::random::uniform_real_distribution<float> dist_uniform(0,1);	

	float b_curr[order+1],a_curr[order+1];
	float b_cand[order+1],a_cand[order+1];
	float y_cand[num_samples], y_curr[num_samples];

	for(int ii =0;ii<order+1;ii++)
	{
		b_curr[ii] = dist_uniform(rng_uniform);
		a_curr[ii] = dist_uniform(rng_uniform);
	}
	a_curr[0] = 1.0;

	//Filter Output

	double chi_curr,chi_cand,ratio,a_ratio;
	int flg = 0;
	int accepted = 0;
	int nn = 0;
	int burnin = 0;
	int count = 0;
	double sigma = 1.0;
	

	//RSS for error functions chi



	//a_save[grid_map(0,0,global_idx)] = dist_norm(rng_normal);
	//a_save[grid_map(0,1,global_idx)] = dist_uniform(rng_uniform);

	while(nn<=Nmax)
	{
		for(int ii=0;ii<order+1;ii++)
		{
			a_cand[ii] = a_curr[ii] + sigma*dist_norm(rng_normal);
			b_cand[ii] = b_curr[ii] + sigma*dist_norm(rng_normal);
		}
		a_cand[0] = 1.0;
		//Filter Output

		//Rss for candidate

		ratio = exp(-(chi_cand)+chi_curr);

		if(dist_uniform(rng_uniform)<=ratio)
		{
			for(int ii=0;ii<order+1;ii++)
			{
				a_curr[ii] = a_cand[ii];
				b_curr[ii] = b_cand[ii];
			}
			chi_curr = chi_cand;
			accepted++;
		}

		if(count%1000==0 && count!=0 && flg==0)
		{
			a_ratio = (double)(accepted)/count;
			if(a_ratio < 0.3)
			{
				sigma = sigma/1.2;
				count = 0;
				accepted = 0;
			}
			else if(a_ratio>0.4)
			{
				sigma = sigma*1.2;
				count = 0;
				accepted = 0;
			}
			else
			{
				burnin = nn-1;
				flg = 1;
			}
		}
		count++;
		nn++;
		if(flg==1)
		{
		}

	}


}

int main(void)
{
	//Read in Parameters
	unsigned int num_samples1,order1;
	float fs,fc,fnorm,dt,t1,t0;

	std::ifstream params;
	std::string fn = "data/params.dat";
	params.open(fn.c_str());
	params.ignore(10000,'\n');
	params>>order1;
	params>>fs;
	params>>fc;
	params>>fnorm;
	params>>dt;
	params>>t1;
	params>>t0;
	params>>num_samples1;

	//Read in u t and D
	
	thrust::device_vector<float> u,D,t;
	thrust::host_vector<float> u_host;
	std::ifstream u_dat,D_dat,t_dat;
	u_dat.open("data/u.dat");
	D_dat.open("data/D.dat");
	t_dat.open("data/t.dat");
	std::cout<<"order = "<<order<<std::endl;
	std::cout<<"num_samples = "<<num_samples1<<std::endl;
	float val;
	for(int ii=0;ii<num_samples1;ii++)
	{	
		u_dat>>val;
		u.push_back(val);
		D_dat>>val;
		D.push_back(val);
		t_dat>>val;
		t.push_back(val);
	}
	u_dat.close();
	D_dat.close();
	t_dat.close();

	//Define Host and device vectors
	thrust::device_vector<float> a_save_dev(Nthreads*Nmax*(order+1));
    	thrust::host_vector<float> a_save_host(Nthreads*(order+1));	
	
	//Call Parallel MCMC passing in pointers to save files
	float* a_save_ptr = thrust::raw_pointer_cast(a_save_dev.data());
	float* b_save_ptr = thrust::raw_pointer_cast(a_save_dev.data());
	float* D_ptr = thrust::raw_pointer_cast(D.data());
	float* u_ptr = thrust::raw_pointer_cast(u.data());

	kernel<<<N,M>>>(a_save_ptr,u_ptr,D_ptr);
	cudaDeviceSynchronize();

	//Transfer save files from dev to host
	a_save_host = a_save_dev;

	u_host = u;
	for(int ii=0;ii<num_samples;ii++)
		std::cout << "u[ii] = "<<u[ii]<<std::endl;
	
	std::cout << "num_samples = "<<num_samples<<std::endl;

	return 0;
}
