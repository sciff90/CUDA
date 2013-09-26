#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/random.h>
#include <iostream>
#include <fstream>
#include <string>
#define N 1 
#define M 1 

#define Nmax 220000
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

__device__
void filter_out(float a[],float b[],float y[],float u[],int Npts,int n_order)
{
	int ii,jj;

	for(ii=0;ii<Npts;ii++)y[ii] = 0;
	for (ii = n_order; ii < (Npts); ii++)
	{
		for (jj = 1; jj <= n_order; jj++)
		{
			y[ii] = y[ii] - a[jj]*y[ii-jj];
		}
		for (jj = 0; jj <= n_order; jj++)
		{
			y[ii] = y[ii]+b[jj]*u[ii-jj];
		}
		
		y[ii] = y[ii]/a[0];
		//if(abs(y[ii])>1)y[ii] = 10;
	}
}

__device__
float RSS(float y1[],float y2[],int Npts)	
{
	int ii;
	double total = 0;
	for (ii = 0; ii < Npts; ii++)
	{
		total = total + pow((y1[ii]-y2[ii]),2);
	}
	return total;
}

__global__ void kernel(float * a_save,float* b_save,float* u,float* D) {

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
	//printf("a_curr[2] = %f\n",a_curr[2]);
	//Filter Output
	filter_out(a_curr,b_curr,y_curr,u,num_samples,order);
	
	printf("u[1] = %f\n",u[1]);
	a_save[1] = 0.123456;
	printf("a_save[1] = %f\n",a_save[1]);
	double chi_curr,chi_cand,ratio,a_ratio;
	int flg = 0;
	int accepted = 0;
	int nn = 0;
	int burnin = 0;
	int count = 0;
	double sigma = 1.0;
	

	//RSS for error functions chi
	chi_curr = RSS(D,y_cand,num_samples);



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
		//printf("randn*sigma = %f\n",sigma);
		//printf("a_cand[2] = %f\n",a_cand[2]);
		//Filter Output
		filter_out(a_cand,b_cand,y_cand,u,num_samples,order);
		//Rss for candidate
		chi_cand = RSS(D,y_cand,num_samples);
		ratio = exp(-(chi_cand)+chi_curr);
	
		if(dist_uniform(rng_uniform)<=ratio)
		{
			for(int ii=0;ii<order+1;ii++)
			{
				a_curr[ii] = a_cand[ii];
				b_curr[ii] = b_cand[ii];
			}
			chi_curr = chi_cand;
			if(nn%1000==0)	printf("ratio = %f\n",ratio);
			accepted++;
		}

		if(count%1000==0 && count!=0 && flg==0)
		{
			
			a_ratio = (double)(accepted)/count;
			//printf("a_ratio = %f\n",a_ratio);
			printf("sigma = %f\n",sigma);
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
			for(int ii=0;ii<order+1;ii++)
			{
				a_save[grid_map(ii,nn,global_idx)] = a_curr[ii];
				//printf("a_curr[1] = %f\n",a_curr[1]);
				b_save[grid_map(ii,nn,global_idx)] = b_curr[ii];
			}
		}

	}
	for(int ii=0;ii<Nmax;ii++)
		printf("a_save[1] = %f\n",a_save[grid_map(1,ii,global_idx)]);


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
	thrust::device_vector<float> b_save_dev(Nthreads*Nmax*(order+1));
    	thrust::host_vector<float> b_save_host(Nthreads*(order+1));	
	
	//Call Parallel MCMC passing in pointers to save files
	float* a_save_ptr = thrust::raw_pointer_cast(a_save_dev.data());
	float* b_save_ptr = thrust::raw_pointer_cast(b_save_dev.data());
	float* D_ptr = thrust::raw_pointer_cast(D.data());
	float* u_ptr = thrust::raw_pointer_cast(u.data());

	kernel<<<N,M>>>(a_save_ptr,b_save_ptr,u_ptr,D_ptr);
	cudaDeviceSynchronize();

	//Transfer save files from dev to host
	a_save_host = a_save_dev;
	b_save_host = b_save_dev;

	u_host = u;

	//for(int ii=0;ii<num_samples;ii++)
	//	std::cout << "u[ii] = "<<u_host[ii]<<std::endl;
	
	std::cout << "num_samples = "<<num_samples<<std::endl;

	std::ofstream a_dat,b_dat;
	a_dat.open("data/a.dat");
	b_dat.open("data/b.dat");
	for(int ii=0;ii<Nthreads;ii++)
	{
		for(int jj=0;jj<Nmax;jj++)
		{
			for(int kk=0;kk<order+1;kk++)
			{
				a_dat << a_save_host[grid_map(kk,jj,ii)] << "\t";
				b_dat << b_save_host[grid_map(kk,jj,ii)] << "\t";
			}

		}
		a_dat << std::endl;
		b_dat <<std::endl;
	}



	return 0;
}

