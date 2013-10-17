/*********************************************************************
 * mex_cuda.cpp
 *
 * This file shows the basics of setting up a mex file to work with
 * Matlab. Linking matlab arrays into cpp for transfer onto cuda card.
 * 
 * Keep in mind:
 * <> Use 0-based indexing as always in C or C++
 * <> Indexing is column-based as in Matlab (not row-based as in C)
 * <> Use linear indexing.  [x*dimy+y] instead of [x][y]
 *
 *
 ********************************************************************/
#include <mex.h> 
#include <func.cuh>
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

//declare variables
    mxArray *u_in_m, *y_in_m, *theta_out_m;
    const mwSize *dims;
    double *u, *y, *theta;
    int num_samples,chain_length, order,numdims;
    
//associate inputs
    u_in_m = mxDuplicateArray(prhs[0]);
    y_in_m = mxDuplicateArray(prhs[1]);

//figure out dimensions
    dims = mxGetDimensions(prhs[0]);
    numdims = mxGetNumberOfDimensions(prhs[0]);
    num_samples = (int)dims[0];
    
    //dims = mxGetDimensions(plhs[0]);
    //numdims = mxGetNumberOfDimensions(plhs[0]);
    chain_length = 1000;
    order = 30;

//associate outputs
    theta_out_m = plhs[0] = mxCreateDoubleMatrix((mwSize)chain_length,(mwSize)order,mxREAL);;
    

//associate pointers
    u = mxGetPr(u_in_m);
    y = mxGetPr(y_in_m);
    theta = mxGetPr(theta_out_m);

    func(u,y,theta,order,chain_length,num_samples);
    

    return;
}
