#mcmc CUDA Makefile
#Define 
OBJ = obj
SRC = src
DATA = data
PLOTS = plots
INC = inc

#Executable
EXEC = mex_cuda

#Define Directories
MATLAB_DIR = /usr/local/MATLAB/R2013a
CUDA_LIB64='/usr/local/cuda/lib64'

OBJECTS = ${OBJ}/mex_cuda.o\
	  ${OBJ}/func.o
	  		

#Define Compiler

#############Compilers############
MEX = mex
CXX = g++
NVCC = nvcc
	
#Define Switches
CXX_SWITCH = -fPIC
NVCC_SWITCH = -arch=sm_30 -shared -Xcompiler -fPIC
MEX_SWITCH = -cxx  

#Define Libraries
LIB_DIR = -L${CUDA_LIB64}
LIB = -lcudart
IDIR = -I. -I$(MATLAB_DIR)/extern/include -I inc

##############################

#Makefile
all: DIR $(OBJECTS)		
		${MEX} ${MEX_SWITCH} ${OBJECTS} ${LIB_DIR} ${LIB} -o ${EXEC}

${OBJ}/mex_cuda.o: ${SRC}/mex_cuda.cpp ${OBJ}/func.o
		$(CXX) -c $(CXX_SWITCH) $(LIB_DIR) $(LIB) $(IDIR) ${SRC}/mex_cuda.cpp -o ${OBJ}/mex_cuda.o

${OBJ}/func.o: ${SRC}/func.cu ${INC}/func.cuh
		${NVCC} -c $(NVCC_SWITCH) ${LIB_DIR} ${LIB} ${IDIR} ${SRC}/func.cu -o ${OBJ}/func.o 
DIR:
		mkdir -p $(OBJ) $(DATA)	$(PLOTS)
clean:
		rm -rf ${OBJ}		
		rm -rf ${DATA}
		rm -rf ${PLOTS}

#Cleaning Everything
cleanest: clean
		rm -f ${EXEC}.mexa64
# End of the makefile
