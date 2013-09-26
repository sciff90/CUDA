#mcmc CUDA Makefile
#Define Objects

OBJ = obj
SRC = src
PLOTS = plots
DATA = data
EXEC = xmcmc

OBJECTS = ${OBJ}/mcmc.o

#Define Compiler

#############nvcc############
COMP = nvcc
	
#Define Switches
#SWITCH = -O2 -fopenmp -std=c++0x
SWITCH = -arch=sm_20 -g

#Define Libraries
#LIB_DIR = 
#LIB = 
IDIR = -I.
##############################


#Makefile
all: $(OBJECTS)
		$(COMP) $(SWITCH) ${OBJECTS} $(IDIR) -o ${EXEC} $(LIB_DIR) $(LIB)
			
${OBJ}/mcmc.o: ${SRC}/mcmc.cu
		$(COMP) -c $(SWITCH) $(LIB_DIR) $(LIB) $(IDIR) ${SRC}/mcmc.cu -o ${OBJ}/mcmc.o
			
clean:
		rm ${OBJ}/*.o		
		rm ${DATA}/*.dat

#Cleaning Everything
cleanest: clean
		rm ${EXEC}
# End of the makefile
