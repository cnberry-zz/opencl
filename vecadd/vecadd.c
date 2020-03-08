#include <stdio.h>
#include <stdlib.h>
#include <OpenCL/opencl.h>

const char * programSource = 
"__kernel \n"
"void vecadd(__global int *A,	\n"
"		__global int *B,	\n"
"		__global int *C)	\n"
"{					\n"
"					\n"
"	int idx = get_global_id(0);	\n"
"					\n"
"	C[idx] = A[idx] + B[idx];	\n"
"}					\n"
;


int main() {

	const int elements = 2048;


	// Allocate space for input/output host data
	size_t datasize = sizeof(int)*elements;
	int *A = (int*)malloc(datasize);
	int *B = (int*)malloc(datasize);
	int *C = (int*)malloc(datasize);

	for(int i; i < elements; i++) {
		A[i] = i;
		B[i] = i;
	}

	cl_int status;

	// Get a platform
	cl_platform_id platform;
	status = clGetPlatformIDs(1,&platform,NULL);

	// Get a device
	cl_device_id device;
	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device,NULL);

	// Create a context and associate it with the device
	cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
	
	// Create a command-queue with a device
	cl_command_queue cmdQueue = clCreateCommandQueue(context,device,0,&status);

	// Create buffers for kernel
	cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize,NULL,&status);
	cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize,NULL,&status);
	cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY,datasize,NULL,&status);

	// Write data to kernel buffers
	status = clEnqueueWriteBuffer(cmdQueue,bufA,CL_FALSE,0,datasize,A,0,NULL,NULL);
	status = clEnqueueWriteBuffer(cmdQueue,bufB,CL_FALSE,0,datasize,B,0,NULL,NULL);

	// Create a program from source above
	cl_program program = clCreateProgramWithSource(context,1,(const char**)&programSource,NULL,&status);
	
	// Build the kenerl program
	status = clBuildProgram(program, 1, &device, NULL,NULL,NULL);

	// Create kernel
	cl_kernel kernel = clCreateKernel(program, "vecadd", &status);

	// Setup kernel arguements
	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
	status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufA);
	status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufB);
	
	// TODO: understand this part more
	// Define an index space of work-items 	
	size_t indexSpaceSize[1] = {elements};
	size_t  workGroupSize[1] = {256};

	// TODO: execute kernel

	// Free OpenCL resources
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(cmdQueue);
	clReleaseMemObject(bufA);
	clReleaseMemObject(bufB);
	clReleaseMemObject(bufC);
	clReleaseContext(context);

	// Free host resources
	free(A);
	free(B);
	free(C);

	printf("Finished!\n");
	return 0;
}
