#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <CL/cl.hpp>
 
using TimeUnit = std::chrono::microseconds;
using SystemClock = std::chrono::system_clock;

int main(){
    //get all platforms (drivers)
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if(all_platforms.size()==0){
        std::cout<<" No platforms found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Platform default_platform=all_platforms[0];
    std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";
 
    //get default device of the default platform
    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if(all_devices.size()==0){
        std::cout<<" No devices found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Device default_device=all_devices[0];
    std::cout<< "Using device: "<<default_device.getInfo<CL_DEVICE_NAME>()<<"\n";
 
 
    cl::Context context({default_device});
 
    cl::Program::Sources sources;
 
    // kernel calculates for each element C=A+B
    std::ifstream t("simpleadd.cl");
    std::stringstream buffer;
    buffer << t.rdbuf();
    std::string kernel_code= buffer.str();
    sources.push_back({kernel_code.c_str(),kernel_code.length()});
 
    cl::Program program(context,sources);
    if(program.build({default_device})!=CL_SUCCESS){
        std::cout<<" Error building: "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device)<<"\n";
        exit(1);
    }
 
    // create buffers on the device
    cl::Buffer buffer_A(context,CL_MEM_READ_WRITE,sizeof(int)*10);
    cl::Buffer buffer_B(context,CL_MEM_READ_WRITE,sizeof(int)*10);
    cl::Buffer buffer_C(context,CL_MEM_READ_WRITE,sizeof(int)*10);
 
    int A[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int B[] = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0};
    std::vector<int> C(10);
 
    cl::Kernel kernel_add=cl::Kernel(program,"simple_add");
    kernel_add.setArg(0,buffer_A);
    kernel_add.setArg(1,buffer_B);
    kernel_add.setArg(2,buffer_C);
    cl::Event event;

    //create queue to which we will push commands for the device.
    cl::CommandQueue queue(context,default_device,CL_QUEUE_PROFILING_ENABLE);
 
    TimeUnit start_t = std::chrono::duration_cast<TimeUnit>(SystemClock::now().time_since_epoch());
    {
        //write arrays A and B to the device, schedule kernel, wait for kernel, read result
        queue.enqueueWriteBuffer(buffer_A,CL_TRUE,0,sizeof(int)*10,A);
        queue.enqueueWriteBuffer(buffer_B,CL_TRUE,0,sizeof(int)*10,B);
        queue.enqueueNDRangeKernel(kernel_add,cl::NullRange,cl::NDRange(10),cl::NullRange,NULL,&event);
        queue.finish();
        queue.enqueueReadBuffer(buffer_C,CL_TRUE,0,sizeof(int)*10,C.data());
    }
    TimeUnit end_t = std::chrono::duration_cast<TimeUnit>(SystemClock::now().time_since_epoch());
        
 
    std::vector<int> result({0,2,4,3,5,7,6,8,10,9});
    if (std::equal(result.begin(),result.end(),C.begin())) {
        std::cout << "Simpleadd PASSED | ";
    } else {
        std::cout << "Simpleadd FAILED | ";
    }
    auto start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    auto end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    std::cout << "GPU: " <<(end-start) << "ns | ";
    std::cout << "Total: " << std::chrono::duration_cast<TimeUnit>(end_t - start_t).count()
              << " µs." << std::endl;


    std::cout<<" result: \n";
    for(const auto &elem : C){
        std::cout<<elem<<" ";
    }
    std::cout<<std::endl;
 
    return 0;
}
