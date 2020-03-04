#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <cuda.h>
#include <string.h>
#include <pthread.h>
#include <cuda_runtime.h>
#include <sys/types.h>
#include <unistd.h>
#include <time.h>
#include <sys/stat.h>
#include <fcntl.h>
//#include <helper_cuda.h>
#define SIZE 10000

unsigned long long mod = 9973L;

static const char LIB_STRING[] = "/usr/local/cuda-10.1/targets/x86_64-linux/lib/libcudatr.so";
static const char CONFIG_STRING[] = "WRAPPER_MAX_MEMORY";
static const char LOG_FILENAME[] = "/tmp/wrapper-log";

int open_flag = 0;
static size_t total_mem = 0L;
static size_t total_quota = 4217928960L;
static size_t pytorch_offset_size = 500000000L;
static pthread_mutex_t mem_cnt_lock;
char *error;
char timebuf[30];

struct HashArray
{
    unsigned long long key;
    size_t value;
    struct HashArray* next;
}allocsize[10000];

void getCurrentTime(char *buff) {
    struct tm *sTm;
    time_t now = time (0);
    sTm = gmtime (&now);
    strftime (buff, sizeof(buff), "%Y-%m-%d %H:%M:%S", sTm);
}

void addHash(unsigned long long key,size_t value) {
    int temp=(key >> 51);
    //printf("addHash:%d\n",temp);
    if(allocsize[temp].key==0) {
	allocsize[temp].key=key;
        allocsize[temp].value=value;
    } 
    else if(allocsize[temp].key==key) {
        allocsize[temp].value=value;
    } 
    else {
        struct HashArray *p=&allocsize[temp];       
        while(p->key!=key&&p->next!=NULL) {  
            p=p->next;
        }
        if(p->key==key) {
            p->value=value;
        } 
        else {
            p->next=(struct HashArray*)malloc(sizeof(struct HashArray));
            p=p->next;
            p->key=key;
            p->value=value;
            p->next=NULL;
        }
    }
    getCurrentTime(timebuf);
    printf("addHash\nTime: %s  addHash: key: %lld value: %zu\n", timebuf, key, value);  
}
size_t getHash(unsigned long long key) {
    int temp=key%mod;
    struct HashArray *p=&allocsize[temp];
    //printf("%lld %d\n",key,temp);
    if (p == NULL) {
        printf("getHash miss\n");
        getCurrentTime(timebuf);
        printf("Time: %s  key: %lld \n", timebuf, key );
        return 0;
    }
    //printf("pkey: %lld\n", p->key);
    while(p->key!=key&&p->next!=NULL) {
        p=p->next;
    }       
    if (p->key == key) {
        printf("getHash hit\n");
        getCurrentTime(timebuf);
        printf("Time: %s  key: %lld value: %zu \n", timebuf, key ,p->value);
        return p->value;
    }
    else {
        printf("hash hit and miss\n");
        getCurrentTime(timebuf);
        printf("Time: %s  key: %lld \n", timebuf, key );        
        return 0;
    }
}

void set_quota() {
    char *q = NULL;
    q = getenv(CONFIG_STRING);
    if (q == NULL) {
        printf("set_quota: no env %s found. use default: %zu", CONFIG_STRING, total_quota);
    }
    else {
        total_quota = strtoull(q, NULL, 10);
        printf("set_quota: set total_quota: %zu", total_quota);
    }
}

void init_func() {
    if(open_flag == 0 ) {
        /*int fd;
        fd = open(LOG_FILENAME, O_WRONLY | O_CREAT, 0644);
        if (fd == -1) {
            perror("open log file failed");
            exit(1);
        }

        if (dup2(fd, 1) == -1) {
            perror("dup2 failed"); 
            exit(1);
        }*/

        //char *error;
	open_flag = 1;
    	dlerror();
    	pthread_mutex_init(&mem_cnt_lock, NULL);
	    set_quota();
	    
        printf("Init!\n");
        getCurrentTime(timebuf);
        printf("Time: %s  total_quota: %zu\n", timebuf, total_quota);

    }
}
void *handle;
int check_alloc_valid(size_t bytesize) {
    //printf("lock mem in check_alloc_valid\n");
    pthread_mutex_lock(&mem_cnt_lock);	
    //printf("&&&&&&&&&&&&total_mem %zu\n", total_mem);
    if(total_mem + bytesize + pytorch_offset_size  > total_quota) {
        fprintf (stderr, "alloc %zu failed, total_mem %zu, quota %zu\n", bytesize, total_mem,  total_quota);
	    //printf("unlock mem in check_alloc_valid:1\n");
	    pthread_mutex_unlock(&mem_cnt_lock);
	    return 0;
    }
    //printf("unlock mem in check\n");
    pthread_mutex_unlock(&mem_cnt_lock);
    return 1;
}

/*cudaError_t cudaFree( void *devPtr ) {
    init_func();
    void *hand;    
    //pthread_mutex_lock(&mem_cnt_lock);
    hand = dlopen (LIB_STRING, RTLD_LAZY|RTLD_PRIVATE);
    if (!hand) {
       	fprintf (stderr, "%s\n", dlerror());
     	exit(1);
    }
    dlerror();
    cudaError_t (*fakecudaFree)( void* );
    fakecudaFree = dlsym(hand, "cudaFree");
    printf("cudaFree\n");
    if ((error = dlerror()) != NULL)  {
        fprintf (stderr, "%s\n", error);
        exit(1);
    }
    printf("hand:0x%x\n",hand);
    printf("before free\n");
    printf("!!!!!!devPtr:0x%x\n",devPtr);
    cudaError_t r= (*fakecudaFree)( devPtr );
    dlclose(hand);
    //pthread_mutex_unlock(&mem_cnt_lock);
    printf("free result: %d", r); 
    return r;
}*/

cudaError_t cudaMalloc( void** devPtr, size_t bytesize ) {
    init_func();    
    //pthread_mutex_lock(&mem_cnt_lock);
    handle = dlopen (LIB_STRING, RTLD_LAZY);
    if (!handle) {
       	fprintf (stderr, "%s\n", dlerror());
     	exit(1);
    }
    printf("handle:0x%x\n",handle);
    dlerror();
    cudaError_t (*fakecudaMalloc)( void** ,size_t );
    fakecudaMalloc = dlsym(handle, "cudaMalloc");
    printf("cudaMalloc\n");
    if ((error = dlerror()) != NULL)  {
        fprintf (stderr, "%s\n", error);
        exit(1);
    }
    cudaError_t r= (*fakecudaMalloc)( devPtr , bytesize );
    //dlclose(handle);
    //pthread_mutex_unlock(&mem_cnt_lock);
    printf("malloc result: %d", r);
    return r;
}

/*cudaError_t cudaGetDeviceCount( int* count ) {
        
    init_func();
    handle = dlopen (LIB_STRING, RTLD_LAZY);
    cudaError_t (*fakecudaGetDeviceCount)(int*);
    fakecudaGetDeviceCount = dlsym(handle, "cudaGetDeviceCount");
    printf("cudaGetDeviceCount:\n");
    if ((error = dlerror()) != NULL)  {
        fprintf (stderr, "%s\n", error);
        exit(1);
    }
    //nvml_memquery();
    cudaError_t r= (*fakecudaGetDeviceCount)(count);
    //nvml_memquery();
    dlclose(handle);
    return r;
}*/
