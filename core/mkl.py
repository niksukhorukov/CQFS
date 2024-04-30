import ctypes
def mkl_set_num_threads(num_threads=20):
    def mkl_set_num_threads(cores):
        mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(cores)))
        
    mkl_rt = ctypes.CDLL('libmkl_rt.so')
    mkl_max_threads = mkl_rt.mkl_get_max_threads()
    mkl_set_num_threads(num_threads)
    print(f'[mkl]: set up num_threads={mkl_rt.mkl_get_max_threads()}/{mkl_max_threads}')
