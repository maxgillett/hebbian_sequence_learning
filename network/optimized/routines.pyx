cdef extern from "lif_transfer_function.h":
    cpdef double Phi(double mu, double sigma, double taum, double threshold, double reset, double taurp)
