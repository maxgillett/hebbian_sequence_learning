cdef extern from "integration.h":
    cpdef float C0(float a, float mu, float sigma, int phif)
    cpdef float C1(float a, float b, float mu, float sigma, int phif)
    cpdef float C2(float a, float b, float c, float d, float mu, float sigma, int phif)
    cpdef float C3(float a, float b, float c, float mu, float sigma, int phif)
    cpdef float C4(float a, float b, float c, float d, float mu, float sigma, int phif)

    cpdef float C0_1(float a, float b, float mu, float sigma, int phif)

