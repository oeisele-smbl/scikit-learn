#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

from ._typedefs cimport DTYPE_t, ITYPE_t, DITYPE_t, DTYPECODE
from ._typedefs import DTYPE, ITYPE
from libc.math cimport sin, cos, acos, exp, sqrt, fabs, M_PI, fmod, modf


######################################################################
# Numpy 1.3-1.4 compatibility utilities
######################################################################

DEF max_dims=100

## define function pointer
ctypedef DTYPE_t (*METRIC)(DTYPE_t x, DTYPE_t y  ) nogil

## pointer to function pointer list
cdef METRIC* get_funcs_pointer( METRIC [max_dims] lst   ):
    return &lst[0] 


## metric functions
#-------------------
cdef DTYPE_t gaussian( DTYPE_t x, DTYPE_t y) nogil:
        return (x-y)**2

#cdef DTYPE_t mises( DTYPE_t x, DTYPE_t y) nogil:
#        cdef DTYPE_t d = 0.
#        cdef DTYPE_t pi = M_PI
#        d = fmod( x-y + pi , 2.*pi)-pi 
#        d = d/(2.*pi) ## dist range is 0,pi scale to unit lenght
#        return fabs(d)

cdef DTYPE_t mises( DTYPE_t x, DTYPE_t y) nogil:
        cdef DTYPE_t d = 0.
        cdef DTYPE_t pi = M_PI
        d = (1 - cos(x-y))/(2.*M_PI)
        return fabs(d*d)

cdef DTYPE_t categorical( DTYPE_t x, DTYPE_t y) nogil:
        cdef DTYPE_t d = 0
        if x==y:
            d = 0.
        else:
            d = 9999.9        
        return d



cdef class SBGowerDistance(DistanceMetric):

    ## pointer to function pointer list

    cdef METRIC  funcs[max_dims] 
    cdef METRIC* funcs_pointer
    cdef list funcs_py
    r"""Minkowski Distance

    .. math::
       D(x, y) = [\sum_i (x_i - y_i)^p] ^ (1/p)

    Minkowski Distance requires p >= 1 and finite. For p = infinity,
    use ChebyshevDistance.
    Note that for p=1, ManhattanDistance is more efficient, and for
    p=2, EuclideanDistance is more efficient.
    """
    def __cinit__(self):
        for i in range(max_dims):
            self.funcs[i] = gaussian
            self.funcs_py = list()

    def __getstate__(self):
        """
        get state for pickling
        """
        return (self.funcs_py,)

    def __setstate__(self, state):
        """
        set state for pickling
        """
        self.funcs_py = state[0]
        for ii,f in enumerate(self.funcs_py):
            if f ==0:
                self.funcs[ii] = gaussian
            elif f==1:
                 self.funcs[ii] = mises
            elif f==2:
                 self.funcs[ii] = categorical



    def __init__(self, type_mapper = None ):
        # sort based on dimensions
        ktypes =list() 
        dims = list()
        for k, v in type_mapper.items():
            ktypes.extend([k]*len(v))
            dims.extend(v)
       
         # create list of function pointers
        funcs_py = list() 
        for ii in np.argsort(dims):
             if ktypes[ii] == 'gaussian':
                 self.funcs[dims[ii]]==gaussian
                 funcs_py.append(0)
             if ktypes[ii] == 'mises':
                 self.funcs[dims[ii]] = mises
                 funcs_py.append(1)
             if ktypes[ii] == 'categorical':
                 self.funcs[dims[ii]] = categorical
                 funcs_py.append(2)

             self.funcs_py = funcs_py

        # store functions for pickling
       

#        self.funcs_pointer = get_funcs_pointer(self.funcs) 



    cdef inline DTYPE_t rdist(self, DTYPE_t* x1, DTYPE_t* x2,
                              ITYPE_t size) nogil except -1:
        cdef DTYPE_t d=0
        cdef DTYPE_t _d=0

        cdef np.intp_t j
        for j in range(size):
            _d =  self.funcs[j]( x1[j] , x2[j])
            if _d ==9999.9:
                d = _d
                break
            else:
               d += _d
            
        return sqrt(d)

    cdef inline DTYPE_t dist(self, DTYPE_t* x1, DTYPE_t* x2,
                             ITYPE_t size) nogil except -1:
        return self.rdist(x1,x2,size)

    cdef inline DTYPE_t _rdist_to_dist(self, DTYPE_t rdist) nogil except -1:
        return rdist

    cdef inline DTYPE_t _dist_to_rdist(self, DTYPE_t dist) nogil except -1:
        return dist

    def rdist_to_dist(self, rdist):
        return rdist 

    def dist_to_rdist(self, dist):
        return dist 




