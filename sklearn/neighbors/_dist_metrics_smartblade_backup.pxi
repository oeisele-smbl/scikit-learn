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




cdef class SBGowerDistance(DistanceMetric):
    r"""Minkowski Distance

    .. math::
       D(x, y) = [\sum_i (x_i - y_i)^p] ^ (1/p)

    Minkowski Distance requires p >= 1 and finite. For p = infinity,
    use ChebyshevDistance.
    Note that for p=1, ManhattanDistance is more efficient, and for
    p=2, EuclideanDistance is more efficient.
    """
    def __cinit__(self):
        ##
        self.gaussian =  np.zeros(1, dtype=ITYPE, order='c')
        self.gaussian_ptr = self.get_idx_ptr(self.gaussian)
        self.ng = self.gaussian.shape[0]

        self.mises =  np.zeros(1, dtype=ITYPE, order='c')
        self.mises_ptr = self.get_idx_ptr(self.mises)
        self.nm = self.mises.shape[0]

        self.categorical =  np.zeros(1, dtype=ITYPE, order='c')
        self.categorical_ptr = self.get_idx_ptr(self.categorical)
        self.nc = self.categorical.shape[0]


    def __getstate__(self):
        """
        get state for pickling
        """
        return ( self.gaussian,int(self.ng),
                 self.mises,int(self.nm), 
                 self.categorical, int(self.nc))#,int(self.nm),int(self.nc))

    def __setstate__(self, state):
        """
        set state for pickling
        """
        self.gaussian = state[0]
        self.ng = state[1]
        self.gaussian_ptr = self.get_idx_ptr(self.gaussian )

        self.mises = state[2]
        self.nm = state[3]
        self.mises_ptr = self.get_idx_ptr(self.mises )

        self.categorical = state[4]
        self.nc = state[5]
        self.categorical_ptr = self.get_idx_ptr(self.categorical )



    def __init__(self, gaussian = None, mises = None, categorical=None):

        self.gaussian =  np.asarray(gaussian , dtype=ITYPE)
        self.gaussian_ptr = self.get_idx_ptr(self.gaussian)
        self.ng = self.gaussian.shape[0]


        self.mises =  np.asarray(mises , dtype=ITYPE)
        self.mises_ptr = self.get_idx_ptr(self.mises)
        self.nm = self.mises.shape[0]

        self.categorical =  np.asarray(categorical , dtype=ITYPE)
        self.categorical_ptr = self.get_idx_ptr(self.categorical)
        self.nc = self.categorical.shape[0]


#

#        self.mises =  np.asarray( mises , dtype=ITYPE)
#        self.mises_ptr = self.get_idx_ptr(self.mises)
# 
#        self.categorical =  np.asarray(categorical , dtype=ITYPE)
#        self.categorical_ptr = self.get_idx_ptr(self.categorical)
#
#
#    
#        self.nm = self.mises.shape[0]
#        self.nc = self.categorical.shape[0]


    cdef ITYPE_t* get_idx_ptr(self, np.ndarray[ITYPE_t, ndim=1, mode='c'] idx):
        return &idx[0]

    cdef inline DTYPE_t dist(self, DTYPE_t* x1, DTYPE_t* x2,
                              ITYPE_t size) nogil except -1:
        cdef DTYPE_t d=0
        cdef DTYPE_t d_=0
        cdef ITYPE_t jj, ii, 
        cdef ITYPE_t ndims
        cdef bint match

        ndims = self.nc + self.nm + self.ng        


        if size !=ndims:
            with gil:
                raise ValueError('SBGower {:g},{:g},{:g}'.format(self.ng, self.nm, self.nc))
                                                                           

        d = 1. 
        match = True
        for ii  from 0 <= ii < self.nc:
            jj = self.categorical_ptr[ii]
            match = match & ( x1[jj] == x2[jj])
        
        if match:
             
            d = 0.
            for ii  from 0 <= ii < self.nm:

                jj = self.mises_ptr[ii]         
                d_ = fmod( (x1[jj]-x2[jj]) + M_PI , 2.*M_PI)-M_PI 
                d_ = d_/(2*M_PI) ## dist range is 0,pi scale to unit lenght
                d = d  + d_*d_ 
       
       
            for ii  from 0 <= ii < self.ng:

                jj = self.gaussian_ptr[ii]
                d_= (x1[jj] - x2[jj])**2
                d = d + d_

            d = sqrt(d)
        else:
               d = 999.

        return d



    cdef inline DTYPE_t rdist(self, DTYPE_t* x1, DTYPE_t* x2,
                             ITYPE_t size) nogil except -1:
        return self.dist(x1, x2, size)

    cdef inline DTYPE_t _rdist_to_dist(self, DTYPE_t rdist) nogil except -1:
        return rdist

    cdef inline DTYPE_t _dist_to_rdist(self, DTYPE_t dist) nogil except -1:
        return dist

    def rdist_to_dist(self, rdist):
        return rdist

    def dist_to_rdist(self, dist):
        return dist 








####cdef class SBGowerDistance(DistanceMetric):
####    r"""Weighted Minkowski Distance
####
####    .. math::
####       D(x, y) = [\sum_i |w_i * (x_i - y_i)|^p] ^ (1/p)
####
####    Weighted Minkowski Distance requires p >= 1 and finite.
####
####    Parameters
####    ----------
####    p : int
####        The order of the norm of the difference :math:`{||u-v||}_p`.
####    w : (N,) array_like
####        The weight vector.
####
####    """
####    def __cinit__(self):
####        self.gidx = np.zeros(1, dtype=ITYPE, order='c')
####        self.gidx_ptr = get_idx_ptr(self.gidx)
####        
####
####
####        self.midx = np.zeros(1, dtype=ITYPE, order='c')
####        self.midx_ptr = get_idx_ptr(self.midx)
####        
####
####        self.cidx = np.zeros(1, dtype=ITYPE, order='c')
####        self.cidx_ptr = get_idx_ptr(self.cidx)
####       
####
####        ##cdef
####        self.dims = np.ones(3, dtype=ITYPE, order='c')
####        self.dims_ptr = get_dim_ptr(self.dims)
######        self.p = 2
######        self.vec = np.zeros(1, dtype=DTYPE, order='c')
######        self.mat = np.zeros((1, 1), dtype=DTYPE, order='c')
######        self.vec_ptr = get_vec_ptr(self.vec)
######        self.mat_ptr = get_mat_ptr(self.mat)
######        self.size = 1
####
####        
####
####    def __init__(self,  gidx,  midx,  cidx, p):
####        pass
####        
####        self.gidx = np.asarray(gidx, dtype=ITYPE)
####        self.gidx_ptr = get_idx_ptr(self.gidx)
####        self.ng = self.gidx.shape[0]
####
####        self.midx = np.asarray(midx, dtype=ITYPE)
####        self.midx_ptr = get_idx_ptr(self.midx)
####        self.nm = self.midx.shape[0]
####
####        self.cidx = np.asarray(cidx, dtype=ITYPE)
####        self.cidx_ptr = get_idx_ptr(self.cidx)
####        self.nc = self.cidx.shape[0]
####
####        
####        self.dims =  np.asarray([self.ng, self.nm, self.nc], dtype=ITYPE)
####        self.dims_ptr = get_dim_ptr(self.dims)
####
####
####        self.p = p
####
####    cdef inline DTYPE_t dist(self, DTYPE_t* x1, DTYPE_t* x2,
####                              ITYPE_t size) nogil except -1:
####        cdef DTYPE_t d=0
####        cdef DTYPE_t d_=0
####        cdef ITYPE_t jj, ii, 
####        cdef ITYPE_t ndims
####        cdef ITYPE_t nc, ng, nm
####        cdef bint match
####
######        ng = self.dims_ptr[0] ## very ugly, found no other tweak
######        nm = self.dims_ptr[1]
######        nc = self.dims_ptr[2]
####        nc = self.nc
####        nm = self.nm
####        ng = self.ng
####        ndims = nc + nm + ng        
####
####
####        if size !=ndims:
####            with gil:
####                raise ValueError('SBGower {:g},{:g},{:g},{:g}'.format(ng,nc,nm, self.p))# dist:  size of {} does not match{}'.format(str(ndims)))
####                                                                           
####
####        d = 1. 
####        match = True
#####       for ii in range(self.nc):
####        for ii  from 0 <= ii < nc:
####            jj = self.cidx_ptr[ii]
####            match = match & ( x1[jj] == x2[jj])
####        
####        if match:
####             
####            d = 0.
####            for ii  from 0 <= ii < nm:
####
####                jj = self.midx_ptr[ii]         
####                d_ = fmod( (x1[jj]-x2[jj]) + M_PI , 2.*M_PI)-M_PI 
####                d_ = d_/(2*M_PI) ## dist range is 0,pi scale to unit lenght
####                d = d  + d_*d_ 
####        
####        
####            for ii  from 0 <= ii < ng:
####
####                jj = self.gidx_ptr[ii]
####                d_= (x1[jj] - x2[jj])**2
####                d = d + d_
####
####            d = sqrt(d)
####        else:
####               d = 999.
####
####        return d
####
####
####
####    cdef inline DTYPE_t rdist(self, DTYPE_t* x1, DTYPE_t* x2,
####                             ITYPE_t size) nogil except -1:
####        return self.dist(x1, x2, size)
####
####    cdef inline DTYPE_t _rdist_to_dist(self, DTYPE_t rdist) nogil except -1:
####        return rdist
####
####    cdef inline DTYPE_t _dist_to_rdist(self, DTYPE_t dist) nogil except -1:
####        return dist
####
####    def rdist_to_dist(self, rdist):
####        return rdist
####
####    def dist_to_rdist(self, dist):
####        return dist 
####

