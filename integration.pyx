#cython: language_level=3
import numpy as np
cimport numpy as np #pylint: disable=syntax-error
import fcheck_reader
from scipy.special.cython_special cimport hyp1f1, gamma, binom
from libc.math cimport pi, sqrt, exp, fabs, erf
from cython.parallel import prange, parallel
from libcpp cimport bool
import cython
cimport openmp

cdef int count = 0
cdef int print_count = 0
cdef int next_int = 0, this_int = 0
cdef long[::1] checkpoints

cdef void progress(int max, int n, int thread_id) nogil:
    cdef int interval = 50
    global count
    global print_count
    global next_int, this_int
    global checkpoints
    #with gil:
    #    print("Counter: ", count, n)
    if count == 0:
        with gil:
            if thread_id == 0:
                checkpoints = np.array([int(n*max/interval) for n in range(interval)], dtype=int)
                this_int = 0
                next_int = checkpoints[1]
    if count % 100 == 0:
        if count < next_int and count > this_int:
            with gil:
                if thread_id == 0:
                    print(" Progress [", end='')
                    for n in range(print_count):
                        print('#', end='')
                    for n in range(interval - print_count):
                        print(' ', end='')
                    print(']', end='\r')
                    #print(print_count, count, max, count/max, end='\r')
                    print_count += 1
                    this_int = next_int
                    if print_count == interval - 1:
                        next_int = max
                    else:
                        next_int = checkpoints[print_count + 1]
    count += 1

class Hermite:
    def __init__(self):
        self.coeff = np.zeros(0)
        self.mo_coeff = np.zeros(0)
        self.idx = np.zeros((0, 4), dtype=int)

    def mo_coeff_order(self, pairs):
        return pairs.coeff_mo[self.idx[:,0]]

    def calc_hermite_coeff(self, pairs):
        self.coeff, self.idx = calc_hermite_coeff(pairs)

cpdef double overlap_unnorm(int i, int j, double a, double b, double A, double B):
    cdef double total = 0.0
    cdef double p = a+b
    cdef k, l
    for k in range(i + 1):
        for l in range(j + 1):
            total += ((a*(A - B))/(a + b))**(j - l)*((b*(-A + B))/(a + b))**(i - k)*p**((-1 - k - l)/2.)*binom(i,k)*binom(j,l)*gamma((1 + k + l)/2.) * np.mod(i + k + 1, 2)
    return total

def eval_density_old(pairs, point):
    dpoint = np.append(point, point)
    rDiff = dpoint - pairs.centers
    rProd = np.prod(np.power(rDiff, pairs.powers), axis=1)

    preFac = pairs.expPreFact * pairs.coeff_mo * pairs.coeff_prim * rProd
    
    posDiff = point - pairs.centers_p
    r2 = np.linalg.norm(posDiff, axis=1)**2
    exp = np.exp(-pairs.exp_p * r2)
    return np.sum(exp*preFac)

def eval_density_single(pairs, point):
    preFac = pairs.expPreFact * pairs.coeff_mo * pairs.coeff_prim

    dpoint = np.append(point, point)
    rDiff = dpoint - pairs.centers
    rProd = np.prod(np.power(rDiff, pairs.powers), axis=1)

    preFac2 = preFac * rProd
    
    posDiff = point - pairs.centers_p
    r2 = np.linalg.norm(posDiff, axis=1)**2
    exp = np.exp(-pairs.exp_p * r2)
    return np.sum(exp*preFac2)

def eval_density_multi(pairs, point):
    preFac = pairs.expPreFact * pairs.coeff_mo * pairs.coeff_prim


    dpoint = np.append(point, point)
    rDiff = dpoint - pairs.centers
    rProd = np.prod(np.power(rDiff, pairs.powers), axis=1)

    preFac2 = preFac * rProd
    
    posDiff = point - pairs.centers_p
    r2 = np.linalg.norm(posDiff, axis=1)**2
    exp = np.exp(-pairs.exp_p * r2)
    return np.sum(exp*preFac2)

cpdef double boys_hyper(int n, double x) nogil:
    return hyp1f1(n + 0.5, n + 1.5, -x)/(2*n + 1)

cpdef double boys(int n, double x) nogil:
    ''' return Boys function of order n '''
    cdef double total
    cdef Py_ssize_t k
    if x == 0:
        return 1/float((2*n + 1))
    elif n <=0:
        return sqrt(pi/(4*x)) * erf(sqrt(x))
    else:
        if x < 0.01:
            #   if x is small, use a slower, but more accurate method
            return boys_hyper(n, x)
        else:
            return ((2*(n - 1) + 1)*boys(n - 1, x) - exp(-x))/(2*x)

cdef double hermite_poly(int n, double x) nogil:
    if n == 0:
        return 1.0
    elif n < 0:
        return 0.0
    else:
        return 2*x*hermite_poly(n-1, x) - 2*(n-1)*hermite_poly(n-2, x)


def hermite_coeff(t, i, j, exp_mu, exp_p, Xa, Xb, Xp):
    if (t == 0) and (i == 0) and (j == 0):
        Xab = np.linalg.norm(Xa - Xb)
        return exp(-exp_mu*Xab*Xab)
    if t > i+j or t < 0:
        return 0
    if j <= i:
        return          hermite_coeff(t-1, i-1, j, exp_mu, exp_p, Xa, Xb, Xp)/(2*exp_p) \
            + (Xp - Xa)*hermite_coeff(t,   i-1, j, exp_mu, exp_p, Xa, Xb, Xp) \
            + (t+1)*    hermite_coeff(t+1, i-1, j, exp_mu, exp_p, Xa, Xb, Xp)
    else:
        return          hermite_coeff(t-1, i, j-1, exp_mu, exp_p, Xa, Xb, Xp)/(2*exp_p) \
            + (Xp - Xb)*hermite_coeff(t,   i, j-1, exp_mu, exp_p, Xa, Xb, Xp) \
            + (t+1)*    hermite_coeff(t+1, i, j-1, exp_mu, exp_p, Xa, Xb, Xp)


cdef double rInt2(int n, int t, int u, int v, double exp_p, double Xpc, double Ypc, double Zpc) nogil:
    cdef double r2
    ''' From equations 9.9.18 - 9.9.20 in Helgaker, Jorgensen, and Olsen '''
    if n < 0 or t < 0 or u < 0 or v < 0:
        return 0
    if (t == 0) and (u == 0) and (v == 0):
        r2 = Xpc**2 + Ypc**2 + Zpc**2
        return (-2*exp_p)**(n)*boys(n, exp_p*r2)
    if (t != 0):
        return (t-1)*rInt2(n+1, t-2, u, v, exp_p, Xpc, Ypc, Zpc) + Xpc*rInt2(n+1, t-1, u, v, exp_p, Xpc, Ypc, Zpc)
    if (u != 0):
        return (u-1)*rInt2(n+1, t, u-2, v, exp_p, Xpc, Ypc, Zpc) + Ypc*rInt2(n+1, t, u-1, v, exp_p, Xpc, Ypc, Zpc)
    if (v != 0):
        return (v-1)*rInt2(n+1, t, u, v-2, exp_p, Xpc, Ypc, Zpc) + Zpc*rInt2(n+1, t, u, v-1, exp_p, Xpc, Ypc, Zpc)
    #print("HELP: R_INT()")

def calc_hermite_coeff(pairs):
    #   calculate size of arrays
    n_pairs = len(pairs.centers)
    gpx = np.sum(pairs.powers[:, [0, 3]], axis=1)+1
    gpy = np.sum(pairs.powers[:, [1, 4]], axis=1)+1
    gpz = np.sum(pairs.powers[:, [2, 5]], axis=1)+1
    dim = np.sum(gpx * gpy * gpz)

    coeffArray = np.zeros(dim)
    coeffIdx = np.zeros((dim, 4))

    count = 0
    for idx in range(len(pairs.centers)):
        i, k, m, j, l, n = pairs.powers[idx]
        exp_mu = pairs.exp_mu[idx]
        exp_p = pairs.exp_p[idx]
        center_p = pairs.centers_p[idx]
        center = pairs.centers[idx]
        for t in range(0, i + j + 1):
            coeffX = hermite_coeff(t, i, j, exp_mu, exp_p, center[0], center[3], center_p[0])
            for u in range(0, k + l + 1):
                coeffY = hermite_coeff(u, k, l, exp_mu, exp_p, center[1], center[4], center_p[1])
                for v in range(0, m + n + 1):
                    coeffZ = hermite_coeff(v, m, n, exp_mu, exp_p, center[2], center[5], center_p[2])

                    #coeffArray[count] = coeffX * coeffY * coeffZ * pairs.coeff_mo[idx] * pairs.coeff_prim[idx]
                    coeffArray[count] = coeffX * coeffY * coeffZ * pairs.coeff_prim[idx]
                    coeffIdx[count] = [idx, t, u, v]
                    count += 1

    coeffIdx = np.array(coeffIdx, dtype=int)

    #   remove small or zero coefficients
    cutoff = 1E-15
    n_cut = np.sum([abs(x) >= cutoff for x in coeffArray])
    cutArray = np.zeros(n_cut)
    cutIdx = np.zeros((n_cut, 4))
    count = 0
    for n in range(dim):
        if abs(coeffArray[n]) >= cutoff:
            cutArray[count] = coeffArray[n]
            cutIdx[count] = coeffIdx[n]
            count += 1

    cutIdx = np.array(cutIdx, dtype=int)
    return cutArray, cutIdx

@cython.boundscheck(False)  # Deactivate bounds checking
cpdef elec_nuc_pot(pairs, points_in, nuclei, nuc_coords, hermite, hermiteMoCoeff, int dx=0, int dy=0, int dz=0):
    '''
        Calculates the total (electric + nuclear) electrostatic potential
    '''
    coords = nuc_coords.copy()
    coords.resize((int(len(coords)/3), 3))
    cdef Py_ssize_t p
    #   calculate the electric part first
    cdef double[::1] esp = ele_pot3(pairs, points_in, hermite, hermiteMoCoeff, dx, dy, dz)
    cdef double[:, ::1] points = points_in
    if dx==0 and dy==0 and dz==0:
        for p in range(points.shape[0]):
            points2 = points[p] - coords
            dists = np.linalg.norm(points2, axis=1)
            total -= np.sum(nuclei/dists)
            esp[p] = -total
    return esp



@cython.boundscheck(False)  # Deactivate bounds checking
cpdef ele_pot3(pairs, double[:, :] points, hermite, hermiteMoCoeff, int dx=0, int dy=0, int dz=0):
    '''
        Calculates the electric part of the electrostatic potential
    '''
    #cdef double[:, ::1] points = points_in
    cdef double[::1] esp = np.zeros(len(points))
    cdef int idx, t, u, v
    cdef double exp_p, total, integral
    cdef Py_ssize_t p, n, dim
    #cdef np.ndarray point = np.zeros(3), Rcp = np.zeros(3), center_p = np.zeros(3)
    cdef double[::1] coeffProd = hermite.coeff * hermiteMoCoeff
    cdef long[:, ::1] coeffIdx = hermite.idx 
    cdef double[:, ::1] centers_p = pairs.centers_p
    cdef double[::1] exp_p_array = pairs.exp_p
    cdef double Px, Py, Pz
    cdef double Rpq_x = 0, Rpq_y = 0, Rpq_z = 0
    cdef long count

    for p in range(points.shape[0]):
        total = 0
        Px = points[p][0]
        Py = points[p][1]
        Pz = points[p][2]

        dim = coeffProd.shape[0]
        count = 0
        for n in prange(dim, nogil=True, schedule=dynamic):
            if fabs(coeffProd[n]) >= 1E-12:
                count += 1
                idx = coeffIdx[n, 0]
                t = coeffIdx[n, 1] + dx
                u = coeffIdx[n, 2] + dy
                v = coeffIdx[n, 3] + dz
                exp_p = exp_p_array[idx]
                Rpq_x = centers_p[idx][0] - Px
                Rpq_y = centers_p[idx][1] - Py
                Rpq_z = centers_p[idx][2] - Pz
                integral = rInt2(0, t, u, v, exp_p, Rpq_x, Rpq_y, Rpq_z)
                total += integral * coeffProd[n] * 2*pi/exp_p
        
        esp[p] = -total

    return esp

cpdef double coulomb_energy(pairs, hermite, hermiteMoCoeff):
    cdef int idx1, idx2, t, u, v, i, j, k
    cdef double exp_p, exp_q, total, integral, alpha, phase, coeff_m, coeff_n
    cdef Py_ssize_t m, n
    cdef np.ndarray point = np.zeros(3), Rpq = np.zeros(3)
    cdef np.ndarray center_p = np.zeros(3), center_q = np.zeros(3)
    cdef np.ndarray coeffArray = hermite.coeff
    cdef np.ndarray[np.int64_t, ndim=2] coeffIdx = hermite.idx
    cdef np.ndarray mo_coeff = hermiteMoCoeff
    cdef double twoPi5_2 = 2*pi**(5./2.)
    cdef product = 0

    total = 0
    for m in range(coeffArray.shape[0]):
        idx1 = coeffIdx[m, 0]
        coeff_m = coeffArray[m]*mo_coeff[m]

        if abs(coeff_m) >= 1E-14:
            t = coeffIdx[m, 1]
            u = coeffIdx[m, 2]
            v = coeffIdx[m, 3]
            exp_p = pairs.exp_p[idx1]
            center_p = pairs.centers_p[idx1]

            for n in range(coeffArray.shape[0]):
                idx2 = coeffIdx[n, 0]
                coeff_n = coeffArray[n]*mo_coeff[n]

                if abs(coeff_n*coeff_m) >= 1E-14:
                    i = coeffIdx[n, 1]
                    j = coeffIdx[n, 2]
                    k = coeffIdx[n, 3]
                    exp_q = pairs.exp_p[idx2]
                    center_q = pairs.centers_p[idx2]
                    
                    phase = ((-1)**(i + j + k)) * twoPi5_2 /(exp_p * exp_q * sqrt(exp_p + exp_q))

                    product = coeff_m * coeff_n * phase
                    if abs(product) >= 1E-15:
                        Rpq = center_p - center_q
                        alpha = exp_p * exp_q / (exp_p + exp_q)
                        integral = rInt2(0, t+i, u+j, v+k, alpha, Rpq[0], Rpq[1], Rpq[2])
                        total += integral * product
            

    return total*0.5

@cython.boundscheck(False)  # Deactivate bounds checking
cpdef double density_gaussian(double exp_q=0, np.ndarray center_q=None, pairs=None, hermite=None, hermiteMoCoeff=None, int lx=0, int lz=0, int ly=0):
    '''
        Caluclates the Coulomb interaction between the electron density and
        a normalized gaussian density using eq. 19.9.11 in Helgaker et al.
        ONLY DEFINED UP TO P-LIKE GAUSSIANS
    '''
    cdef int idx1, idx2, t, u, v, i, j, k, dim
    cdef double exp_p, total, integral, alpha, phase
    cdef Py_ssize_t m, n
    cdef double[::1] coeffArray = hermite.coeff
    cdef long[:, ::1] coeffIdx = hermite.idx
    cdef double[::1] coeffProd = hermite.coeff * hermiteMoCoeff
    cdef double[:, ::1] centers_p = pairs.centers_p
    cdef double[::1] exp_p_array = pairs.exp_p

    cpdef double Rpq_x, Rpq_y, Rpq_z, P_x, P_y, P_z
    cdef double Q_x = center_q[0]
    cdef double Q_y = center_q[1]
    cdef double Q_z = center_q[2]

    total = 0
    dim = coeffArray.shape[0]
    #for m in range(coeffArray.shape[0]):
    for m in prange(dim, nogil=True, schedule=dynamic):
        if fabs(coeffProd[m]) >= 1E-13:

            idx1 = coeffIdx[m, 0]
            t = coeffIdx[m, 1]
            u = coeffIdx[m, 2]
            v = coeffIdx[m, 3]
            exp_p = exp_p_array[idx1]
            alpha = exp_p * exp_q / (exp_p + exp_q)
            phase = 1/(exp_p * sqrt(exp_p + exp_q))

            Rpq_x = centers_p[idx1][0] - Q_x
            Rpq_y = centers_p[idx1][1] - Q_y
            Rpq_z = centers_p[idx1][2] - Q_z
            integral = rInt2(0, t + lx, u + ly, v + lz, alpha, Rpq_x, Rpq_y, Rpq_z)
            
            total += integral * coeffProd[m] * phase

    '''
        Differes slightly from eq 9.9.11. This implimentation assumes a 
        normalized gaussian rather than a Hermite gaussian, hense the
        extra factor of 2 * pi / sqrt(exp_q).
    '''
    return total * 2 * pi * sqrt(exp_q)

cpdef void coulomb_matrix_slater(double[:, :] coords, double[:] exp_list, double[:, :] c_mat):
    dim = exp_list.shape[0]
    cdef double[3] R_diff
    cdef t1, t2, t3, val
    cdef R, a, b, a2, b2, a2_b2_diff

    for i in range(dim):
        a = exp_list[i]
        a2 = a*a
        for j in range(dim):
            R_diff[0] = coords[i][0] - coords[j][0]
            R_diff[1] = coords[i][1] - coords[j][1]
            R_diff[2] = coords[i][2] - coords[j][2]
            R = sqrt(R_diff[0]**2 + R_diff[1]**2 + R_diff[2]**2)
            b = exp_list[j]
            b2 = b*b
            a2_b2_diff = a2 - b2

            if j >= i:
                if R == 0:
                    c_mat[i][j] = (a*b*(a2 + 3*a*b + b2))/(2.*(a + b)**3)

                else:
                    if a == b:
                        c_mat[i][j] = 1/R - (11*a*exp(-(a*R)))/16. - exp(-(a*R))/R - (3*a2*R*exp(-(a*R)))/16. - (a**3*R**2*exp(-(a*R)))/48.
                    elif abs(a - b) < 0.1:
                        t1 = ((a - b)*(15 + 15*b*R + 6*b2*R**2 + b**3*R**3)*exp(-(b*R)))/96.
                        t2 = ((a - b)**2*(30 + 30*b*R + 15*b2*R**2 + 5*b**3*R**3 + b**4*R**4)*exp(-(b*R)))/(320.*b)
                        t3 = (exp(-(b*R))*(-48 - 33*b*R - 9*b2*R**2 - b**3*R**3 ))/(48.*R)
                        val = t1 - t2 + t3 + 1/R
                        c_mat[i][j] = val
                    else:
                        c_mat[i][j] = 1/R - ((a*b**4)/(2.*a2_b2_diff**2) - (-3*a2*b**4 + b**6)/(a2_b2_diff**3*R))*exp(-(a*R)) - ((a**4*b)/(2.*(-a2 + b2)**2) - (a**6 - 3*a**4*b2)/((-a2 + b2)**3*R))*exp(-(b*R))
            else:
                c_mat[i][j] = c_mat[j][i]


cpdef void coulomb_matrix_slater_exp_deriv(double[:, :] coords, double[:] exp_list, double[:, :] c_mat, double[:, :] exp_deriv):
    dim = exp_list.shape[0]
    cdef double[3] R_diff
    cdef t1, t2, t3, val
    cdef R, a, b, a2, b2, a2_b2_diff


    for i in range(dim):
        a = exp_list[i]
        a2 = a*a
        for j in range(dim):
            R_diff[0] = coords[i][0] - coords[j][0]
            R_diff[1] = coords[i][1] - coords[j][1]
            R_diff[2] = coords[i][2] - coords[j][2]
            R = sqrt(R_diff[0]**2 + R_diff[1]**2 + R_diff[2]**2)
            b = exp_list[j]
            b2 = b*b
            a2_b2_diff = a2 - b2

            if j >= i:
                if R == 0:
                    c_mat[i][j] = (a*b*(a2 + 3*a*b + b2))/(2.*(a + b)**3)

                else:
                    if a == b:
                        c_mat[i][j] = 1/R - (11*a*exp(-(a*R)))/16. - exp(-(a*R))/R - (3*a2*R*exp(-(a*R)))/16. - (a**3*R**2*exp(-(a*R)))/48.
                    elif abs(a - b) < 0.1:
                        t1 = ((a - b)*(15 + 15*b*R + 6*b2*R**2 + b**3*R**3)*exp(-(b*R)))/96.
                        t2 = ((a - b)**2*(30 + 30*b*R + 15*b2*R**2 + 5*b**3*R**3 + b**4*R**4)*exp(-(b*R)))/(320.*b)
                        t3 = (exp(-(b*R))*(-48 - 33*b*R - 9*b2*R**2 - b**3*R**3 ))/(48.*R)
                        val = t1 - t2 + t3 + 1/R
                        c_mat[i][j] = val
                        #print(val, a, b, t1, t2, t3)
                    else:
                        c_mat[i][j] = 1/R - ((a*b**4)/(2.*a2_b2_diff**2) - (-3*a2*b**4 + b**6)/(a2_b2_diff**3*R))*exp(-(a*R)) - ((a**4*b)/(2.*(-a2 + b2)**2) - (a**6 - 3*a**4*b2)/((-a2 + b2)**3*R))*exp(-(b*R))
            else:
                c_mat[i][j] = c_mat[j][i]

            if R == 0:
                if a == b:
                    exp_deriv[i][j] = 0.5*5/16
                else:
                    exp_deriv[i][j] = (b**3*(4*a + b))/(2.*(a + b)**4)
            else:
                if a == b:
                    exp_deriv[i][j] = 0.5*((15 + 15*a*R + 6*a2*R**2 + a**3*R**3)*exp(-(a*R)))/48.
                elif abs(a-b) < 0.1:
                    exp_deriv[i][j] = ((15 + 15*b*R + 6*b2*R**2 + b**3*R**3)*exp(-(b*R)))/96. - ((a - b)*(30 + 30*b*R + 15*b2*R**2 + 5*b**3*R**3 + b**4*R**4)*exp(-(b*R)))/(160.*b)
                else:
                    exp_deriv[i][j] = -(((-2*a2*b**4)/a2_b2_diff**3 + b**4/(2.*a2_b2_diff**2) + (6*a*b**4)/(a2_b2_diff**3*R) + (6*a*(-3*a2*b**4 + b**6))/(a2_b2_diff**4*R))*exp(-(a*R))) + ((a*b**4)/(2.*a2_b2_diff**2) - (-3*a2*b**4 + b**6)/(a2_b2_diff**3*R))*R*exp(-(a*R)) - ((2*a**5*b)/(-a2 + b2)**3 + (2*a**3*b)/(-a2 + b2)**2 - (6*a**5 - 12*a**3*b2)/((-a2 + b2)**3*R) - (6*a*(a**6 - 3*a**4*b2))/((-a2 + b2)**4*R))*exp(-(b*R))
            


cpdef void coulomb_matrix(double[:, :] coords, double[:] exp_list, int[:] type_list, int[:] power_list, double[:, :] c_mat, double[:, :] exp_deriv):

    dim = exp_list.shape[0]
    two_sqrt_pi = 2 / sqrt(pi)
    cdef double[3] R_diff
    #   NOTE: the derivatives are w.r.t. exp_ap
    
    for i in range(dim):
    #for i in [0]:
        exp_a = exp_list[i]
        for j in range(i, dim):
            R_diff[0] = coords[i][0] - coords[j][0]
            R_diff[1] = coords[i][1] - coords[j][1]
            R_diff[2] = coords[i][2] - coords[j][2]
            dist = sqrt(R_diff[0]**2 + R_diff[1]**2 + R_diff[2]**2)
            
            exp_b = exp_list[j]
            exp_ab = exp_a * exp_b / (exp_a + exp_b)
            sqrt_exp = sqrt(exp_ab)
            x = sqrt_exp * dist

            #   s - s integration
            if type_list[i] == 0 and type_list[j] == 0:
                if dist == 0:
                    c_mat[i][j] = 2 * sqrt_exp /sqrt(pi)
                    exp_deriv[i][j] = 1 / (sqrt(pi) * sqrt_exp)
                else:
                    
                    c_mat[i][j] = sqrt_exp * erf(x)/x
                    exp_deriv[i][j] = exp(-x*x) /(sqrt(pi) * sqrt_exp)
                    #print(dist, i, j, c_mat[i][j] )

                c_mat[j][i] = c_mat[i][j]
                exp_deriv[j][i] = exp_deriv[i][j] 
            
            elif (type_list[i] == 0 and type_list[j] == 1) or (type_list[i] == 1 and type_list[j] == 0):
                if dist == 0:
                    c_mat[i][j] = 0
                    c_mat[j][i] = 0

                    exp_deriv[i][j] = 0
                    exp_deriv[j][i] = 0
                else:
                    x2 = x*x
                    B1 = (erf(x)/x - two_sqrt_pi * exp(-x2)) / x2
                    idx = max(power_list[i], power_list[j])
                    R = R_diff[idx]

                    if type_list[i] == 1:
                        R *= -1

                    c_mat[i][j] = R * exp_ab * sqrt_exp * B1
                    c_mat[j][i] = -c_mat[i][j]

                    exp_deriv[i][j] = R*sqrt_exp*two_sqrt_pi * exp(-x2)
                    exp_deriv[j][i] = -exp_deriv[i][j]
            
            elif (type_list[i] == 1 and type_list[j] == 1):
                a = power_list[i]
                b = power_list[j]
                if dist == 0:
                    c_mat[i][j] = (a == b)* exp_ab * sqrt_exp * two_sqrt_pi * 2 / 3
                    c_mat[j][i] = c_mat[i][j]

                    exp_deriv[i][j] = (a == b)*sqrt_exp*two_sqrt_pi
                    exp_deriv[j][i] = exp_deriv[i][j]
                else:
                    x2 = x*x
                    B1 = (  erf(x)/x - two_sqrt_pi *              exp(-x2)) / x2
                    B2 = (3*erf(x)/x - two_sqrt_pi * (3 + 2*x2) * exp(-x2)) /(x2*x2)
                    #B2 = 3*B1/x2 - 2*exp(-x2)*two_sqrt_pi/x2

                    c_mat[i][j] =((a == b)* exp_ab * sqrt_exp * B1 - exp_ab**3 * R_diff[a] * R_diff[b] * B2)
                    c_mat[j][i] = c_mat[i][j]

                    exp_deriv[i][j] = (a == b)*two_sqrt_pi*sqrt_exp*exp(-x2)
                    exp_deriv[i][j] -= R_diff[a] * R_diff[b] * ( 2*two_sqrt_pi*exp_ab*exp_ab*exp(-x2) + exp_ab*exp_ab*B2/2)
                    exp_deriv[j][i] = exp_deriv[i][j]
                
            else:
                print("HELP")
            #if i != j:
            #    exp_deriv[i][j] *= 0.5



@cython.boundscheck(False)  # Deactivate bounds checking
cpdef double overlap_dens_gauss(double exp_q=0, np.ndarray center_q=None, pairs=None, hermite=None, hermiteMoCoeff=None):   
    cdef int idx1, idx2, t, u, v, i, j, k, m, dim
    cdef double exp_p, total, integral, alpha, alpha_sqrt, phase, exp_pq
    cdef double herm_x, herm_y, herm_z
    cdef long[:, ::1] coeffIdx = hermite.idx
    cdef double[::1] coeffProd = hermite.coeff * hermiteMoCoeff
    cdef double[::1] exp_p_array = pairs.exp_p
    cdef double[:, ::1] centers_p = pairs.centers_p
    cdef double Rpq_x = 0, Rpq_y = 0, Rpq_z = 0
    cdef double Q_x = center_q[0]
    cdef double Q_y = center_q[1]
    cdef double Q_z = center_q[2]

    #cdef double[:, ::1] Rpq_array = pairs.centers_p - center_q

    exp_pq = 1; exp_p = 0; alpha_sqrt = 1; alpha = 1
    cdef int last_idx
    dim = coeffProd.shape[0]

    total = 0
    #with nogil, parallel():
    last_idx = -1
    for m in prange(dim, nogil=True, schedule=dynamic):
        if fabs(coeffProd[m]) >= 1E-8:
            idx1 = coeffIdx[m, 0]
            t = coeffIdx[m, 1]
            u = coeffIdx[m, 2]
            v = coeffIdx[m, 3]

            #if last_idx != idx1:
            exp_p = exp_p_array[idx1]
            alpha = exp_p * exp_q / (exp_p + exp_q)
            alpha_sqrt = sqrt(alpha)

            Rpq_x = centers_p[idx1][0] - Q_x
            Rpq_y = centers_p[idx1][1] - Q_y
            Rpq_z = centers_p[idx1][2] - Q_z
            exp_pq = exp(-alpha*(Rpq_x*Rpq_x + Rpq_y*Rpq_y + Rpq_z*Rpq_z))

            phase = (-1)**(t+u+v) * sqrt(alpha / exp_p)**3
            herm_x = hermite_poly(t, alpha_sqrt*Rpq_x)
            herm_y = hermite_poly(u, alpha_sqrt*Rpq_y)
            herm_z = hermite_poly(v, alpha_sqrt*Rpq_z)
            
            
            total += herm_x * herm_y * herm_z * coeffProd[m] * exp_pq * phase * alpha_sqrt**(t+u+v)
            last_idx = idx1

    return total

@cython.boundscheck(False)  # Deactivate bounds checking
cpdef double overlap_dens_dens(pairs=None, hermite=None, hermiteMoCoeff=None):
#cpdef double overlap_dens_dens(double[:, ::1] centers_p, pairs=None, hermite=None, hermiteMoCoeff=None): 
    cdef int idx1, idx2, t, u, v, i, j, k, tuv
    cdef int last_idx, last_i, last_j, last_k
    cdef Py_ssize_t m, n, dim
    cdef double exp_p, exp_q, total, integral, alpha, alpha_sqrt, phase, exp_pq
    cdef double herm_x = 0, herm_y = 0, herm_z = 0
    cdef long[:, ::1] coeffIdx = hermite.idx
    cdef double[::1] coeffArray = hermite.coeff
    cdef double[::1] coeffProd = hermite.coeff * hermiteMoCoeff
    cdef double[:, ::1] centers_p = pairs.centers_p.data
    cdef double[::1] exp_p_array = pairs.exp_p

    cdef double P_x, P_y, P_z, Q_x, Q_y, Q_z
    cdef double coeff1, coeff2
    cdef double Rpq_x = 0, Rpq_y = 0, Rpq_z = 0

    total = 0
    last_idx = -1; last_i = -1; last_j = -1; last_k = -1
    # avoid possible unitialization warning
    exp_pq = 1; exp_q = 0; alpha_sqrt = 1
    dim = coeffArray.shape[0]

    for m in prange(dim, nogil=True):
        progress(dim, m, openmp.omp_get_thread_num())
        if fabs(coeffProd[m]) >= 1E-8:
            idx1 = coeffIdx[m, 0]
            t = coeffIdx[m, 1]
            u = coeffIdx[m, 2]
            v = coeffIdx[m, 3]
            exp_p = exp_p_array[idx1]
            phase = (-1)**(t+u+v)
            coeff1 = coeffProd[m]
            P_x = centers_p[idx1][0]
            P_y = centers_p[idx1][1]
            P_z = centers_p[idx1][2]
            tuv = t+u+v
            for n in range(dim):
                if fabs(coeffProd[n]*coeff1) >= 1E-8:
                    idx2 = coeffIdx[n, 0]
                    i = coeffIdx[n, 1]
                    j = coeffIdx[n, 2]
                    k = coeffIdx[n, 3]
                    coeff2 = coeffProd[n]
                    if last_idx != idx2:
                        exp_q = exp_p_array[idx2]
                        alpha = exp_p * exp_q / (exp_p + exp_q)
                        alpha_sqrt = sqrt(alpha)

                        Q_x = centers_p[idx2][0]
                        Q_y = centers_p[idx2][1]
                        Q_z = centers_p[idx2][2]

                        Rpq_x = alpha_sqrt*(P_x - Q_x)
                        Rpq_y = alpha_sqrt*(P_y - Q_y)
                        Rpq_z = alpha_sqrt*(P_z - Q_z)
                        exp_pq = exp(-(Rpq_x*Rpq_x + Rpq_y*Rpq_y + Rpq_z*Rpq_z))

                    
                    if i != last_i or idx2 != last_idx:
                        herm_x = hermite_poly(t + i, Rpq_x)
                    if j != last_j or idx2 != last_idx:
                        herm_y = hermite_poly(u + j, Rpq_y)
                    if k != last_k or idx2 != last_idx:
                        herm_z = hermite_poly(v + k, Rpq_z)
                
                    total += herm_x * herm_y * herm_z * coeff1*coeff2 * exp_pq * phase * alpha_sqrt**(tuv+i+j+k) * sqrt(pi/(exp_p + exp_q))**3
                    last_idx = idx2
                    last_i = i; last_j = j; last_k = k

    return total


def integrate_basis_func(pairs, point, start, end):
    dpoint = np.append(point, point)
    rDiff = dpoint - pairs.centers[start:end]
    rProd = np.prod(np.power(rDiff, pairs.powers[start:end]), axis=1)

    preFac = pairs.expPreFact[start:end] * pairs.coeff_mo[start:end] * pairs.coeff_prim[start:end] * rProd
    print(rProd)
    
    posDiff = point - pairs.centers_p[start:end]
    r2 = np.linalg.norm(posDiff, axis=1)**2
    exp = np.exp(-pairs.exp_p[start:end] * r2)

    return np.sum(exp*preFac)