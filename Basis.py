
import numpy as np
from math import sqrt, log, pi, exp
import time
import integration

class BasisFunc:
    def __init__(self):
        self.coeff = np.array([])
        self.exp = np.array([])
        self.coord = np.array([])
        self.normFact = 1.0
        self.atomIdx = 0
        self.type = 1
        self.cartPow = np.array([0, 0, 0])

class BasisPairs:
    def __init__(self):
        self.n_pairs = 0
        self.exp_p = np.zeros(0)
        self.exp_mu = np.zeros(0)
        self.coeff_prim = np.zeros(0)
        self.coeff_mo = np.zeros(0)
        self.centers = np.zeros((0, 3))
        self.centers_p = np.zeros((0, 3))
        self.centers_ab = np.zeros((0, 3))
        self.powers = np.zeros((0, 6), dtype=int)
        self.expPreFact = np.zeros(0)
        self.density_mat_idx = np.zeros((0, 0))

class Basis:
    sqrt_pi = sqrt(pi)
    def __init__(self):
        self.func = np.empty(0, dtype=BasisFunc)

    def __overlap_unnorm__(self, i, j, a, b, A, B):
        return integration.overlap_unnorm(i, j, a, b, A, B)
        total = 0.0
        p = a+b
        for k in range(i + 1):
            for l in range(j + 1):
                total += ((a*(A - B))/(a + b))**(j - l)*((b*(-A + B))/(a + b))**(i - k)*p**((-1 - k - l)/2.)*binom(i,k)*binom(j,l)*gamma((1 + k + l)/2.) * np.mod(i + k + 1, 2)
        return total
    def __s_s_overlap__(self, a, b):
        ''' overlap between two s-type gaussians '''
        return (sqrt(2)*a**0.25*b**0.25)/sqrt(a + b)
    def __s_p_overlap__(self, A, B, a, b):
        ''' overlap between an s-type and p-type gaussians 
            ODER MATTERS
        '''
        if A == B:
            return 0.0
        else:
            return (2*sqrt(2)*a**1.25*(A - B))/(sqrt(b**(-1.5))*(a + b)**1.5)
    def __p_s_overlap__(self, A, B, a, b):
        ''' overlap between an p-type and s-type gaussians 
            ODER MATTERS
        '''
        if A == B:
            return 0.0
        else:
            return (2*sqrt(2)*b**1.25*(B - A))/(sqrt(a**(-1.5))*(a + b)**1.5)
    def __p_p_overlap__(self, A, B, a, b):
        ''' overlap between two p-type gaussians 
            ODER MATTERS
        '''
        if A == B:
            return (2*sqrt(2))/(sqrt(a**(-1.5))*sqrt(b**(-1.5))*(a + b)**1.5)
        else:
            return (2*sqrt(2)*(b + a*(1 - 2*b*(A - B)**2))) \
                /(sqrt(a**(-1.5))*sqrt(b**(-1.5))*(a + b)**2.5)
    def _primitive_overlap_(self, A, B, a, b, pow1, pow2):
        ''' overlap between two primitive gaussians '''
        p = a+b 
        mu = a*b/p
        xAB = A - B
        expFactor = exp(-mu*xAB*xAB)
        if pow1 == 0 and pow2 == 0:
            return self.__s_s_overlap__(a, b)*expFactor
        if pow1 == 0 and pow2 == 1:
            return self.__s_p_overlap__(A, B, a, b)*expFactor
        if pow1 == 1 and pow2 == 0:
            return self.__p_s_overlap__(A, B, a, b)*expFactor
        if pow1 == 1 and pow2 == 1:
            return self.__p_p_overlap__(A, B, a, b)*expFactor
        print("HELP")

    def overlapMat(self):
        ''' return the overlap matrix '''
        n_basis = len(self.func)
        overlap = np.zeros(int(n_basis*(n_basis + 1)/2))

        count = 0
        for mu in range(0, n_basis):
            for nu in range(0, mu + 1):
                coeff_mu = self.func[mu].coeff
                exp_mu =   self.func[mu].exp
                coord_mu = self.func[mu].coord
                len_mu = len(exp_mu)

                coeff_nu = self.func[nu].coeff
                exp_nu =   self.func[nu].exp
                coord_nu = self.func[nu].coord
                len_nu = len(exp_nu)

                total = 0.0
                for a in range(len_mu):
                    for b in range(len_nu):

                        
                        Ax, Ay, Az = coord_mu
                        Bx, By, Bz = coord_nu
                        alpha, beta = exp_mu[a], exp_nu[b]
                        ax, ay, az = self.func[mu].cartPow
                        bx, by, bz = self.func[nu].cartPow

                        total += self._primitive_overlap_(Ax, Bx, alpha, beta, ax, bx) \
                               * self._primitive_overlap_(Ay, By, alpha, beta, ay, by) \
                               * self._primitive_overlap_(Az, Bz, alpha, beta, az, bz) \
                               * coeff_mu[a]*coeff_nu[b]

                        #print("OVERLAP: ", total, [Ax, Bx, alpha, beta, ax, bx], coeff_mu[a], coeff_nu[b])
                        #print("OVERLAP: ", total, [Ay, By, alpha, beta, ay, by], coeff_mu[a], coeff_nu[b])
                        #print("OVERLAP: ", total, [Az, Bz, alpha, beta, az, bz], coeff_mu[a], coeff_nu[b])
                        #input()

                                            
                overlap[count] = total
                count += 1

        return overlap

    def cart_powers(self, total):
        out = []
        if total == -1:
            #   should change this to be a recursive deffinition
            return [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]] # sp orbitals
        for i in range(0, total + 1):
            for j in range(total - i + 1):
                out.append([total - i - j, j, i])
        return out

    def Chk2Basis(self, Checkpt):
        self.func.resize(Checkpt.n_basis)
        cartPowers = []
        for n in range(0, 5):
            cartPowers.append(self.cart_powers(n))
        cartPowers.append(self.cart_powers(-1))
        #for n in range(0, 5):
        #    print(cartPowers[n])
        #exit()
        #cartPowers = [
        #             [[0, 0, 0]], # s orbitals
        #             [[1, 0, 0], [0, 1, 0], [0, 0, 1]], # p orbitals
        #             [[2, 0, 0], [0, 2, 0], [0, 0, 2], [1, 1, 0], [1, 0, 1], [0, 1, 1]], # d orbitals
        #             [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]] # sp orbitals
        #             ]
        count = 0
        count_basis = 0
        for n in range(len(Checkpt.atomMap)):
            atom = Checkpt.atomMap[n]
            basisType = Checkpt.shellTypes[n]
            coord = Checkpt.shellCoords[n*3:n*3+3]

            n_prim = Checkpt.primitives[n]
            coeff = Checkpt.coeff[count:count+n_prim]
            exp = Checkpt.exp[count:count+n_prim]

            a = np.outer(exp, exp)
            b = np.array([exp]*len(exp))
            b += b.transpose()

            if basisType == -1:
                n_funcs = 4
            elif basisType == 0:
                n_funcs = 1
            elif basisType == 1:
                n_funcs = 3
            elif basisType == 2:
                n_funcs = 6

            n_funcs = len(cartPowers[basisType])

            for i in range(n_funcs):
                self.func[count_basis] = BasisFunc()
                self.func[count_basis].coord = coord
                self.func[count_basis].exp = exp
                if basisType == -1 and i >= 1:
                    self.func[count_basis].coeff = Checkpt.spCoeff[count:count+n_prim]
                else:
                    self.func[count_basis].coeff = coeff
                self.func[count_basis].atomIdx = atom
                self.func[count_basis].type = basisType
                self.func[count_basis].cartPow = cartPowers[basisType][i]
                #func.normFact = np.sqrt(np.dot(coeff, (expMat/b)**(3/4)*sqrt(8) @ coeff))
                #func.coeff /= func.normFact
                #self.func.append(func)
                count_basis += 1
            count += n_prim

    def create_pairs(self, densityMat):
        dim = len(densityMat)
        dim = int((sqrt(dim*8 + 1) - 1)/2)
        pMat = np.zeros((dim, dim))
        pMat[np.tril_indices(dim)] = densityMat
        pMat += pMat.transpose()
        pMat[np.diag_indices(dim)] = pMat[np.diag_indices(dim)]*0.5

        pairs = BasisPairs()
        n_basis = len(self.func)
        n_elms = np.zeros(n_basis, dtype=int)
        for n in range(n_basis):
            n_elms[n] = len(self.func[n].exp)
        #n_pairs = int(np.sum(n_elms)**2)
        n_pairs = int(np.sum(np.triu(np.outer(n_elms, n_elms))))
        print(" There are {:10d} possible function pairs".format(n_pairs))
        pairs.n_pairs = n_pairs
        pairs.exp_p = np.zeros(n_pairs)
        pairs.exp_mu = np.zeros(n_pairs)
        pairs.coeff_prim = np.zeros(n_pairs)
        pairs.coeff_mo = np.zeros(n_pairs)
        pairs.centers = np.zeros((n_pairs, 6))
        pairs.centers_p = np.zeros((n_pairs, 3))
        pairs.centers_ab = np.zeros((n_pairs, 3))
        pairs.powers = np.zeros((n_pairs, 6), dtype=int)
        pairs.expPreFact = np.zeros(n_pairs)
        pairs.density_mat_idx = np.zeros((n_pairs, 2))
        count = 0

        thresh = 6.0
        s_thresh = thresh/2.0
        min_exp = np.min([self.func[i].exp.min() for i in range(len(self.func))])
        cutoff_dist = sqrt(log((pi/(2*min_exp))**3 * 10**(2*s_thresh)) / min_exp)

        time_overlap = 0
        time_pairs = 0
        
        mult_basis = 1

        count_1 = 0
        count_2 = 0

        for i in range(n_basis):
            for j in range(i, n_basis):
                if i!=j:
                    mult_basis = 2
                else:
                    mult_basis = 1
                for m in range(n_elms[i]):
                    a = self.func[i].exp[m]
                    A = self.func[i].coord
                    ip, jp, kp = self.func[i].cartPow
                    time_overlap -= time.time()
                    norm_x = self.__overlap_unnorm__(ip, ip, a, a, 0, 0)
                    if jp == ip:
                        norm_y = norm_x
                    else:
                        norm_y = self.__overlap_unnorm__(jp, jp, a, a, 0, 0)
                    if kp == ip:
                        norm_z = norm_x
                    elif kp == jp:
                        norm_z = norm_y
                    else:
                        norm_z = self.__overlap_unnorm__(kp, kp, a, a, 0, 0)
                    normFact1 = norm_x*norm_y*norm_z              
                    time_overlap += time.time()
                    normFact1 = 1/sqrt(normFact1)
                    for n in range(n_elms[j]):
                        B = self.func[j].coord
                        b = self.func[j].exp[n]
                        p = a + b
                        mu = a*b/p

                        dist2 = (A[0] - B[0])**2 + (A[1] - B[1])**2 + (A[2] - B[2])**2
                        cutoff_dist_uniq2 = log((pi/(2*p))**3 * 10**(2*s_thresh)) / (2*mu)

                        #   TODO: check cutoff formula and reference
                        if dist2 <= cutoff_dist_uniq2 or cutoff_dist_uniq2 < 0:

                            if dist2 <= cutoff_dist_uniq2: count_1 += 1
                            if cutoff_dist_uniq2 < 0: count_2 += 1

                        #if True:


                            ip, jp, kp = self.func[j].cartPow
                            time_overlap -= time.time()

                            norm_x = self.__overlap_unnorm__(ip, ip, b, b, 0, 0)
                            if jp == ip:
                                norm_y = norm_x
                            else:
                                norm_y = self.__overlap_unnorm__(jp, jp, b, b, 0, 0)
                            if kp == ip:
                                norm_z = norm_x
                            elif kp == jp:
                                norm_z = norm_y
                            else:
                                norm_z = self.__overlap_unnorm__(kp, kp, b, b, 0, 0)
                            normFact2 = norm_x*norm_y*norm_z
                            time_overlap += time.time()
                            normFact2 = 1/sqrt(normFact2)

                            time_pairs -= time.time()
                            pairs.exp_p[count] = p
                            pairs.exp_mu[count] = a*b/p
                            pairs.centers[count] = np.append(A, B)
                            pairs.centers_p[count] = (A*a + B*b)/p
                            pairs.centers_ab[count] = A - B
                            pairs.powers[count] = np.array(list(self.func[i].cartPow) + list(self.func[j].cartPow))
                            pairs.coeff_prim[count] = self.func[i].coeff[m] * self.func[j].coeff[n]*normFact1*normFact2 * mult_basis
                            pairs.coeff_mo[count] = pMat[i, j]
                            pairs.expPreFact[count] = np.exp(-a*b*np.linalg.norm(A - B)**2/p)
                            pairs.density_mat_idx[count] = np.array([i, j])
                            time_pairs += time.time()

                            count += 1

        print(" Overlap time:          {:9.2f}".format(time_overlap))
        print(" Pair Calculation Time: {:9.2f}".format(time_pairs))
        print(" Total Time:            {:9.2f}".format(time_pairs + time_overlap))
        print(" A cutoff threshhold of 10^-{:d} results in {:d} pairs".format(int(thresh), count))
        print(" Minimum exponent: {:g}".format(min_exp))
        print(" Maximum cut-off distance: {:.2f} a.u.".format(cutoff_dist))

        return pairs

    def print_array(self, array):
        n_elms = len(array)
        for n in range(n_elms):
            print("{:15.8E} ".format(array[n]), end='')
            if np.mod(n, 5) == 4:
                print("\n", end='')
        print("\n", end='')
