"""
experimening with different matrix factorization techniques



Original code from Chris Johnson:
https://github.com/MrChrisJohnson/implicit-mf

Multithreading added by Thierry Bertin-Mahieux (2014)
"""

import copy
import numpy as np
import scipy.sparse as sparse
import scipy.linalg
from scipy.sparse.linalg import spsolve
from multiprocessing import Process, Queue
import time

def load_matrix(filename, num_users, num_items):
    t0 = time.time()
    counts = np.zeros((num_users, num_items))
    total = 0.0
    num_zeros = num_users * num_items
    for i, line in enumerate(open(filename, 'r')):
        user, item, count = line.strip().split('\t')
        user = int(user)
        item = int(item)
        count = float(count)
        if user >= num_users:
            continue
        if item >= num_items:
            continue
        if count != 0:
            counts[user, item] = count
            total += count
            num_zeros -= 1
        if i % 100000 == 0:
            print('loaded %i counts...' % i)
    print('loaded %i counts...' % len(counts))
    alpha = num_zeros / total
    print('alpha %.2f' % alpha)
    #counts *= alpha
    #counts = sparse.csr_matrix(counts)
    t1 = time.time()
    print('Finished loading matrix in %f seconds' % (t1 - t0))
    return counts


class ImplicitMF():

    def __init__(self, counts, num_factors=40, num_iterations=30,
                 reg_param=0.8, num_threads=1):
        self.counts = counts
        self.num_users = counts.shape[0]
        self.num_items = counts.shape[1]
        self.num_factors = num_factors
        self.num_iterations = num_iterations
        self.reg_param = reg_param
        self.num_threads = num_threads

    def train_model(self):
        self.user_vectors = np.random.normal(size=(self.num_users,
                                                   self.num_factors))
        self.item_vectors = np.random.normal(size=(self.num_items,
                                                   self.num_factors))

        for i in range(self.num_iterations):
            t0 = time.time()

            user_vectors_old = copy.deepcopy(self.user_vectors)
            item_vectors_old = copy.deepcopy(self.item_vectors)

            self.user_vectors = self.iteration(True, sparse.csr_matrix(self.item_vectors))
            self.item_vectors = self.iteration(False, sparse.csr_matrix(self.user_vectors))
            t1 = time.time()
            norm_diff = scipy.linalg.norm(user_vectors_old - self.user_vectors) + scipy.linalg.norm(item_vectors_old - self.item_vectors)

            if i % int(self.num_iterations/5.0) == 0:
                print('norm difference epoch %s: %s' % (i, norm_diff))

        print(np.dot(self.user_vectors, self.item_vectors.T))

    def iteration(self, user, fixed_vecs):
        num_solve = self.num_users if user else self.num_items
        num_fixed = fixed_vecs.shape[0]
        YTY = fixed_vecs.T.dot(fixed_vecs)
        eye = sparse.eye(num_fixed)
        lambda_eye = self.reg_param * sparse.eye(self.num_factors)
        solve_vecs = np.zeros((num_solve, self.num_factors))

        batch_size = int(np.ceil(num_solve * 1. / self.num_threads))
        idx = 0
        processes = []
        done_queue = Queue()
        while idx < num_solve:
            min_i = idx
            max_i = min(idx + batch_size, num_solve)
            p = Process(target=self.iteration_one_vec,
                        args=(user, YTY, eye, lambda_eye, fixed_vecs, min_i, max_i, done_queue))
            p.start()
            processes.append(p)
            idx += batch_size

        cnt_vecs = 0
        while True:
            is_alive = False
            for p in processes:
                if p.is_alive():
                    is_alive = True
                    break
            if not is_alive and done_queue.empty():
                break
            time.sleep(.1)
            while not done_queue.empty():
                res = done_queue.get()
                i, xu = res
                solve_vecs[i] = xu
                cnt_vecs += 1
        assert cnt_vecs == len(solve_vecs)

        done_queue.close()
        for p in processes:
            p.join()

        return solve_vecs

    def iteration_one_vec(self, user, YTY, eye, lambda_eye, fixed_vecs, min_i, max_i, output):
        t = time.time()
        cnt = 0
        for i in range(min_i, max_i):
            if user:
                counts_i = self.counts[i].toarray()
            else:
                counts_i = self.counts[:, i].T.toarray()
            CuI = sparse.diags(counts_i, [0])
            pu = counts_i.copy()
            pu[np.where(pu != 0)] = 1.0
            YTCuIY = fixed_vecs.T.dot(CuI).dot(fixed_vecs)
            YTCupu = fixed_vecs.T.dot(CuI + eye).dot(sparse.csr_matrix(pu).T)
            xu = spsolve(YTY + YTCuIY + lambda_eye, YTCupu)
            output.put((i, list(xu)))
            cnt += 1
            if cnt % 1000 == 0:
                print('Solved %d vecs in %d seconds (one thread)' % (cnt, time.time() - t))
        output.close()

def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in range(steps):
        for i in range(N):
            for j in range(M):
                if R[i, j] > 0:
                    eij = R[i, j] - np.dot(P[i,:],Q[:,j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = np.dot(P,Q)
        e = 0
        for i in range(N):
            for j in range(M):
                if R[i, j] > 0:
                    e = e + pow(R[i, j] - np.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
        if e < 0.001:
            break
    return P, Q.T

if __name__ == "__main__":
    N = 3
    M = 10
    counts = load_matrix('data.txt', N, M)
    #mf = ImplicitMF(counts, num_iterations=10, num_factors=10, num_threads=8)
    #mf.train_model()
    print()

    R = counts

    K = 2

    P = np.random.rand(N,K)
    Q = np.random.rand(M,K)

    nP, nQ = matrix_factorization(R, P, Q, K)
    nR = np.dot(nP, nQ.T)
    print(nR)






