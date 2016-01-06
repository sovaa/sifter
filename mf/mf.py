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
import os


class ImplicitMF:
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


def save_model(outfile, model):
    npz_file = outfile + '.npz'
    np.savez(npz_file, M=model)
    print("saved model %s." % npz_file)


def load_model(path):
    npz_file = np.load(path)
    print("loaded model from %s" % path)
    return npz_file['M']


def load_matrix(filename_orig):
    filename = filename_orig.rstrip('.txt')
    if os.path.isfile(filename + '.npz'):
        return load_model(filename + '.npz')

    t0 = time.time()
    total = 0.0
    raw_counts = []
    num_users = 0
    num_items = 0
    num_non_zeros = 0
    for i, line in enumerate(open(filename + '.txt', 'r')):
        user, item, count = line.strip().split('\t')
        user = int(user)
        item = int(item)
        count = float(count)
        if count != 0:
            if user > num_users:
                num_users = user
            if item > num_items:
                num_items = item

            raw_counts.append((user, item, count))
            #counts[user, item] = count
            total += count
            num_non_zeros += 1
        if i % 100000 == 0:
            print('loaded %i counts...' % i)

    num_users += 1
    num_items += 1
    num_zeros = num_users * num_items - num_non_zeros
    print("num_users %s, num_items %s" % (num_users, num_items))

    counts = np.zeros((num_users, num_items))
    for user, item, count in raw_counts:
        counts[user, item] = count

    print('loaded %i counts...' % len(counts))
    print('num_zeros / total: %s / %s' % (num_zeros, total))
    print(counts.shape)
    alpha = num_zeros / total
    print('alpha %.2f' % alpha)

    t1 = time.time()
    print('finished loading matrix in %f seconds' % (t1 - t0))
    #counts *= alpha
    save_model(filename, counts)
    return counts


def to_sparse_matrix(matrix):
    t0 = time.time()
    counts = sparse.csr_matrix(matrix)
    t1 = time.time()
    print('sparse matrix creation took %f seconds' % (t1 - t0))
    return counts


def get_remaining_time(start, steps, step):
    avg_step_time = (time.time() - start) / step
    remaining = int(avg_step_time * (steps - step))

    seconds = remaining % 60
    minutes = int((remaining / 60) % 60)
    hours = int(remaining / 60 / 60)
    return "%sh %sm %ss" % (hours, minutes, seconds)


def matrix_factorization(N, M, R, P, Q, K, steps=1000, alpha=0.0002, beta=0.02):
    start = time.time()

    Q = Q.T
    for step in range(steps):
        if step > 0 and step % int(steps/10) == 0:
            remaining = get_remaining_time(start, steps, step)
            average = round((time.time() - start) / step, 2)
            print("on step %s/%s, average step time %sms, remaining time: %s" % (step, steps, average, remaining))

        for i in range(N):
            for j in range(M):
                if R[i, j] > 0:
                    eij = R[i, j] - np.dot(P[i,:],Q[:,j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        #eR = np.dot(P,Q)
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

def get_filename():
    import sys
    if len(sys.argv) > 1:
        return sys.argv[1]
    return 'data.txt'

def run():
    #mf = ImplicitMF(counts, num_iterations=10, num_factors=10, num_threads=8)
    #mf.train_model()

    filename = get_filename()

    counts = load_matrix(filename)
    counts = to_sparse_matrix(counts)

    N, M = counts.shape
    R = counts
    K = 2
    P = np.random.rand(N,K)
    Q = np.random.rand(M,K)
    print("created random P,Q")

    t0 = time.time()
    nP, nQ = matrix_factorization(N, M, R, P, Q, K)
    t1 = time.time()
    print("matrix factorization took %f seconds" % (t1 - t0))

    t0 = time.time()
    nR = np.dot(nP, nQ.T)
    t1 = time.time()
    print("P dot Q took %f seconds" % (t1 - t0))

    save_model(filename.rstrip('.txt') + '.nr', nR)

if __name__ == "__main__":
    run()