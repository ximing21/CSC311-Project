import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)
vsqres = []
tsqres = []

def svd_reconstruct(matrix, k):
    """Given the matrix, perform singular value decomposition
    to reconstruct the matrix.

    :param matrix: 2D sparse matrix
    :param k: int
    :return: 2D matrix
    """
    # First, you need to fill in the missing values (NaN) to perform SVD.
    # Fill in the missing values using the average on the current item.
    # Note that there are many options to do fill in the
    # missing values (e.g. fill with 0).
    new_matrix = matrix.copy()
    mask = np.isnan(new_matrix)
    masked_matrix = np.ma.masked_array(new_matrix, mask)
    item_means = np.mean(masked_matrix, axis=0)
    new_matrix = masked_matrix.filled(item_means)

    # Next, compute the average and subtract it.
    item_means = np.mean(new_matrix, axis=0)
    mu = np.tile(item_means, (new_matrix.shape[0], 1))
    new_matrix = new_matrix - mu

    # Perform SVD.
    Q, s, Ut = np.linalg.svd(new_matrix, full_matrices=False)
    s = np.diag(s)

    # Choose top k eigenvalues.
    s = s[0:k, 0:k]
    Q = Q[:, 0:k]
    Ut = Ut[0:k, :]
    s_root = sqrtm(s)

    # Reconstruct the matrix.
    reconst_matrix = np.dot(np.dot(Q, s_root), np.dot(s_root, Ut))
    reconst_matrix = reconst_matrix + mu
    return np.array(reconst_matrix)


def squared_error_loss(data, u, z):
    """Return the squared-error-loss given the data.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param u: 2D matrix
    :param z: 2D matrix
    :return: float
    """
    loss = 0
    for i, q in enumerate(data["question_id"]):
        loss += (data["is_correct"][i] - np.sum(u[data["user_id"][i]] * z[q])) ** 2.0
    return 0.5 * loss


def update_u_z(train_data, lr, u, z):
    """Return the updated U and Z after applying
    stochastic gradient descent for matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param u: 2D matrix
    :param z: 2D matrix
    :return: (u, z)
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Randomly select a pair (user_id, question_id).
    i = np.random.choice(len(train_data["question_id"]), 1)[0]

    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]

    err = c - np.dot(u[n], z[q])

    u[n] = u[n] + lr * err * z[q]
    z[q] = z[q] + lr * err * u[n]
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return u, z


def als(train_data, k, lr, num_iteration, val_data=None):
    """Performs ALS algorithm, here we use the iterative solution - SGD
    rather than the direct solution.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :return: 2D reconstructed Matrix.
    """
    # Initialize u and z
    u = np.random.uniform(
        low=0, high=1 / np.sqrt(k), size=(len(set(train_data["user_id"])), k)
    )
    z = np.random.uniform(
        low=0, high=1 / np.sqrt(k), size=(len(set(train_data["question_id"])), k)
    )

    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    for i in range(num_iteration):
        u, z = update_u_z(train_data, lr, u, z)
        if val_data is not None and i % 500 == 0:
            tsqre = squared_error_loss(train_data, u, z)
            vsqre = squared_error_loss(val_data, u, z)
            tsqres.append(tsqre)
            vsqres.append(vsqre)

    mat = u.dot(z.T)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return mat


def main():
    train_matrix = load_train_sparse("./data").toarray()
    train_data = load_train_csv("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    #####################################################################
    # TODO:                                                             #
    # (SVD) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    # SVD Part
    ks = [5, 25, 40, 60, 80]

    best_k = None
    best_acc = -float('inf')
    for k in ks:
        reconstructed_matrix = svd_reconstruct(train_matrix, k)
        acc = sparse_matrix_evaluate(val_data, reconstructed_matrix)
        print(f"k:{k}, acc:{acc}")
        if acc > best_acc:
            best_acc = acc
            best_k = k

    print(f"Best k for SVD: {best_k} with acc: {best_acc}")
    print("accuracy with validation data(svd): ", sparse_matrix_evaluate(val_data, svd_reconstruct(train_matrix, best_k)))
    print("accuracy with test data(svd): ", sparse_matrix_evaluate(test_data, svd_reconstruct(train_matrix, best_k)))




    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # (ALS) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################

    ks = [5, 25, 40, 60, 80]
    lrs = [0.0005, 0.005, 0.01, 0.05, 0.1]
    iterations = [5000, 10000, 50000, 100000, 150000]


    acc_param = {}
    accs = []
    for k in ks:
        for lr in lrs:
            for iteration in iterations:
                reconstruct_matrix = als(train_data, k, lr, iteration)
                acc = sparse_matrix_evaluate(val_data, reconstruct_matrix)
                param = f"k:{k} lr:{lr} iteration:{iteration} acc:{acc}"
                print(param)
                acc_param[acc] = param
                accs.append(acc)

    max_param = acc_param[max(accs)]
    print("max is ", max_param)

    #get parameters with max acc
    k = int(max_param.split(" ")[0].strip("k:"))
    lr = float(max_param.split(" ")[1].strip("lr:"))
    iteration = int(max_param.split(" ")[2].strip("iteration:"))

    matrix = als(train_data, k, lr, iteration, val_data)
    print("accuracy with validation data(als): ", sparse_matrix_evaluate(val_data, matrix))
    print("accuracy with test data(als): ", sparse_matrix_evaluate(test_data, matrix))

    iterations_lst = list(range(0, iteration, 500))
    plt.figure()
    plt.plot(iterations_lst, tsqres, label='Train')
    plt.plot(iterations_lst, vsqres, label='Validation')
    plt.xlabel('Iterations')
    plt.ylabel('Squared Error Loss')
    plt.legend()
    plt.show()



    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
