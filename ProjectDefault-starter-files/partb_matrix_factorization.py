import numpy as np
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

def add_weights(data):
    question_id = data["question_id"]
    is_correct = data["is_correct"]
    user_id = data["user_id"]

    dictionary = dict()
    question_count = dict()
    user_count = dict()
    weights = dict() #{question:{user:weight}}

    for id in set(question_id):
        dictionary[id] = [0,0] #dictionary[id][0] = total, [1] for correct
        question_count[id] = 0
        weights[id] = dict()
    for id in set(user_id):
        user_count[id] = 0
        for q in set(question_id):
            weights[q][id] = 0

    for i in range(len(question_id)):
        dictionary[question_id[i]][0] += 1
        if is_correct[i] == 1:
            dictionary[question_id[i]][1] += 1
        question_count[question_id[i]] += 1
    weights_acc = dict()
    for u in user_id:
        user_count[u] += 1

    for key in dictionary:
        if dictionary[key][0] != 0:
            weights_acc[key] = dictionary[key][1] / dictionary[key][0]
        else:
            weights_acc[key] = 0
    mean = np.mean([weights_acc[key] for key in weights_acc])

    for key in dictionary:
        weights_acc[key] = 1 - (np.abs(mean - weights_acc[key]))

    for q in set(question_id):
        w_1 = weights_acc[q]
        for u in set(user_id):
            w_2 = 1 / (question_count[q] * user_count[u])
            weight = w_2 * w_1
            weights[q][u] = 1 - (1 - weight) * 0.05
    return weights


def update_u_z(train_data, lr, u, z, lam, weights):
    i = \
        np.random.choice(len(train_data["question_id"]), 1)[0]

    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]
    weight = weights[q][n]
    err = weight * (c - np.dot(u[n], z[q]))
    u[n] += lr * (err * z[q] - lam * u[n])
    z[q] += lr * (err * u[n] - lam * z[q])
    return u, z

def als(train_data, k, lr, num_iteration, lam, weights, val_data=None):
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
        u, z = update_u_z(train_data, lr, u, z, lam, weights)
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

    weights = add_weights(train_data)

    ks = [5, 25, 40, 60, 80]
    lrs = [0.0005, 0.005, 0.01, 0.05, 0.1]
    iterations = [5000, 10000, 50000, 100000, 150000]
    lam = 0.01

    acc_param = {}
    accs = []
    for k in ks:
        for lr in lrs:
            for iteration in iterations:
                reconstruct_matrix = als(train_data, k, lr, iteration, lam, weights)
                acc = sparse_matrix_evaluate(val_data, reconstruct_matrix)
                param = f"k:{k} lr:{lr} iteration:{iteration} acc:{acc} lam:{lam}"
                print(param)
                acc_param[acc] = param
                accs.append(acc)
    max_param = acc_param[max(accs)]
    print("max is ", max_param)

    # get parameters with max acc
    k = int(max_param.split(" ")[0].strip("k:"))
    lr = float(max_param.split(" ")[1].strip("lr:"))
    iteration = int(max_param.split(" ")[2].strip("iteration:"))

    matrix = als(train_data, k, lr, iteration, lam, weights, val_data)
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



if __name__ == "__main__":
    main()
