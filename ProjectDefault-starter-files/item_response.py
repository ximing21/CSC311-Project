from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """Apply sigmoid function."""
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    
    log_lklihood = 0.0
    for i in range(len(data["is_correct"])):
        c_ij = data["is_correct"][i]
        u = data["user_id"][i]
        q = data["question_id"][i]
        sig = sigmoid(theta[u] - beta[q])
        log_lklihood += c_ij * np.log(sig) - (1 - c_ij) * np.log(1 - sig)

    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    for i in range(len(data["is_correct"])):
        c_ij = data["is_correct"][i]
        u = data["user_id"][i]
        q = data["question_id"][i]
        sig = sigmoid(theta[u] - beta[q])

        theta[u] += lr * (c_ij - sig) 
        beta[q] -= lr * (c_ij - sig)
    
    return theta, beta


def irt(data, val_data, lr, iterations):
    """Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """

    N = len(data['user_id'])
    D = len(data['question_id'])
    theta = np.random.randn(N, 1) * 0.05
    beta = np.random.randn(D, 1) * 0.01

    # for part b
    train_log_like = []
    val_log_like = []
 
    val_acc_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_neg_lld = neg_log_likelihood(data=val_data, theta=theta, beta=beta)

        train_log_like.append(neg_lld)
        val_log_like.append(val_neg_lld)
        val_acc_lst.append(score)

        theta, beta = update_theta_beta(data, lr, theta, beta)

        if i % 10 == 0:
            print(f"iteration: {i}, neg_log_likelihood: {neg_lld}, "
                  f"val_acc: {score}")

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, train_log_like, val_log_like


def evaluate(data, theta, beta):
    """Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])


def main():
    train_data = load_train_csv("./ProjectDefault-starter-files/data")
    val_data = load_valid_csv("./ProjectDefault-starter-files/data")
    test_data = load_public_test_csv("./ProjectDefault-starter-files/data")

    # You may optionally use the sparse matrix.
    # sparse_matrix = load_train_sparse("./data")

    # TODO:
    # Tune learning rate and number of iterations. With the implemented 
    # code, report the validation and test accuracy. 
    num_iteration = 120
    lr = 0.001

    theta, beta, val_acc, train_log_like, val_log_like = irt(train_data, val_data, lr, num_iteration)

    # part b
    print(f"hyperparameters: \n num_iteration = "
          f"{num_iteration}, learning rate = {lr}")
    plt.figure()
    plt.plot(range(num_iteration), train_log_like, label='Train Log Likelihood')
    plt.plot(range(num_iteration), val_log_like, label='Validation Log Likelihood')
    plt.xlabel('Iteration')
    plt.ylabel('Negative Log Likelihood')
    plt.legend()
    plt.title('Negative Log Likelihood vs. Iteration')
    plt.savefig("./ProjectDefault-starter-files/plot/log_likelihood_plot.png")
    plt.show()
    
    # part c
    print("val accuracy: ", evaluate(val_data, theta, beta))
    print("test accuracy: ", evaluate(test_data, theta, beta))

    # part d
    selected_questions = [0, 1, 2]  # question IDs

    for q in selected_questions:
        theta_range = np.linspace(-3, 3, 100)
        probabilities = sigmoid(theta_range - beta[q])

        plt.plot(theta_range, probabilities, label=f'Question {q}')

    plt.xlabel('theta')
    plt.ylabel('probability p(c_ij = 1)')
    plt.legend()
    plt.title('Probability of Correct Response vs. Theta')
    plt.savefig("./ProjectDefault-starter-files/plot/prob_correct_response.png")
    plt.close()


if __name__ == "__main__":
    main()
