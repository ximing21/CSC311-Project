from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)
import numpy as np


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
        theta[u] -= lr * (sig - c_ij) * beta[q]
        beta[q] -= lr * (sig - c_ij) * theta[u]
    
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

    N = data['user_id'].nunique()
    D = data['question_id'].nunique()
    theta = np.random.randn(N, 1) * 0.1
    beta = np.random.randn(D, 1) * 0.1

    val_acc_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst


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
    train_data = load_train_csv("./data")
    # You may optionally use the sparse matrix.
    # sparse_matrix = load_train_sparse("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    learning_rates = [0.01, 0.05, 0.1]
    num_iterations = [50, 100, 200]

    best_val_acc = 0
    best_lr = None
    best_iter = None
    best_theta = None
    best_beta = None

    for lr in learning_rates:
        for iters in num_iterations:
            theta, beta, val_acc_lst = irt(train_data, val_data, lr, iters)
            val_acc = val_acc_lst[-1]  # Last value in validation accuracy list

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_lr = lr
                best_iter = iters
                best_theta = theta
                best_beta = beta

    print(f"Best Validation Accuracy: {best_val_acc}")
    print(f"Best Learning Rate: {best_lr}")
    print(f"Best Number of Iterations: {best_iter}")

    test_acc = evaluate(test_data, best_theta, best_beta)
    print(f"Test Accuracy: {test_acc}")

    selected_questions = [0, 1, 2]  # question IDs you want to plot

    import matplotlib.pyplot as plt

    for q in selected_questions:
        theta_range = np.linspace(-3, 3, 100)
        probabilities = sigmoid(theta_range - best_beta[q])

        plt.plot(theta_range, probabilities, label=f'Question {q}')

    plt.xlabel('Theta (Ability)')
    plt.ylabel('P(Correct)')
    plt.legend()
    plt.title('Probability of Correct Response vs. Ability')
    plt.show()


if __name__ == "__main__":
    main()
