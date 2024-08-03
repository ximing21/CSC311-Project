# TODO: complete this file.
# TODO: complete this file.
import random

import numpy as np
from matrix_factorization import *
from item_response import *
from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)

def matrix_factorization(data):
    reconstruct_matrix = als(data, 25, 0.005, 150000)
    return reconstruct_matrix

def knn():
    pass

def item_response():
    pass




def boostrapping(data):
    generated_data = {'user_id': [], 'question_id': [], 'is_correct': []}
    sampled_data = np.random.randint(len(data['user_id']), size=len(data['user_id']))

    n = len(data["user_id"])
    index = np.random.randint(n, size=len(data['user_id']))
    user_id = data['user_id']
    question_id = data['question_id']
    is_correct = data['is_correct']
    generated_data = {
        'user_id': [user_id[i] for i in index],
        'question_id': [question_id[i] for i in index],
        'is_correct': [is_correct[i] for i in index]
    }
    return generated_data





def main():
    train_data = load_train_csv("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    data_a = boostrapping(train_data)
    data_b = boostrapping(train_data)
    data_c = boostrapping(train_data)

    matrix_a = matrix_factorization(data_a)
    matrix_b = matrix_factorization(data_b)
    matrix_c = matrix_factorization(data_c)
    # matrix_a = item_response(data_a)
    # matrix_b = item_response(data_b)
    # matrix_c = item_response(data_c)


    acc_val_a = sparse_matrix_evaluate(test_data, matrix_a)
    acc_val_b = sparse_matrix_evaluate(test_data, matrix_b)
    acc_val_c = sparse_matrix_evaluate(test_data, matrix_c)
    avg_val = (acc_val_a + acc_val_b + acc_val_c) / 3
    print(f"avg acc for validation is:{avg_val}")

    acc_test_a = sparse_matrix_evaluate(test_data, als(data_a, 25, 0.05, 150000))
    acc_test_b = sparse_matrix_evaluate(test_data, als(data_b, 25, 0.05, 150000))
    acc_test_c = sparse_matrix_evaluate(test_data, als(data_c, 25, 0.05, 150000))
    avg_test = (acc_test_a + acc_test_b + acc_test_c) / 3
    print(f"avg acc for test is:{avg_test}")




if __name__ == "__main__":
    main()
