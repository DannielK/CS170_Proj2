import random
from problem import Problem
from forward_selection import forward_selection
from backward_elimination import backward_elimination

# intro
print("Welcome to group 27's Feature Selection Algorithm.\n")

# ask for the number of features
feature_num = input("Please enter total number of features: ")
while not feature_num.isdigit():
    feature_num = input("Invalid input, type an integer")
feature_num = int(feature_num)

# print the algorithm choices
print("\nType the number of the algorithm you want to run.\n"
        "\tForward Selection\n"
        "\tBackward Elimination\n")

# ask for which algorithm to run
chosen_alg = int(input())
algo_functions = {
    1: forward_selection,
    2: backward_elimination
}
while chosen_alg not in algo_functions:
    chosen_alg = int(input("Invalid input, type 1 or 2"))

# list of dataset files
file_list = {
    1: "CS170_Spring_2024_Small_data__27.txt",
    2: "CS170_Spring_2024_Large_data__27.txt",
    3: "small-test-dataset-1.txt",
    4: "large-test-dataset-1.txt",
}

print("\nType the number of the dataset file you want to run.\n"
        "\tCS170_Spring_2024_Small_data__27\n"
        "\tCS170_Spring_2024_Large_data__27\n"
        "\tsmall-test-dataset-1\n"
        "\tlarge-test-dataset-1.txt\n")
file_name = int(input())

# print the random starting accuracy
random_accuracy = round(random.uniform(0,100),1)
print("\n\n\nUsing no features and \"random\" evaluation, I get an accuracy of " + str(random_accuracy) + "%\n"
      "\nBeginning search.\n")

problem = Problem(feature_num)

solution = algo_functions[chosen_alg](problem, file_list[file_name])

if random_accuracy > 100*solution[1]:
    print("Initial random accuracy: " + str(random_accuracy) + "was higher than the algorithm's solution: " + "{}".format(solution[0]) + ", which has an accuracy of " + str(solution[1]) + "%")
else:
    print("Finished search!! The best feature subset is " + "{}".format(solution[0]) + ", which has an accuracy of " + str(100*solution[1]) + "%")