from problem import Problem
from forward_selection import forward_selection
from backward_elimination import backward_elimination
import time

# intro
print("Welcome to group 27's Feature Selection Algorithm.\n")

# list of dataset files
file_list = {
    1: "CS170_Spring_2024_Small_data__27.txt",
    2: "CS170_Spring_2024_Large_data__27.txt",
    3: "small-test-dataset-1.txt",
    4: "large-test-dataset-1.txt",
}

# ask for which dataset to run
print("\nType the number of the dataset file you want to run.\n"
        "\tCS170_Spring_2024_Small_data__27\n"
        "\tCS170_Spring_2024_Large_data__27\n"
        "\tsmall-test-dataset-1\n"
        "\tlarge-test-dataset-1\n")
file_name = int(input())

# list of algorithms
algo_functions = {
    1: forward_selection,
    2: backward_elimination
}

# ask for which algorithm to run
print("\nType the number of the algorithm you want to run.\n"
        "\tForward Selection\n"
        "\tBackward Elimination\n")
chosen_alg = int(input())

# print the number of features and instances in the dataset
data_map = {}
num_features = 0
num_instances = 0
num_classes = {}
with open(file_list[file_name], 'r') as file:
    for line in file:
        # split the line by spaces and get the features
        instance = line.strip().split()
        data_map[num_instances] = (int(float(instance[0])), )
        #check the class label  and increment the count
        if int(float(instance[0])) in num_classes:
            num_classes[int(float(instance[0]))] += 1
        else:
            num_classes[int(float(instance[0]))] = 1
        num_instances += 1
    num_features = len(instance) - 1

print("\nThis dataset has", num_features, "features (not including the class attribute), with", num_instances, "instances.\n")

# print the accuracy of default rate
default_rate_result = max(num_classes.values())/num_instances * 100
print("Running nearest neighbor with no features (default rate), using \"leaving-one-out\" evaluation, I get an accuracy of", str(default_rate_result), "%\n")

problem = Problem(num_features)

# record start time
overall_start_time = time.time()

solution = algo_functions[chosen_alg](problem, file_list[file_name])

# record end time
overall_end_time = time.time()
overall_time_taken = overall_end_time - overall_start_time
print("Time taken: " + str(overall_time_taken) + " seconds\n")

if default_rate_result > 100*solution[1]:
    print("Default rate accuracy: " + str(default_rate_result) + "was higher than the algorithm's solution: " + "{}".format(solution[0]) + ", which has an accuracy of " + str(solution[1]) + "%")
else:
    print("Finished search!! The best feature subset is " + "{}".format(solution[0]) + ", which has an accuracy of " + str(100*solution[1]) + "%")