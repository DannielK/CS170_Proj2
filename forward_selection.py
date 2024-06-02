from problem import Problem
from leave_one_out_validator import leave_one_out_validator

def read_file(file_name, features_list):
    data_map = {}
    features_mean = [float("0")] * len(features_list)
    features_std = [float("0")] * len(features_list)
    num_instances = 0.0

    # calculate the mean for the standard deviation of the features
    with open(file_name, "r") as file:
        for line in file:
            # split the line by spaces and get the features
            instance = line.strip().split()
            for i, feature in enumerate(features_list):
                feature_value = float(instance[feature])
                features_mean[i] += feature_value
            num_instances += 1.0
        # calculate the mean of the features
        for i in range(len(features_mean)):
            features_mean[i] /= num_instances

    # calculate the standard deviation of the features
    for i, feature in enumerate(features_list):
        sum_squared_diff = [float("0")] * len(features_list)
        diff = 0.0
        with open(file_name, "r") as file:
            for line in file:
                instance = line.strip().split()
                feature_value = float(instance[feature])
                diff = feature_value - features_mean[i]
                sum_squared_diff[i] += diff**2
        features_std[i] = (sum_squared_diff[i] / (num_instances - 1.0)) ** 0.5

    # read and normalize the data
    with open(file_name, "r") as file:
        # read_start_time = time.time()
        instance_index = 0
        for line in file:
            # split the line by spaces and get the features
            instance = line.strip().split()
            # get the class label from the first element in the list and convert it to an integer
            class_label = int(float(instance[0]))
            # create a tuple with the class label
            instance_tuple = (class_label,)
            # get the features from the rest of the elements in the list
            for i, feature in enumerate(features_list):
                feature_value = float(instance[feature])
                # normalize the feature value
                normalized_value = (feature_value - features_mean[i]) / features_std[i]
                # add the normalized value to the tuple
                instance_tuple += (normalized_value,)
            # add the tuple to the data map
            data_map[instance_index] = instance_tuple
            # increment the instance index
            instance_index += 1

    return data_map


def forward_selection(problem: Problem, filename) -> tuple[tuple, float]:
    # Create local alias
    accuracy_map = problem.set_accuracy_map
    current_accuracy = 0.0
    while len(problem.features_remaining):
        # Generate new subset
        problem.new_subsets("Forward")

        # Set accuracies
        for subset in problem.set_accuracy_map:
            features_list = []
            for feature in subset:
                features_list.append(feature)
            data_map = read_file(filename, features_list)
            accuracy = leave_one_out_validator(data_map)
            problem.set_accuracy_map[subset] = accuracy

        # Get the best subset and the chosen feature
        best_subset = max(accuracy_map, key=accuracy_map.get)
        for chosen_feature in problem.features_remaining:
            if chosen_feature in best_subset:
                break

        # Check if accuracy decreased
        new_accuracy = problem.set_accuracy_map[best_subset]
        decreased = new_accuracy < current_accuracy
        current_accuracy = new_accuracy

        # Select the feature
        problem.select_feature(best_subset, chosen_feature, "Forward")
        if decreased:
            print("(Warning, Accuracy has decreased!)\n\n")

    # return the subset with the best accuracy
    return max(problem.chosen_sets, key=lambda chosen_set: chosen_set[1])
