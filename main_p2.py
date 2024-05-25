import time
from leave_one_out_validator import leave_one_out_validator

# intro
print("Welcome to group 27's NN-Classifier and Leave One Out Validator.\n")

# list of dataset files
file_list = {
    1: "small-test-dataset-1.txt",
    2: "large-test-dataset-1.txt",
}

# print the file choices and ask for the file number
print("\nType the number of the dataset file you want to run.\n"
        "\tsmall-test-dataset-1\n"
        "\tlarge-test-dataset-1\n")
file_name = int(input())

# ask the user for a list of features separated by commas
input_str = input("Enter a list of features separated by commas: ")

# record start time
overall_start_time = time.time()

# split the input string by commas and convert each element to an integer
features_list = [int(feature) for feature in input_str.split(',')]

# read the dataset file, and return a dictionary with the class labels as keys and the instance's features as values
def read_file(file_name, features_list):
    data_map = {}
    features_mean = [float('0')] * len(features_list)
    features_std = [float('0')] * len(features_list)
    num_instances = 0.0
    mean_std_start_time = time.time()
    # calculate the mean for the standard deviation of the features
    with open(file_name, 'r') as file:
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
        # print the means of all the features
        print("Means:")
        for i, feature in enumerate(features_list):
            print("Feature", feature, ":", features_mean[i])
    # calculate the standard deviation of the features
    for i, feature in enumerate(features_list):
        sum_squared_diff = [float('0')] * len(features_list)
        diff = 0.0
        with open(file_name, 'r') as file:
            for line in file:
                instance = line.strip().split()
                feature_value = float(instance[feature])
                diff = feature_value - features_mean[i]
                sum_squared_diff[i] += diff ** 2
        features_std[i] = (sum_squared_diff[i] / (num_instances - 1.0)) ** 0.5
    # print the standard deviations of all the features
    print("Standard Deviations:")
    for i, feature in enumerate(features_list):
        print("Feature", feature, ":", features_std[i])
    mean_std_end_time = time.time()
    mean_std_time_taken = mean_std_end_time - mean_std_start_time
    print("\nMean and standard deviation values found in: " + str(mean_std_time_taken) + " seconds")
                
    # read and normalize the data
    with open(file_name, 'r') as file:
        read_start_time = time.time()
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
        read_end_time = time.time()
        read_time_taken = read_end_time - read_start_time
        print("Data read and normalized in: " + str(read_time_taken) + " seconds")
    # {instance_index: [class_lable, feature1, feature2, ...]}
    return data_map

# get the data map from the file
data_map = read_file(file_list[file_name], features_list)

# perform leave one out validation
result = leave_one_out_validator(data_map)

# record end time
overall_end_time = time.time()

# print the result
print("Accuracy: " + str(result) + " using features: " + str(features_list))

# calculate and print the time taken
overall_time_taken = overall_end_time - overall_start_time
print("Overall time taken: " + str(overall_time_taken) + " seconds")


    # leave_one_out_validator(data_map):
        # correct_predictions = 0
        # for instance_index in data_map:
            # train_set = data_map.copy()
            # train_set.pop(instance_index)
            # test_instance = data_map[instance_index]
            # prediction = nn_classifier(test_instance, train_set)
            # if prediction == test_instance[0]:
                # correct_predictions += 1
        # return correct_predictions / len(data_map)

    # nn_classifier(test_instance, train_set):
        # min_distance = float('inf')
        # for train_instance in train_set:
            # distance = euclidean_distance(test_instance, train_instance)
            # if distance < min_distance:
                # min_distance = distance
                # prediction = train_instance[0]
        # return prediction
    
    # euclidean_distance(test_instance, train_instance):
        # distance = 0
        # for i in range(1, len(test_instance)):
            # distance += (test_instance[i] - train_instance[i]) ** 2
        # return distance ** 0.5