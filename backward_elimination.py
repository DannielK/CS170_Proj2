from problem import Problem
from leave_one_out_validator import leave_one_out_validator
import random

def read_file(file_name, features_list):
    data_map = {}
    features_mean = [float('0')] * len(features_list)
    features_std = [float('0')] * len(features_list)
    num_instances = 0.0
    # mean_std_start_time = time.time()
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
        #print("Means:")
        #for i, feature in enumerate(features_list):
            #print("Feature", feature, ":", features_mean[i])
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
    #print("Standard Deviations:")
    #for i, feature in enumerate(features_list):
        #print("Feature", feature, ":", features_std[i])
    # mean_std_end_time = time.time()
    # mean_std_time_taken = mean_std_end_time - mean_std_start_time
    # print("\nMean and standard deviation values found in: " + str(mean_std_time_taken) + " seconds")
                
    # read and normalize the data
    with open(file_name, 'r') as file:
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
        #read_end_time = time.time()
        #read_time_taken = read_end_time - read_start_time
        #print("Data read and normalized in: " + str(read_time_taken) + " seconds")
    # {instance_index: [class_lable, feature1, feature2, ...]}
    return data_map

def backward_elimination(prob: Problem, filename):

    while len(prob.features_remaining) > 1:
        #create new subsets
        prob.new_subsets("backward")

        # assign each subset an accuracy score 
        for subset in prob.set_accuracy_map:
             # get the data map from the file
            features_list = []
            for feature in subset:
                features_list.append(feature)
            data_map = read_file(filename, features_list)

            # perform leave one out validation
            result = leave_one_out_validator(data_map)
            prob.set_accuracy_map[subset] = result

        #Find the subset with highest accuracy score
        highest_score = 0
        best_subset = ()
        for subset in prob.set_accuracy_map:
            if prob.set_accuracy_map[subset] > highest_score:
                #print("found higher score\n")
                highest_score = prob.set_accuracy_map[subset]
                best_subset = subset
        
        #print("Subset to remove is: " + str(best_subset) + "\n")

        # Find feature to remove
        chosen_feature = 0
        for feature in prob.features_remaining:
            if not (feature in best_subset):
                chosen_feature = feature

        #print("Chosen feature to remove is: " + str(chosen_feature) + "\n")
        #Select feature
        prob.select_feature(best_subset, chosen_feature, "backward")
    
    #loop through chosen_features and find the subset with highest accuracy score
    final_highest_score = 0
    final_best_subset = ()
    for chosen in prob.chosen_sets:
        if(final_highest_score > chosen[1]):
            print("(Warning! Accuracy has decreased!)")
        if chosen[1] > final_highest_score:
            final_highest_score = chosen[1]
            final_best_subset = chosen[0]
    
    

    return (final_best_subset, final_highest_score)