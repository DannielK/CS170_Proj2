import time
from nn_classifier import nn_classifier

def leave_one_out_validator(data_map):
    # Initialize the count of correct predictions to 0
    correct_predictions = 0.0

    for instance_index in data_map:
        train_start_time = time.time()
        # Create a copy of the data map to use as the training set
        train_set = data_map.copy()
        # Remove the current instance from the training set
        train_set.pop(instance_index)
        # Use the current instance as the test instance
        test_instance = data_map[instance_index]
        # Classify the test instance using the nearest neighbor classifier
        prediction = nn_classifier(test_instance, train_set)
        # Check if the predicted label matches the actual label of the test instance
        if prediction[0] == test_instance[0]:
            # If the prediction is correct, increment the count of correct predictions
            correct_predictions += 1.0
        train_end_time = time.time()
        train_time_taken = train_end_time - train_start_time
    # Return the accuracy of the classifier
    return correct_predictions / float(len(data_map))
