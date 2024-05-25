def nn_classifier(test_instance, train_set):
    # Initialize the minimum distance to infinity
    min_distance = float('inf')
    # Initialize the prediction to None
    prediction = None
    # Initialize the closest instance to None
    closest_instance = None
    # Iterate over each instance in the training set
    for i, train_instance in train_set.items():
        # Calculate the Euclidean distance between the test instance and the current training instance
        distance = euclidean_distance(test_instance, train_instance)
        # Check if the distance is less than the minimum distance
        if distance < min_distance:
            # If the distance is less than the minimum distance, update the minimum distance
            min_distance = distance
            # Update the prediction to be the class label of the current training instance
            prediction = train_instance[0]
            closest_instance = i
    # Return the predicted class label and the minimum distance to the closest instance
    return prediction, min_distance, closest_instance

def euclidean_distance(test_instance, train_instance):
    distance = 0.0
    # Iterate over each feature in the test and training instances
    for i in range(1, len(test_instance)):
        # Calculate the squared difference between the feature values
        distance += (test_instance[i] - train_instance[i]) ** 2
    # Return the square root of the sum of squared differences
    return distance ** 0.5