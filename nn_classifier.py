def nn_classifier(test_instance, train_set):
    min_distance = float('inf')
    prediction = None
    for train_instance in train_set:
        distance = euclidean_distance(test_instance, train_instance)
        if distance < min_distance:
            min_distance = distance
            prediction = train_instance[0]
    return prediction

def euclidean_distance(test_instance, train_instance):
    distance = 0
    for i in range(1, len(test_instance)):
        distance += (test_instance[i] - train_instance[i]) ** 2
    return distance ** 0.5