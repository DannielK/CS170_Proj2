# Problem class sets up the necessary data structures and functions for the algorithms
class Problem:
    # Initialization function
    def __init__(self, numFeatures: int):
        # This is a list of the features that haven't been picked yet
        self.features_remaining = list(range(1, numFeatures + 1))
        # This is a list of the (set,accuracy) that we picked to keep for tracing
        self.chosen_sets = []
        # This is the current tuple of the features we've picked
        self.chosen_features = ()
        # This is a map that ((new subsets that have the chosen features + 1 non-chosen): accuracy)
        self.set_accuracy_map = {}
    
    # Function to make the new subsets depending on the algorithm ((chosen features + or - 1 non-chosen): default accuracy)
    def new_subsets(self, algo: str):
        # Loop through the features that haven't been chosen yet
        for num in self.features_remaining:
            # Make a new tuple that has chosen features + or - 1 non-chosen depending on algorithm
            temp_tuple = self.chosen_features + (num, ) if algo == "Forward" else tuple(x for x in self.features_remaining if x!= num)
            # Add that new tuple subset to the map with default accuracy
            self.set_accuracy_map[temp_tuple] = -1.0

    # Function to select the new subset that has a new feature with the highest accuracy
    def select_feature(self, best_set: tuple, chosen_feature: int, algo: str):
        # Add to the list of chosen subsets the new feature subset
        self.chosen_sets.append((best_set, self.set_accuracy_map[best_set]))
        # Remove that selected feature from the remaining features list
        self.features_remaining.remove(chosen_feature)
        # Forward: add the new feature to the list of chosen features; Backward: remove the feature from the chosen features
        self.chosen_features = self.chosen_features + (chosen_feature, ) if algo == "Forward" else tuple(x for x in self.chosen_features if x!= chosen_feature)
        # Print out the trace step
        for subset in self.set_accuracy_map:
            print("\tUsing feature(s) " + "{}".format(subset) + " accuracy is " + str(100*self.set_accuracy_map[subset]) + "%\n")
        print("Feature set " + "{}".format(best_set) + " was best, accuracy is " + str(100*self.set_accuracy_map[best_set]) + "%\n\n")
        # Clear the map to get rid of the non-chosen new feature subsets
        self.set_accuracy_map.clear()
