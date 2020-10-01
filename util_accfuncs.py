def predict_accuracy(instance, X_train, X_val, y_train):
    '''
    Inputs:
    - A classifier instance
    - The following values:
        X_train, X_val, y_train, y_val
    Process:
    - Fits the instance on the training data supplied
    - Determines the prediction accuracy of the instance on the training and validation data
    Outputs:
    - Training and validation accuracies of the classifier instance
    '''
    instance.fit(X_train, y_train)
    
    pred_train = instance.predict(X_train)
    pred_val = instance.predict(X_val)
    
    return (pred_train, pred_val)



def calc_accuracy(instance, **data):
    '''
    Inputs:
    - A classifier instance
    - A dictionary that comprises the following values:
        X_train, X_val, y_train, y_val
    Process:
    - Calculates training and validation accuracies using the predict_accuracy() function
    - Determines accuracy scores based on the accuracies
    Outputs:
    - Accuracy scores of the classifier on the training and validation data
    '''
    
    from sklearn.metrics import accuracy_score
    
    X_train = data['X_train']
    X_val = data['X_val']
    y_train = data['y_train']
    y_val = data['y_val']
    
    data_pred = [X_train, X_val, y_train]
    
    pred_train, pred_val = predict_accuracy(instance, *data_pred)
    
    accuracy_train = accuracy_score(y_true=y_train, y_pred=pred_train)
    accuracy_val = accuracy_score(y_true=y_val, y_pred=pred_val)
    
    return (accuracy_train, accuracy_val)



def print_scores(desc, instance, **data):    
    '''
    Inputs:
    - A classifier description, such as Bagging or Random forest
    - A classifier instance
    - A dictionary that comprises the following values:
        X_train, X_val, y_train, y_val
    Process:
    - Calculates accuracy scores using the calc_accuracy() function
    Outputs:
    - Prints the calculated accuracy scores to the standard output
    '''
    
    accuracy_train, accuracy_val = calc_accuracy(instance, **data)
    
    print(f'{desc}')
    print(f'> Accuracy on training data = {accuracy_train:.4f}')
    print(f'> Accuracy on validation data = {accuracy_val:.4f}')