from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, fbeta_score

def train_and_evaluate_model(x_train, y_train, x_test, y_test, sampling_method):
    nn_model = MLPClassifier(hidden_layer_sizes=(1000,), activation='relu', solver='adam', max_iter=1000)

    nn_model.fit(x_train, y_train)

    y_pred_test = nn_model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred_test)

    recall = recall_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test)

    f1 = f1_score(y_test, y_pred_test)
    fbeta = fbeta_score(y_test, y_pred_test, beta=0.5)

    return {
        'Sampling Method': sampling_method,
        'Accuracy': accuracy,
        'Recall': recall,
        'Precision': precision,
        'F1 Score': f1,
        'Fbeta Score': fbeta
    }