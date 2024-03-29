from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error, precision_score, recall_score, accuracy_score, f1_score

def get_grid_search(x_train, y_train):
    param_grid = {
        'criterion': ['log_loss'],
        'max_depth': [6, 10, 13, 16, 20, 30],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2, 4]
    }

    tree_classifier = DecisionTreeClassifier()

    scorers = {
		'MSE' : make_scorer(mean_squared_error),
		'precision_score': make_scorer(precision_score),
		'recall_score': make_scorer(recall_score),
		'accuracy_score': make_scorer(accuracy_score),
		'F1_score' : make_scorer(f1_score)
	}

    grid_search = GridSearchCV(estimator=tree_classifier, param_grid=param_grid, scoring=scorers, cv=5, refit='MSE')

    grid_search.fit(x_train, y_train)

    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    print("Best Hyperparameters:")
    for param, value in best_params.items():
        print(f"{param}: {value}")

    print(best_model)

    return grid_search