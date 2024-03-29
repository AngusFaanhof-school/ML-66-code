class LogisticRegression:
	def __init__(self, learning_rate, num_iter,fit_intercept=True,verbose=False,regularization=0.1):
		self.learning_rate = learning_rate
		self.num_iter = num_iter
		self.fit_intercept = fit_intercept
		self.verbose = verbose
		self.regularization = regularization

	def add_intercept_to_features(self, features):
		if self.fit_intercept:
			intercept_column = np.ones((features.shape[0], 1))
			return np.concatenate((intercept_column, features), axis=1)
		return features


	def sigmoid(self, z_values):
		z_in = -z_values + 1e-7
		z = np.array(z_in, dtype='O')

		exp = np.array([np.exp(i) for i in z])

		return 1 / (1 + exp)

	def cost(self, predicted_probabilities, target_values):
		num_instances = target_values.shape[0]
		cost = (-target_values * np.log(predicted_probabilities + 1e-7) - (1 - target_values) * np.log(1 - predicted_probabilities + 1e-7)).mean()
		cost += (self.regularization / (2 * num_instances)) * np.sum(self.theta[1:] ** 2)
		return cost

	def fit(self, features, target_values):
		features = self.add_intercept_to_features(features)
		self.theta = np.zeros(features.shape[1])

		for i in range(self.num_iter):
			z = np.dot(features, self.theta)

			predicted_probabilities = self.sigmoid(z)

			gradient = np.dot(features.T, (predicted_probabilities - target_values)) / target_values.size
			gradient[1:] += (self.regularization / target_values.size) * self.theta[1:]

			theta = np.subtract(self.theta, self.learning_rate * np.array(gradient))
			self.theta = theta

			if(self.verbose == True and i % 10000 == 0):
				z = np.dot(features, self.theta)
				predicted_probabilities = self.sigmoid(z)
				print(f'loss: {self.cost(predicted_probabilities, target_values)} \t')

	def probability(self, features):
		features = self.add_intercept_to_features(features)
		return self.sigmoid(np.dot(features, self.theta))

	def predict(self, features, threshold):
		return self.probability(features) >= threshold