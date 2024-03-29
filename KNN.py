class KNN:
	def __init__(self, k):
		self.k = k

	def fit(self, x_points, y_points):
		self.x_points = x_points.values
		self.y_points = y_points.values

	def euclidean_distance(self, new_point):
		return np.sqrt(np.sum((self.x_points - new_point) ** 2, axis=1))

	def predict_single(self, new_instance):
		distances = np.linalg.norm(self.x_points - new_instance, axis=1)

		nearest_neighbor_ids = distances.argsort()[:self.k]
		nearest_neighbors = self.y_points[nearest_neighbor_ids]

		mode = scipy.stats.mode(nearest_neighbors.flatten())

		return mode.mode

	def predict(self, new_instances):
		return [self.predict_single(x) for x in new_instances]
