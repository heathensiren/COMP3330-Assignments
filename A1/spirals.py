import os 						# os 			-- OS operations, read, write etc
import tensorflow as tf 		# tensorflow 	-- machine learning
import numpy as np 				# numpy			-- python array operations
import matplotlib.pyplot as plt # matplotlib 	-- plotting
import csv						# csv 			-- reading from CSVs easily
import yaml						# yaml 			-- reading/writing config files
import time						# time 			-- performance measure
import random
from sklearn.svm import SVC		# sklearn		-- SVM utility

# TODO: Find less hacky way to include parent directories
import sys
sys.path.insert(0, '../../')
import util 					# util 			-- our bag of helper functions!

# Will load data in from the spiral dataset csv as store as x = [(x, y) ...] and y = [(c) ...]
def load_data(data_file):
	with open(data_file, 'r') as csvfile:
		csvreader = csv.reader(csvfile, delimiter=',')
		x = []
		y = []

		for row in csvreader:
			x.append(list(map(float, row[:-1])))
			y.append([int(row[-1])])
		return x, y
	print('[ERR] Failed to load data from file \'{0}\''.format(data_file))
	exit()

# Plot data for a 2D coordinate vector (x) and class [0, 1] (y)
def plot_data(x, y, train_type, c, title):
	# Gather points within class a and b for two spiral problem
	# TODO: Find more efficient way to do this
	a = []; b = []
	for i in range(0, len(x)):
		if y[i] == [1]:
			a.append(x[i])
		else:
			b.append(x[i])

	# Plot both classes
	plt.scatter([d[0] for d in a], [d[1] for d in a], color=c[0])
	plt.scatter([d[0] for d in b], [d[1] for d in b], color=c[1])

	# Format plot
	plt.title('{0} ({1})'.format(title, train_type))
	plt.xlabel('x')
	plt.ylabel('y')

### ANN
def construct_network(inp, weights, biases, neurons):
	fc1 = tf.nn.sigmoid(tf.add((tf.matmul(inp, weights['fc1'])), biases['fc1']), name='fc1')
	fc2 = tf.nn.sigmoid(tf.add((tf.matmul(fc1, weights['fc2'])), biases['fc2']), name='fc2')
	fc3 = tf.nn.sigmoid(tf.add((tf.matmul(fc2, weights['fc3'])), biases['fc3']), name='fc3')
	fc4 = tf.nn.sigmoid(tf.add((tf.matmul(fc3, weights['fc4'])), biases['fc4']), name='fc4')
	fc5 = tf.nn.sigmoid(tf.add((tf.matmul(fc4, weights['fc5'])), biases['fc5']), name='fc5')

	return fc5

def get_fitness(error, time_to_train):
	return error

def train_network(x, y, cfg):
	## Create network ##
	# Alias config vars
	neuron_lims = cfg['training']['nn']['neurons']
	epoch_lims  = cfg['training']['nn']['epochs']
	lr_lims     = cfg['training']['nn']['learning_rate']
	iterations  = cfg['training']['nn']['iterations']
	acc_thresh  = cfg['training']['nn']['accuracy_threshold']

	# TODO: Remove when train_network and test_network can be separated
	c1      = cfg['plotting']['c1']
	c2      = cfg['plotting']['c2']
	plot_en = cfg['plotting']['enabled']

	# Create placeholders for tensors
	x_ = tf.placeholder(tf.float32, [None, 2], name='x_placeholder')  # Input of (x, y)
	y_ = tf.placeholder(tf.float32, [None, 1], name='y_placeholder')  # Output of [0, 1]

	opt_model = {'accuracy' : 0., 'fitness' : None}

	# Iterate through models and choose the best one -- evolution!
	# TODO: Use gradient descent with epochs, learning rate and neurons per layer to find better near-model
	while(opt_model['accuracy'] <= acc_thresh):
		# Generate new random learning parameters
		learning_rate = random.uniform(lr_lims[0], lr_lims[1])
		neurons = random.randint(neuron_lims[0], neuron_lims[1])
		epochs = random.randint(epoch_lims[0], epoch_lims[1])

		# Generate new random weights for new network
		weights = {
			'fc1' : tf.Variable(tf.random_normal([2, neurons]),      name='w_fc1'),
			'fc2' : tf.Variable(tf.random_normal([neurons, neurons]), name='w_fc2'),
			'fc3' : tf.Variable(tf.random_normal([neurons, neurons]), name='w_fc3'),
			'fc4' : tf.Variable(tf.random_normal([neurons, neurons]), name='w_fc4'),
			'fc5' : tf.Variable(tf.random_normal([neurons, 1]), name='w_fc5')
		}

		# Generate new random biases for new network
		biases = {
			'fc1' : tf.Variable(tf.random_normal([neurons]), name='b_fc1'),
			'fc2' : tf.Variable(tf.random_normal([neurons]), name='b_fc2'),
			'fc3' : tf.Variable(tf.random_normal([neurons]), name='b_fc3'),
			'fc4' : tf.Variable(tf.random_normal([neurons]), name='b_fc4'),
			'fc5' : tf.Variable(tf.random_normal([1]), name='b_fc5')
		}

		final_layer = construct_network(x_, weights, biases, neurons)

		# Define error function
		cost = tf.reduce_mean(tf.losses.mean_squared_error(labels=y_, predictions=final_layer))

		# Define optimiser and minimise error function task
		optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

		## Train ##
		# Create error logging storage
		errors = []

		# Start timing network creation and training
		t_start = time.time()

		# Create new TF session
		sess = tf.InteractiveSession()
		tf.global_variables_initializer().run()

		for i in range(epochs):
			_, error = sess.run([optimiser, cost], feed_dict={x_: x, y_: y})
			errors.append(error)
			# Stop model early if we're under an acceptable threshold
			if error <= 1 - acc_thresh:
				epochs = i
				break

		# End time measurement
		t_elapsed = time.time() - t_start

		fitness = get_fitness(error, t_elapsed)

		# Calculate new accuracy #TODO: Calculate error properly
		accuracy = 1 - error

		# If we have a better model, store 
		# TODO store model properly using tf functions
		if(opt_model['fitness'] is None
			or fitness < opt_model['fitness']):
			opt_model = {
				'accuracy'      : accuracy,
				'duration'      : t_elapsed,
				'fitness'		: fitness,
				'epochs'        : epochs,
				'neurons'       : neurons,
				'learning_rate' : learning_rate,
				'errors'        : errors,
				'final_layer'   : final_layer # Should not store model this way -- will be undefined when sess.close is called
			}
			print('\t[ANN] New model:')
			print('\t[ANN] \tTraining parameters: epochs={0}, learning_rate={1:.2f}, neurons={2}, fitness={3:.5f}'.format(opt_model['epochs'], opt_model['learning_rate'], opt_model['neurons'], fitness))
			print('\t[ANN] \tModel accuracy: {0:.3f}%, Time to train: {1:.2f}s'.format(opt_model['accuracy']*100, opt_model['duration']))

	# Set size of figure and create first subplot
	plt.subplot(2, 2, 1)

	# Set plot settings
	plt.plot(opt_model['errors'][:opt_model['epochs']])
	plt.title('Error vs Epoch')
	plt.xlabel('Epoch')
	plt.ylabel('Error')
	plt.grid()

	## Test ##
	# Create second subplot
	plt.subplot(2, 2, 2)

	# Create test data
	lim = cfg['testing']['limits']
	test_range = np.arange(lim[0], lim[1], 0.1)
	x_test = [(x, y) for x in test_range for y in test_range]
	
	# Start timing the length of time training takes
	t_test = time.time()

	# Classify test data
	y_test = np.round(sess.run(opt_model['final_layer'], feed_dict={x_ : x_test}))

	# Average out the test timing
	t_avg_test = (time.time() - t_test) / float(len(y_test))

	print('\t[ANN] Average time to test: {0:.2f}us'.format(1000000 * t_avg_test))

	# Create class lists
	plot_data(x_test, y_test, 'ANN', c1, cfg['plotting']['title'])

	# print(opt_model)
	return opt_model

### SVM
def train_svm(x, y, cfg):
	# Read in SVM parameters
	C      = cfg['training']['svm']['C']
	kernel = cfg['training']['svm']['kernel']
	gamma  = cfg['training']['svm']['gamma']

	print('\t[SVM] Training parameters: C={0}, kernel={1}, gamma={2}'.format(C, kernel, gamma))

	# Create SVM with parameters
	svm = SVC(C=C, kernel=kernel, gamma=gamma)
	
	# Start timing SVM creation and training
	t_start = time.time()
	
	# Train SVM on data
	svm.fit(np.array(x), np.ravel(y))

	# End SVM timing
	t_train = time.time() - t_start

	print('\t[SVM] Model accuracy: {0:.2f}%, Time to train: {1:.5f}s'.format(100*svm.score(x, np.ravel(y)), t_train))

	return svm

def test_svm(svm, x):
	# Start timing testing of SVM
	t_start = time.time()

	y = [svm.predict([sample]) for sample in x]

	# Average 
	t_avg_test = (time.time() - t_start) / float(len(y))

	print('\t[SVM] Average time to test: {0:.2f}us'.format(1000000 * t_avg_test))

	return y

def main():
	sub_modules = ['two_spiral', 'two_spiral_dense']

	# Read config file
	cfg = util.read_config('config/spiral.yaml')

	for m in sub_modules:
		# Alias module configuration so we don't have to read from dict each time!
		cfg_m = cfg[m]

		# Load from config
		lim     = cfg_m['testing']['limits']
		c1      = cfg_m['plotting']['c1']
		c2      = cfg_m['plotting']['c2']
		plot_en = cfg_m['plotting']['enabled']
		title   = cfg_m['plotting']['title']

		print('Training on Module: {0}'.format(title))

		# Load two spiral data from dataset
		x_train, y_train = load_data(cfg_m['dataset'])

		# Create test set
		act_range = np.arange(lim[0], lim[1], 0.1)

		# Perform testing on SVM
		x_test = [(x, y) for x in act_range for y in act_range]

		# Initilise figure size
		if plot_en: 
			fig = plt.figure(figsize=(8, 8))

		## Neural Network
		if cfg_m['training']['nn']['enabled']:
			# Train network on dataset
			train_network(x_train, y_train, cfg_m)

			# Create plot of training data and show all plotting
			if plot_en:
				plot_data(x_train, y_train, 'ANN', c2, title)

			# New line
			print()

		## SVM
		if cfg_m['training']['svm']['enabled']:
			# Train SVM on training dataset
			svm = train_svm(x_train, y_train, cfg_m)

			# Test SVM on test dataset
			y_test = test_svm(svm, x_test)

			# Create plot of training data over testing data
			if plot_en:
				plt.subplot(2, 2, 4)
				plot_data(x_test, y_test, 'SVM', c1, title)
			plot_data(x_train, y_train, 'SVM', c2, title)
			
			# New line 
			print()

	plt.show()

if __name__ == '__main__':
	main()