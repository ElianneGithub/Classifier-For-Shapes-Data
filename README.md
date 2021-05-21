# Classifier-For-Shapes-Data
Building of a classifier that automatically determine the shape (square, disk...) of the image vectors that are contained in the x_test file.

The shapes.py file contains image data of shapes taken by a camera: square, triangle, disk, or star. The images are in bitmap form (matrix of 10 pixels by 10 pixels) flattened into a vector of size 100.
The input data is in the form of a Numpy array. A line corresponds to an image, there are as many lines as images. The output data is in the form of a Numpy vector of the size of the number of images.
An output is an integer 0, 1, 2, or 3 representing one of the possible shapes: square, triangle, disk, or star.
The training data are in three arrays (sample size: 500 images):

x_learn represents the input vectors (500 lines, one line per vector)
y_learn represents the expected outputs, as an integer between 0 and 3 (4 possible values) (size 500)
z_learn represents the matrices corresponding to the inputs, of size 10x10 (500 matrix 10x10)

We can visualise the images using the matplotlib.pyplot.imshow() function using the matrices contained in z_learn.
The validation data are in two arrays (sample size: 100 images):

x_valid represents the input vectors (100 lines, one line per vector)
y_valid represents the expected outputs, as an integer between 0 and 3 (4 possible values) (size 100)


The validation data are used to validate the choice of the model and/or the parameters.
The test data are in an array (sample size: 100 images):

x_test represents the input vectors (100 lines, one line per vector)

The project focuses on the construction of a classifier that automatically determine the shape (square, disk...) of the image vectors contained in x_test. The shape is encoded as an integer 0, 1, 2 or 3.
Our task is to build a classifier using machine learning techniques, from the labelled data x_learn and y_learn, and to apply this classifier to the test data, to assign the correct shape to the unlabelled data of x_test.

Our goal is to provide a vector that contains the shape (square, disk...) predicted by our classifier on the test sample. The shape is encoded by integer.


The file graddsec.py contains the gradient descent, and the file classif.py implements a gradient descent for a linear model.

The data are contained in the variable S

The parameters of the model are in the variable T_vec

The function nb_error(T_vec, S) returns the number of errors made on the sample S with the model parameters T_vec

The function prediction(T_vec, S) returns the prediction vector for sample S with the parameter model T_vec

The function loss(T_vec, S) returns the loss for sample S with parameter model T_vec, calculated as the sum of the log probabilities of the observations (log-likelihood) 

The function grad_desc_n(loss, S, (dim input+1)*nb class, 100, step = 0.0001) takes as argument:
- The loss function
- the training sample
- the dimension of the parameter vector
- the number of steps of the gradient descent
- an optional step parameter giving the step size (default: 0.01)
- an optional parameter x 0 giving an initial vector for the parameters (default: random initialization)
