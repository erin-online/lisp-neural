; Basic neural network intended for simple image recognition.
; Doesn't work with stuff like chess because I haven't figured that out yet.
; Chess is hard and neural networks are hard.
;
; NOTE: This program requires the array-operations or aops package. Install it with (ql:quickload :array-operations), then do (in-package :array-operations) to get in.
;
; TO-DO:
; 1. Node as an object, represent each layer as a list of nodes. Also, allow for the creation of multiple networks.
; 2. Create an alternative to activate-network that returns the list of activations for every layer, not just the last ones.
; 3. Finish backpropagation using the function from 2
; 4. Work on file I/O for image recognition using data from the MNIST database http://yann.lecun.com/exdb/mnist/. This will result in a basic functioning neural network.
; 5. Explore xenodes and other ideas


(defun relu (input)
  "ReLU (rectified linear unit). Sets all nodes that would be negative to 0.
This is done because negative nodes are bad, apparently. Don't really know why,
I think they seem cool. Probably don't need a separate function for this
but it makes it easier to explain."
  (max 0 input))

(defun activate-node (bias weights prev-nodes norm-function) 
  "Assigns a value to a node ('neuron'). Returns a float that >= 0.
Formula looks like norm-function(bias + weight1*prevnode1 + weight2*prevnode2 + ...)
@bias The bias of the node. Starts at this value before any of the weights are added.
@weights Vector with each of the node's weights.
@prevnodes Vector with values of each of the nodes of the previous layer.
@norm-function Normalizing function applied after adding up all the weights.
ReLU is used for the normalizing function here."
  (funcall norm-function 
	   (+ bias (reduce #'+ (map 'vector #'* weights prev-nodes)))))

(defun activate-network (inputs)
  "Activates every node in the network, running it to get an output
based on a set of inputs.
@inputs 1D Vector representing the input layer."
  (let ((prev-nodes inputs) (current-nodes NIL) (layer-size 0))
    (dotimes (layer (length *sizes-no-input*))
      (setf layer-size (elt *sizes-no-input* layer))
      (setf current-nodes (make-array layer-size :fill-pointer 0)) ;makes empty array to store the nodes for the current layer
      (dotimes (node layer-size)
        (vector-push (activate-node (elt (elt (elt *network* layer) 0) node)
                                    (get-weights layer node)
                                    prev-nodes *norm-function*)
                     current-nodes)) ;adds current node to array
      (setf prev-nodes current-nodes))
    (return-from activate-network prev-nodes))) ;returns activation list for final (output) layer
  
(defun initialize-network-mono (sizes initializer)
  "Creates the network of nodes, sets every weight and bias according to the initializer.
Each layer is formatted as (#(biases) (#(weights1) #(weights2) ...)).
In other words, it's a list containing a 1D vector and a 2D vector.
The entire network is a list containing every layer.
@sizes List of integers describing number of nodes in each layer.
@initializer Function that initializes value for each weight and bias. Currently using get-one-random."
  (defparameter *sizes* sizes)
  (defparameter *sizes-no-input* (remove (elt *sizes* 0) *sizes* :count 1))
  (defparameter *network* ()) ;start with empty list
  (defparameter *norm-function* #'relu)
  (dotimes (layer (length *sizes-no-input*)) ;for each layer
    (setf *network* (append *network* (list (list (aops:generate initializer (list (elt *sizes-no-input* layer))) ;add initialized biases vector to list
			  (aops:generate initializer (list (elt *sizes-no-input* layer) (elt *sizes* layer)))))))) ;add initialized weights vector to list
  *network*) ;returns network

(defun get-one-random ()
  (- (random 1.5) 0.5))

(defun get-weights (layer node)
  "Returns the 1D vector of weights for a given node."
  (make-array (array-dimension (elt (elt *network* layer) 1) 1)
              :displaced-to (elt (elt *network* layer) 1)
              :displaced-index-offset (* node (array-dimension (elt (elt *network* layer) 1) 1))))

(defun backpropagate (errors)
  "Goes backwards through the network to find gradients for each weight and bias. Currently unfinished."
  (let ((layer-size 0) (prev-gradients errors) (current-gradients NIL))
    (dolist (layer-data (reverse *network*)) ;for each layer, starting at the end and going backwards. note that layer-data is the actual layer representation in *network*, not the cardinal or size
      (setf layer-size (length (elt layer-data 0)) current-gradients (make-array layer-size :fill-pointer 0)) ;delta(cost)/delta(current-nodes), for bias/weight gradient calculating in previous layer
      
      
