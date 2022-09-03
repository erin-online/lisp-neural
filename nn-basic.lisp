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
(defclass node (standard-object)
  ((prev-node-count :initarg :prev-node-count :initform (error "prev-node-count not defined") :accessor prev-node-count
               :documentation "Number of nodes in the previous layer.")
   (weights :initarg :weights :initform (error "weights not provided") :accessor weights
            :documentation "2D array with weights. Each sub-array is the weights for one prev-node.")
                                        ; If the above is confusing: For an inner-function that takes 2 weights; for example,
                                        ; a(L) = b + w1alpha * sin(w1beta * a(L-1, 1)) + ..., the weights array will look like
                                        ; #2A((w1alpha w1beta) (w2alpha w2beta) ...)
   (outer-params :initarg :outer-params :initform (error "outer params not provided") :accessor outer-params
                 :documentation "List of params applied to the outer function i.e. not on each weight. Bias and coefficient go here.")
                                        ; There are a bunch of functions here. This is how the node activation is calculated:
                                        ; First, inner-function is called on each prev-node activation value; for example, (* weight prev-node).
                                        ; After that, connecting-function (inner-function-results*) is called to bind together every inner-function result. This might involve adding them up with #'+.
                                        ; Next, outer-function is called on the result of connecting-function. outer-function is typically pretty simple and looks something like (+ bias connecting-function-result).
                                        ; Finally, norm-function is called on the result of outer-function. This produces the number for the node.
   (norm-function :initarg :norm-function :initform #'relu :accessor norm-function
                  :documentation "Normalizing function such as ReLU, applied to the final value of outer-function.")
   (outer-function :initarg :outer-function :initform `(+ node bias) :accessor outer-function
                   :documentation "Function applied to the node as a whole, not to individual weights. Example is adding bias.")
   (connecting-function :initarg :connecting-function :initform #'+ :accessor connecting-function
                        :documentation "Function that binds every inner-function together. Should be commutative f(b, a) = f(a, b), so * and + are good fits.")
   (inner-function :initarg :inner-function :initform `(* (elt weights 0) prev-node) :accessor inner-function
                   :documentation "Function called on every prev-node value. Uses weights.")))

(defgeneric activate-node (node prev-nodes)
  )

(defmethod activate-node ((node node) prev-nodes)
                                        ; call inner-function on each prev-node
  (let ((inner-function-results (make-array prev-node-count :fill-pointer 0)))
    (dotimes (prev-node-counter (prev-node-count node))
      (let ((weights (slice-2d-array (weights node) prev-node-counter)) (prev-node (elt prev-nodes prev-node-counter))
            (vector-push (eval (inner-function node)) inner-function-results))))
    (let ((connecting-function-result (reduce (connecting-function node) inner-function-results)))
      (let ((outer-function-result (funcall outer-function connecting-function-result)))
        (let ((norm-function-result (funcall norm-function outer-function-result)))
          norm-function)))))

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

(defun initialize-network-mono (sizes initializer)
  "Creates the network of nodes, sets every weight and bias according to the initializer.
Each layer is formatted as (#(biases) (#(weights1) #(weights2) ...)).
In other words, it's a list containing a 1D vector and a 2D vector.
The entire network is a list containing every layer.
@sizes List of integers describing number of nodes in each layer.
@initializer Function that initializes value for each weight and bias. Currently using get-one-random."
  (defparameter *norm-function* #'relu)
  (let ((sizes-no-input (remove (elt sizes 0) sizes :count 1)) (network NIL))
    (dotimes (layer (length sizes-no-input))
      (let ((layer-size (elt sizes-no-input layer)))
        (let ((layer-data (make-array layer-size :fill-pointer 0)))
          (dotimes (current-node layer-size)
            (vector-push (make-instance 'node
                                        :prev-node-count (elt sizes layer)
                                        :weights (generate-weights initializer (elt sizes layer) 1)
                                        :outer-params (aops:generate initializer 1))
                         layer-data))
          (setf network (append network (list layer-data))))))
    (return-from initialize-network-mono network)))

(defun get-one-random ()
  (- (random 1.5) 0.5))

(defun generate-weights (initializer prev-node-count weights-per-prev-node)
  (aops:generate initializer (list prev-node-count weights-per-prev-node)))
      
(defun get-weights (layer node)
  "Returns the 1D vector of weights for a given node."
  (make-array (array-dimension (elt (elt *network* layer) 1) 1)
              :displaced-to (elt (elt *network* layer) 1)
              :displaced-index-offset (* node (array-dimension (elt (elt *network* layer) 1) 1))))

(defun slice-2d-array (array index)
  "Returns the 1D array at the specified index of a 2D array."
  (make-array (array-dimension array 1)
              :displaced-to array
              :displaced-index-offset (* index (array-dimension array 1))))

(defun backpropagate (errors)
  "Goes backwards through the network to find gradients for each weight and bias. Currently unfinished."
  (let ((layer-size 0) (prev-gradients errors) (current-gradients NIL))
    (dolist (layer-data (reverse *network*)) ;for each layer, starting at the end and going backwards. note that layer-data is the actual layer representation in *network*, not the cardinal or size
      (setf layer-size (length (elt layer-data 0)) current-gradients (make-array layer-size :fill-pointer 0)) ;delta(cost)/delta(current-nodes), for bias/weight gradient calculating in previous layer
      )))
