; Basic neural network intended for simple image recognition.
; Doesn't work with stuff like chess because I haven't figured that out yet.
; Chess is hard and neural networks are hard.
;
; NOTE: This program requires the array-operations or aops package. Install it with (ql:quickload :array-operations), then do (in-package :array-operations) to get in.
;
; TO-DO:
;✓1. Node as an object, represent each layer as a list of nodes. Also, allow for the creation of multiple networks.
; 2. Create an alternative to activate-network that returns the list of activations for every layer, not just the last ones.
; 3. Finish backpropagation using the function from 2
; 4. Work on file I/O for image recognition using data from the MNIST database http://yann.lecun.com/exdb/mnist/. This will result in a basic functioning neural network.
; 5. Explore xenodes and other ideas

; PART 1: MATH FUNCTIONS

(defun get-one-random ()
  "Produces a random number between -0.5 and 1. Pretty arbitrary."
  (- (random 1.5) 0.5))

(defun relu (number)
  "ReLU (rectified linear unit). Sets all nodes that would be negative to 0.
This is done because negative nodes are bad, apparently. Don't really know why,
I think they seem cool. Probably don't need a separate function for this
but it makes it easier to explain."
  (max 0 number))

(defun nsin (number)
  "Negative sin. Defined as the derivative of (cos x)."
  (* -1 (sin number)))

(defun inv (number)
  "1/x. Defined as the derivative of (log x)."
  (/ 1 number))

; PART 2: NODE OBJECT

(defclass node (standard-object)
                                        ; This is the node object. Every network is a list of layers, each layer is a list of nodes.
                                        ; The node stores four main things: prev-node-count, weights, outer-params, and function.
                                        ; Right now the function is split up into several parts, but this is really unnecessary.
                                        ; Also, it prevents true xenodes such as output = weight1*prevnode1 + prevnode2^weight2,
                                        ; where each prevnode can have a different function attached to it.
                                        ; This isn't on my top priority of things to implement, but consolidating the function might
                                        ; also make derivative calculating easier, so I might just do it anyway.

                                        ; Behavior:
                                        ; When the network is activated, each node gets passed all the output values from the previous layer of nodes ("prevnodes").
                                        ; The node then plugs all these values into its function to determine its own output value.
                                        ; TODO: When backpropagation is implemented, the node finds the derivative of every weight compared to the node activation
                                        ; (i.e. if you change the weight by a small amount, what will the comparative change in activation be?)
                                        ; and adjusts the weights depending on the error values ("cost") of network activations.
  
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
   (outer-function :initarg :outer-function :initform (lambda (node-value bias) (+ node-value bias)) :accessor outer-function
                   :documentation "Function applied to the node as a whole, not to individual weights. Example is adding bias.")
   (connecting-function :initarg :connecting-function :initform #'+ :accessor connecting-function
                        :documentation "Function that binds every inner-function together. Should be commutative f(b, a) = f(a, b), so * and + are good fits.")
   (inner-function :initarg :inner-function :initform (lambda (prev-node weights) (* prev-node (elt weights 0))) :accessor inner-function
                   :documentation "Function called on every prev-node value. Uses weights.")))

(defgeneric activate-node (node prev-nodes)
  )

(defmethod activate-node ((node node) prev-nodes)
  "Takes in a node and a list of outputs in the previous layer. Returns the output (number) for that node."
                                        ; call inner-function on each prev-node
  (let ((inner-function-results (make-array (prev-node-count node) :fill-pointer 0)) (bias (elt (outer-params node) 0)))
    (dotimes (prev-node-counter (prev-node-count node))
      (let ((weights (slice-2d-array (weights node) prev-node-counter)) (prev-node (elt prev-nodes prev-node-counter)))
        (vector-push (funcall (inner-function node) prev-node weights) inner-function-results))) ; push result to the inner-function-results vector
    (let* ((connecting-function-result (reduce (connecting-function node) inner-function-results)) ; call connecting-function to reduce the inner function results
           (outer-function-result (funcall (outer-function node) connecting-function-result bias)) ; call outer function on the connecting function result
           (norm-function-result (funcall (norm-function node) outer-function-result))) ; call norm function on the outer function result
      norm-function-result))) ; return norm function result

(defgeneric access-weight (node prev-node-index weight-index)
  )

(defmethod access-weight ((node node) prev-node-index weight-index)
  ; Helper function so I don't have to constantly use slice-2d-array and elt to access a specific weight.
  (elt (slice-2d-array (weights node) prev-node-index) weight-index))

(defun activate-network (network input-nodes)
  "Activates an entire network given a list of input nodes."
  (let ((prev-nodes input-nodes) (current-nodes) (nodes-in-layer))
    (dotimes (layer-number (length network) current-nodes)
      (setf nodes-in-layer (length (elt network layer-number))) ; find how many nodes are in the layer so we can make our vector
      (setf current-nodes (make-array nodes-in-layer :fill-pointer 0)) ; make an empty vector for the current layer
      (dotimes (node-number nodes-in-layer) ; for each node in the layer
        (vector-push (activate-node (elt (elt network layer-number) node-number) prev-nodes) current-nodes)) ; push the activate-node result to the vector
      (setf prev-nodes current-nodes)))) ; moving to the next layer. This also returns current-nodes, meaning we don't need a dedicated return statement.

(defun generate-weights (initializer prev-node-count weights-per-prev-node)
  "Generates a 2D weights array for a node using the aops library. Sets each weight to the result of a call to the (typically random) initializer function."
  (aops:generate initializer (list prev-node-count weights-per-prev-node)))

(defun initialize-network-mono (sizes initializer)
  "Creates the network of nodes, sets every weight and bias according to the initializer.
Each layer is formatted as (#(biases) (#(weights1) #(weights2) ...)).
In other words, it's a list containing a 1D vector and a 2D vector.
The entire network is a list containing every layer.
@sizes List of integers describing number of nodes in each layer.
@initializer Function that initializes value for each weight and bias. Currently using get-one-random."
  (defparameter *norm-function* #'relu)
  (let ((sizes-no-input (remove-first sizes)) (network NIL))
                                        ; NOTE: The input layer does NOT have dedicated node objects, because its values are given by an outside source.
                                        ; For example, an input layer might be how dark a pixel in an image is, for an image recognition network.
                                        ; The input layer itself goes through no function. This is why we use remove-first.
    
    (dotimes (layer (length sizes-no-input))
      (let ((layer-size (elt sizes-no-input layer)))
        (let ((layer-data (make-array layer-size :fill-pointer 0))) ; make an empty vector for the layer. Could have used let* to have this line and the previous in 1 but oh well.
          (dotimes (current-node layer-size)
            (vector-push (make-instance 'node ; Makes a new node to put in the vector.
                                        :prev-node-count (elt sizes layer) ; (elt sizes layer) is equal to (elt sizes-no-input layer) - 1, meaning it's a handy way of accessing the size of the previous layer.
                                        ; This is necessary to give the nodes input about how many nodes are in the previous layer; for example, how many weights they need.
                                        :weights (generate-weights initializer (elt sizes layer) 1)
                                        :outer-params (aops:generate initializer 1))
                         layer-data))
          (setf network (append network (list layer-data))))))
    (return-from initialize-network-mono network)))

; PART 3: DERIVATIVES (or: Functional Programming Hell) (Also Featuring Helper Functions)

(defun generate-derivative-table ()
  "The derivative table is the base for the get-derivative function. It's a lookup table for the most elementary derivatives."
  (defparameter *derivative-table* (make-hash-table)) ; Global hash table. Access this using (gethash [function] *derivative-table*)
  (add-derivatives
   #'sin #'cos
   #'cos #'nsin
   #'exp #'exp
   #'log #'inv
   ))

(defun add-derivatives (&rest args)
                                        ; Helper function for generate-derivative-table.
                                        ; Works like setf, taking an even number of arguments. Instead of setting A to B, it sets hash A to value B.
  (dotimes (binding (/ (length args) 2))
    (let ((func (elt args (* binding 2))) (deriv (elt args (+ (* binding 2) 1))))
      (add-one-deriv func deriv))))

(defmacro add-one-deriv (x y)
                                        ; Helper macro for add-derivatives. Adds one function and its derivative to the table.
  `(progn (setf (gethash ,x *derivative-table*) ,y)
          (setf (gethash (get-function-name ,x) *derivative-table*) T)))

(defun get-derivative (func x)
  "Takes in a lambda expression, such as (sin x), and returns a lambda expression corresponding to the function's derivative, such as (cos x).
  @func The lambda expression.
  @x The variable we're finding the derivative with respect to. (x sin(z))' with respect to x is sin(z), but with respect to z it's x cos(z).
  Since each node has many weights and outer-params attached to it, and we need to know the derivative with respect to each one, this is very important."
                                        ; A note: This function uses "atom" terminology.
                                        ; I don't know if this is like an official thing or not, I came up with it randomly.
                                        ; Basically an atom is one of the following things: an elementary function, a variable, or a nested function.
                                        ; x is an atom, sin is an atom, (sin (cos x)) is an atom.
                                        ; If this is unintuitive and confusing, sorry. I couldn't think of a better word.
                                        ; The idea of recursive calls to this function is that we can eventually break down the larger atoms by splitting them into first-atom and rest-atoms,
                                        ; splitting something like (sin (cos x)) into sin and (cos x), then going down to (cos x).
  
  (if (equal (length func) 1) ; One atom is in the provided lambda expression. This is a variable. If it is not, someone else screwed up.
      (if (equal (elt func 0) x) ; Are we finding the derivative with respect to this variable?
          (return-from get-derivative 1) ; Return 1 if yes
          (return-from get-derivative 0))) ; Return 0 if no (we are keeping this variable constant)
  (let ((first-atom (elt func 0)) ; The first item in the func list. This is a function, probably.
                                        ; (second-atom (if (eql (type-of (elt func 1)) 'cons) (elt (elt func 1) 0) (elt func 1)))
                                        ; I don't know if I need second-atom so I'm keeping it around in comment form
        
        (rest-atoms ; The rest of the atoms. Everything except first-atom.
          (if (and (eql (type-of (elt (remove-first func) 0)) 'cons) ; Condition 1: The second item in the func list is a cons cell (list).
                   (< (length func) 3)) ; Condition 2: There are only two items in the func list.
              (elt (remove-first func) 0) ; These conditions activate when the second atom is a nested function. Removing the first atom gives a nested list that we don't want.
                                        ; For example, (remove-first (sin (cos x)) will give us ((cos x)). This is silly and dumb. (elt (remove-first func 0)) gets rid of the outer layer.
              (remove-first func)))) ; If one of the above conditions are false, we set rest-atoms to func with the first item removed.
                                        ; For more explanation on how this works, see https://media.discordapp.net/attachments/384539353109495818/1017943151299801208/unknown.png?width=980&height=669
    
    (if (gethash first-atom *derivative-table*)
                                        ; outermost atom is a function in the derivative table
        (let ((first-deriv (gethash (eval `(function ,first-atom)) *derivative-table*))) ; looks up the derivative
          (return-from get-derivative ; Uses the chain rule.
            `(* (,(get-function-name first-deriv) ,rest-atoms) ; f'(g(x))
              ,(get-derivative rest-atoms x))))) ; g'(x)
    (if (equal first-atom '+) ; sum
        (return-from get-derivative `(reduce #'+ ,(map 'list #'get-derivative rest-atoms (make-array (length rest-atoms) :initial-element x))))) ; This is still a little sketchy
    ; (if (equal first-atom '*) ; product rule
        ; )
    ))

(defun product-rule (&rest atoms)
  ; idk lol
  `(+ (dolist (atom atoms)))) 



(defun slice-2d-array (array index)
  "Returns the 1D array at the specified index of a 2D array."
  (make-array (array-dimension array 1)
              :displaced-to array
              :displaced-index-offset (* index (array-dimension array 1))))

(defun get-lambda-exp (func)
  "Gets the lambda body of a function. For example, if you put in (lambda (x) (* 2 x)), the function will output (* 2 x).
  Watch out for variable name conflicts--variable names are returned as-is."
  (elt (function-lambda-expression func) 2))

(defun get-function-name (func)
  "Returns the name of a function. Used to convert a function into a symbol."
  (nth-value 2 (function-lambda-expression func)))

(defmacro evaluate-name (name)
  "Inverse of get-function-name. You put in a function name and it returns the function."
  `(function ,(eval name))) ; the more eval forms you have the better your code is

(defmacro evaluate-lambda-exp (lambda-list lambda-exp)
  "Takes in a variable holding a lambda list and another holding a lambda expression, returns a lambda function that uses them.
  Example: a = (x y), b = (+ x y), (evaluate-lambda-exp a b) -> (lambda (x y) (+ x y).
  I'm not sure whether this actually works so it needs more testing."
  `(lambda ,lambda-list ,(eval lambda-exp)))

(defun remove-first (sequence)
  "Remove the first item in a sequence. This is basically the best function in the program."
  (remove (elt sequence 0) sequence :count 1))

(defun backpropagate (network errors)
  "Goes backwards through the network to find gradients for each weight and bias. Currently unfinished."
  (let ((layer-size 0) (prev-gradients errors) (current-gradients NIL))
    (dolist (layer-data (reverse *network*)) ;for each layer, starting at the end and going backwards. note that layer-data is the actual layer representation in *network*, not the cardinal or size
      (setf layer-size (length (elt layer-data 0)) current-gradients (make-array layer-size :fill-pointer 0)) ;delta(cost)/delta(current-nodes), for bias/weight gradient calculating in previous layer
      )))
