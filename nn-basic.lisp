; Basic neural network intended for simple image recognition.
; Doesn't work with stuff like chess because I haven't figured that out yet.
; Chess is hard and neural networks are hard.
;
; NOTE: This program requires the array-operations or aops package. Load it with (ql:quickload :array-operations). Eventually I'll add something into the code to load it automatically.
;
; TO-DO:
;✓1. Node as an object, represent each layer as a list of nodes. Also, allow for the creation of multiple networks.
;✓2. Create an alternative to activate-network that returns the list of activations for every layer, not just the last ones.
;✓3. Finish backpropagation using the function from 2
; 4. Work on file I/O for image recognition using data from the MNIST database http://yann.lecun.com/exdb/mnist/. This will result in a basic functioning neural network.
; 5. Allow for networks to be imported and exported via files.
; 6. Explore xenodes and other ideas

; PART 1: MATH FUNCTIONS

(ql:quickload :array-operations)

(defun get-one-random ()
  "Produces a random number between -0.5 and 1. Pretty arbitrary."
  (- (random 1.5) 0.5))

(defun get-one-rational ()
  "Produces a random number from the set {-1, -0.9, -0.8, ..., 0.8, 0.9, 1}. Not for production code use, more for making testing calculations easier."
  (* (- (random 21) 10) 0.1)) ; don't use / 10 here because then you get fractions which are really annoying to read

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

(defun nis (number)
  "Negative inverse square; -1/x^2. Defined as the derivative of (inv x)."
  (* -1 (expt number -2)))

(defun sgn (number)
  "Modified signum. 0 if x<=0, 1 if x>0. Defined as the derivative of (relu x)."
  (if (> number 0) 1 0))

(defun asinp (number)
  "Arcsine prime. 1/sqrt(1-x^2). Defined as the derivative of (asin x)."
  (inv (sqrt (- 1 (expt number 2)))))

(defun acosp (number)
  "Arccosine prime. -1/sqrt(1-x^2). Defined as the derivative of (acos x)."
  (* -1 (inv (sqrt (- 1 (expt number 2))))))

(defun atanp (number)
  "Arctangent prime. 1/(1+x^2). Defined as the derivative of (atan x)."
  (inv (+ 1 (expt number 2))))

; PART 2: HELPER FUNCTIONS

(defun slice-2d-array (array index)
  "Returns the 1D array at the specified index of a 2D array."
  (make-array (array-dimension array 1)
              :displaced-to array
              :displaced-index-offset (* index (array-dimension array 1))))

(defun get-lambda-body (func)
  "Gets the lambda body of a function. For example, if you put in (lambda (x) (* 2 x)), the function will output (* 2 x).
  Watch out for variable name conflicts--variable names are returned as-is.
  For multi-line functions, returns only the last form in the body."
  (elt (last (nth-value 0 (function-lambda-expression func))) 0))

(defun get-function-name (func)
  "Returns the name of a function. Used to convert a function into a symbol."
  (nth-value 2 (function-lambda-expression func)))

(defmacro evaluate-name (name)
  "Inverse of get-function-name. You put in a function name and it returns the function."
  `(function ,(eval name))) ; the more eval forms you have the better your code is

(defun evaluate-lambda-exp (lambda-list &rest lambda-body)
  "Takes in a variable holding a lambda list and another holding a lambda expression, returns a lambda function that uses them.
  Example: a = (x y), b = (+ x y), (evaluate-lambda-exp a b) -> (lambda (x y) (+ x y).
  I'm not sure whether this actually works so it needs more testing."
  (eval `(lambda ,lambda-list ,@lambda-body)))

(defun make-full-vector (object expression)
  "Makes a vector of the length of the given expression with initial-element object. Used as a helper for the map function."
  (make-array (length expression) :initial-element object))

(defun remove-nth (sequence index)
  "Remove the nth item in a sequence."
  (append (subseq sequence 0 index) (subseq sequence (+ 1 index))))

(defun replace-with-randoms (expression)
  "Replaces all calls to the elt and access-weight functions with a call to the get-one-random function.
I programmed this while in a call with a bunch of anarchists and it worked on the first try. I don't know why it works. Just go with it."
  (if (not (equal (type-of expression) 'cons)) ; not cons
                                        ;for demo purposes only. The following two lines of code are unnecessary for the functioning of the program
      (if (or (equal expression 'x) (equal expression 'z) (equal expression 'zl))
          (return-from replace-with-randoms (get-one-random))
          (return-from replace-with-randoms expression)) ;This one is necessary
      (if (or (equal (elt expression 0) 'elt) (equal (elt expression 0) 'aref))
          (return-from replace-with-randoms (get-one-random))
          (return-from replace-with-randoms (map 'list #'replace-with-randoms expression)))))

(defun replace-with-accesses (expression n)
  "Replaces all references to prev-node with (elt prev-nodes n), and all instances of (elt weights m) with (access-weight node n m)."
  (when (not (equal (type-of expression) 'cons))
      (if (equal expression 'prev-node)
          (return-from replace-with-accesses `(elt prev-nodes ,n)))
      (if (equal expression 'network-output)
          (return-from replace-with-accesses `(elt network-outputs ,n)))
      (if (equal expression 'labelled-output)
          (return-from replace-with-accesses `(elt labelled-outputs ,n)) ; this is obviously garbage, but not putting in the effort for the general case yet
          (return-from replace-with-accesses expression)))
  (if (equal (elt expression 0) 'elt)
      (return-from replace-with-accesses `(aref weights ,n ,(elt expression 2)))
      (return-from replace-with-accesses (map 'list #'replace-with-accesses expression (make-full-vector n expression)))))

(defun get-zl-function (expression)
  "Goes through expression until it finds a (reduce-map #'*) function, then returns that function. Returns NIL if no such function is found."
  (if (not (equal (type-of expression) 'cons))
      (return-from get-zl-function)
      (if (and (equal (elt expression 0) 'reduce-map) (equal (elt expression 1) '#'*)) ; if expression has only one element, it shouldn't be reduce-map, lol
          (return-from get-zl-function expression)
          (let ((map-result (remove NIL (map 'list #'get-zl-function expression))))
            (return-from get-zl-function (if map-result (elt map-result 0) map-result))))))

(defun replace-zl-function (expression)
  "Goes through expression until it finds a (reduce-map #'*) function, then returns the original expression with the reduce-map replaced by zl."
  (if (not (equal (type-of expression) 'cons))
      (return-from replace-zl-function expression)
      (if (and (equal (elt expression 0) 'reduce-map) (equal (elt expression 1) '#'*))
          (return-from replace-zl-function 'zl)
          (return-from replace-zl-function (map 'list #'replace-zl-function expression)))))

(defmacro locked-lambda (&body body)
  "Lambda function with the arguments locked into (prev-node weights). The user will have to use these arguments in the body."
  `(lambda (prev-node weights) ,@body))

(defmacro cost-lambda (&body body)
  "Locked lambda for the cost function. Locked into the arguments (labelled-output network-output)."
  `(lambda (labelled-output network-output) ,@body))

(defmacro reduce-map (func1 func2 &rest sequences)
  "Calls (reduce func1) on the result of (map func2 sequences)."
  `(reduce ,func1 (map 'list ,func2 ,@sequences)))

(defmethod map-array (array function
                      &optional (retval (make-array (array-dimensions array)))) ; copied from lisp-stat/numerical-utilities cause they haven't moved it back to aops
  "Apply FUNCTION to each element of ARRAY
Return a new array, or write into the optional 3rd argument."
  (dotimes (i (array-total-size array) retval)
    (setf (row-major-aref retval i)
          (funcall function (row-major-aref array i)))))

(defmacro funcall-break-list (func arg-list)
  `(funcall ,func ,@arg-list))

(defun gen-pattern-random (size func node-spreader arg-count return-count lower-bound upper-bound)
  "Generates a labelled list of data based on a given function. Calls the function with random numbers between lower-bound and upper-bound.
This is not a very powerful function. It's used mainly for tests on various networks using extremely basic patterns such as y=x^2.
@size The amount of data to generate.
@func The function in question. This cannot take anything besides numbers as input.
@node-spreader Function that takes in one number and returns a list of numbers, if you want several output nodes. Use #'list for no change.
@arg-count The number of arguments with which we are calling function. This is how many random numbers we generate for each call.
@return-count The number of elements in node-spreader's output list. This is how many output nodes are in the network.
@lower-bound Lower bound for random number generation.
@upper-bound Upper bound for random number generation."
  (let* ((generator (evaluate-lambda-exp '() `(+ ,lower-bound (random ,(float (- upper-bound lower-bound)))))) (inputs (aops:generate generator (list size arg-count))) (outputs (make-array (list size return-count))))
    (dotimes (input size)
      (let* ((input-nums (loop for i upto (- arg-count 1) collect (aref inputs input i))) (output (funcall node-spreader (eval `(funcall ,func ,@input-nums)))))
        (dotimes (output-num (length output))
          (setf (aref outputs input output-num) (elt output output-num)))))
    (list inputs outputs)))
            

; PART 3: DERIVATIVES (or: Functional Programming Hell)

(defparameter *derivative-table* NIL)

(defmacro add-one-deriv (x y)
                                        ; Helper macro for add-derivatives. Adds one function and its derivative to the table.
  `(progn (setf (gethash ,x *derivative-table*) ,y)
          (setf (gethash (get-function-name ,x) *derivative-table*) T)))

(defun add-derivatives (&rest args)
                                        ; Helper function for generate-derivative-table.
                                        ; Works like setf, taking an even number of arguments. Instead of setting A to B, it sets hash A to value B.
  (dotimes (binding (/ (length args) 2))
    (let ((func (elt args (* binding 2))) (deriv (elt args (+ (* binding 2) 1))))
      (add-one-deriv func deriv))))

(defun generate-derivative-table ()
  "The derivative table is the base for the get-derivative function. It's a lookup table for the most elementary derivatives."
  (defparameter *derivative-table* (make-hash-table)) ; Global hash table. Access this using (gethash [function] *derivative-table*)
  (add-derivatives
                   #'sin #'cos
                   #'cos #'nsin
                   #'exp #'exp
                   #'log #'inv
                   #'inv #'nis
                   #'relu #'sgn
                   #'asin #'asinp
                   #'acos #'acosp
                   #'atan #'atanp
                   ))

(generate-derivative-table)

(defun get-derivative (func x &optional zl-function)
  "Takes in a lambda body, such as (sin x), and returns a lambda body corresponding to the function's derivative, such as (cos x).
  @func The function to be differentiated.
  @x The variable we're finding the derivative with respect to. (x * sin(z))' with respect to x is sin(z), but with respect to z it's x * cos(z).
  @zl-function If the symbol zl is encountered in func, its derivative is defined as the derivative of zl-function. We use this to increase efficiency rather than calculating the same expression multiple times.
  Since each node has many weights and outer-params attached to it, and we need to know the derivative with respect to each one, this is very important."
  (if (equal func x) (return-from get-derivative 1)) ; derivative of x with respect to x is always 1. This is in here to cover weird forms of x such as elt expressions.
  (if (equal func 'zl) (return-from get-derivative (get-derivative zl-function x))) ; zl is essentially a mini-function stored in a separate variable. we take the derivative of that when we encounter it
  (if (or (not (equal (type-of func) 'cons)) (equal (elt func 0) 'aref) (equal (elt func 0) 'elt)) ; func is a symbol not equal to x, return 0
      (return-from get-derivative 0))
  (if (equal (length func) 1) ; func is a list containing 1 symbol. This is a variable.
      (if (equal (elt func 0) x) ; Are we finding the derivative with respect to this variable?
          (return-from get-derivative 1) ; Return 1 if yes
          (return-from get-derivative 0))) ; Return 0 if no (we are keeping this variable constant)
  (let ((func-car (car func)) ; The first item in the func list. This is a function, probably.        
        (func-cdr ; Everything else in the func list. These will be either variables or nested function calls.
          (if (< (length func) 3) ; if func contains only 2 elements, a function and a nested function
              (elt (cdr func) 0) ; We remove the nested list in this case. Otherwise recursive calls get messy e.g. (get-derivative '(sin (cos x)) 'x) -> (get-derivative '((cos x)) 'x) which is just a huge pain.
              (cdr func)))) ; Otherwise, we set func-cdr to (cdr func) as normal.
    
    (if (gethash func-car *derivative-table*) ; func-car is a function in the derivative table
        (let ((first-deriv (gethash (eval `(function ,func-car)) *derivative-table*))) ; Looks up the derivative.
          (return-from get-derivative ; Uses the chain rule.
            `(* (,(get-function-name first-deriv) ,func-cdr) ; f'(g(x))
                ,(get-derivative func-cdr x zl-function))))) ; g'(x)
    (if (equal func-car '+) ; sum
        (return-from get-derivative `(+ ,@(map 'list #'get-derivative func-cdr                       ; Use ,@ to break the outer list so we can access the elements using the + function. Reduce doesn't work.
                                               (make-full-vector x func-cdr)
                                               (make-full-vector zl-function func-cdr)))))                     ; Call get-derivative on each rest-atom, so we need this array of xs to pass into map
    (if (equal func-car '*) ; product rule
        (let* ((factors func-cdr) (factor-derivs (map 'list #'get-derivative factors (make-full-vector x factors) (make-full-vector zl-function factors))) (indices))
                                        ; Evaluate each item in factor-derivs with a random number inserted in place of all weight calls. Yes, this is unsafe. No, I do not care.
          (dotimes (index (length factors))
            (if (not (= 0 (eval (replace-with-randoms (elt factor-derivs index)))))
                (setf indices (append indices (list index))))) ; Keep the index of every evaluation that isn't 0
          (if (not indices) ; all derivatives are 0
              (return-from get-derivative 0))
          (if (< (length indices) 3) ; 1 or 2 derivatives are non-zero. Most efficient to store in the form f'gh + fg'h (where f g h are functions, and h' = 0)
              (let ((return-value '(+)))
                (if (< (length indices) 2) ; Expression will be only 1 element, remove the + at the beginning
                    (setf return-value NIL))
                (dolist (index indices)
                  (setf return-value (append return-value (list `(* ,(elt factor-derivs index) ,@(remove-nth factors index))))))
                (return-from get-derivative (if (> (length indices) 1) return-value (elt return-value 0))))
              
              (let ((return-value `(* ,func)) (plus-value '(+)))
                                        ; 3 or more derivatives are non-zero. Most efficient to store in the form fghj*(f'/f + g'/g + h'/h + j'/j) (where f g h j are functions, and all derivatives are non-zero)
                (dolist (index indices)
                  (setf plus-value (append plus-value (list `(* ,(elt factor-derivs index) (inv ,(elt factors index)))))))
                (return-from get-derivative (append return-value (list plus-value)))))))

    (if (equal func-car 'reduce-map) ; reduce-map function. This can only be used when looping through all the prevnodes.
                                        ; (elt rest-atoms 1) WILL BE a locked-lambda or node-lambda function. We can make assumptions based on this.
        (let ((replaced-func (replace-with-accesses (get-lambda-body (eval (elt func-cdr 1))) (elt x 2))))
          (if (equal (elt func-cdr 0) '#'+) ; adding up a list
              (return-from get-derivative (get-derivative replaced-func x zl-function)))
          (if (equal (elt func-cdr 0) '#'*) ; multiplication (works; kinda weird)
              (return-from get-derivative `(* zl ,(get-derivative replaced-func x zl-function) (inv ,replaced-func))))))))

; PART 4: NODE OBJECT

(defclass node (standard-object)
                                        ; This is the node object. Every network is a list of layers, each layer is a list of nodes.
                                        ; The node stores four main things: prev-node-count, weights, outer-params, and function.

                                        ; Behavior:
                                        ; When the network is activated, each node gets passed all the output values from the previous layer of nodes ("prev-nodes").
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
   (node-function :initarg :node-function :initform
                  `(lambda (prev-nodes weights outer-params) (+ (elt outer-params 0) (* (elt outer-params 1) (reduce-map #'* (locked-lambda (* prev-node (elt weights 0))) prev-nodes (aops:split weights 1))))) :accessor node-function
             :documentation "Function for the activation (output) of the node. Called every time the node is activated.")
   (weights-derivatives :accessor weights-derivatives
                        :documentation "2D array of functions for calculating the derivative of the node with respect to each weight.")
   (outer-params-derivatives :accessor outer-params-derivatives
                             :documentation "1D array of functions for calculating the derivative of the node with respect to each outer-param.")
   (prev-nodes-derivatives :accessor prev-nodes-derivatives
                           :documentation "List of functions for calculating the derivative of the node with respect to each prev-node.")
   (zl-function :accessor zl-function :initform NIL
                :documentation "Formula for the zl, which is anything that involves multiplying together a sequence. We calculate this separately to speed up derivative computation.")))

(defmethod initialize-instance :after ((node node) &key)
                                        ; This part handles finding the derivative equations for the node.
  (let* ((dims (array-dimensions (weights node))) (func (get-lambda-body (eval (node-function node)))) (wd-array (make-array dims)) (outer-params (outer-params node)) (zl-function (get-zl-function func))
         (deriv-lambda-list `(prev-nodes weights outer-params &optional zl)) (ignorable-list `(prev-nodes weights outer-params zl)))
    (setf func (replace-zl-function func))
    (when zl-function (setf (slot-value node 'zl-function) (evaluate-lambda-exp deriv-lambda-list zl-function)))
                                        ; If there is a zl function (the node function uses reduce-map #'*), then store it in the node.
    (dotimes (prev-node (elt dims 0)) ; Loop through the 2D array of weights
      (dotimes (weight (elt dims 1))
        (setf (aref wd-array prev-node weight) (evaluate-lambda-exp deriv-lambda-list `(declare (ignorable ,@ignorable-list)) ; This stops the compiler from giving us thousands of lines of warnings (good code alert)
                                                                    (get-derivative func `(aref weights ,prev-node ,weight) zl-function))))) ; Take derivative of formula with respect to each weight
    (setf (weights-derivatives node) wd-array)
                                        ; Loop through the 1D array of outer-params, and the list of prev-nodes
    (setf (outer-params-derivatives node) (map 'vector (lambda (func outer-param) (evaluate-lambda-exp deriv-lambda-list `(declare (ignorable ,@ignorable-list)) (get-derivative func outer-param zl-function)))
                                               (make-full-vector func outer-params) (loop for i upto (length outer-params) collect `(elt outer-params ,i))))
    (setf (prev-nodes-derivatives node) (map 'list (lambda (func prev-node) (evaluate-lambda-exp deriv-lambda-list `(declare (ignorable ,@ignorable-list)) (get-derivative func prev-node zl-function)))
                                             (make-array (elt dims 0) :initial-element func) (loop for i upto (elt dims 0) collect `(elt prev-nodes ,i))))
    (setf (node-function node) (evaluate-lambda-exp deriv-lambda-list `(declare (ignorable ,@ignorable-list)) (replace-zl-function func)))))

(defgeneric activate-node (node prev-nodes)
  )

(defmethod activate-node ((node node) prev-nodes)
  "Takes in a node and a list of outputs in the previous layer. Returns the output (number) for that node."
  (let ((zl NIL))
    (if (zl-function node) (progn (setf zl (funcall (zl-function node) prev-nodes (weights node) (outer-params node)))
                                  (funcall (node-function node) prev-nodes (weights node) (outer-params node) zl))
        (funcall (node-function node) prev-nodes (weights node) (outer-params node)))))

(defgeneric access-weight (node prev-node-index weight-index)
  )

(defmethod access-weight ((node node) prev-node-index weight-index)
  ; Helper function so I don't have to constantly use slice-2d-array and elt to access a specific weight. Obsolete, we now use aref.
  (elt (slice-2d-array (weights node) prev-node-index) weight-index))

(defgeneric print-node (node)
  )

(defmethod print-node ((node node))
  "Prints information about a node in a human-readable format."
  (format t "Weights: ~15t~a~%" (weights node))
  (format t "Outer params: ~15t~a~%" (outer-params node))) ; i don't know how to cleanly print the node function

(defun generate-weights (initializer prev-node-count weights-per-prev-node)
  "Generates a 2D weights array for a node using the aops library. Sets each weight to the result of a call to the (typically random) initializer function."
  (aops:generate initializer (list prev-node-count weights-per-prev-node)))

; PART 5: NETWORK OPERATIONS

(defun activate-network (network input-nodes)
  "Activates an entire network given a list of input nodes."
  (let ((prev-nodes input-nodes) (current-nodes) (nodes-in-layer))
    (dotimes (layer-number (length network) current-nodes) ; for each layer in the network
      (setf nodes-in-layer (length (elt network layer-number))) ; find how many nodes are in the layer so we can make our vector
      (setf current-nodes (make-array nodes-in-layer :fill-pointer 0)) ; make an empty vector for the current layer
      (dotimes (node-number nodes-in-layer) ; for each node in the layer
        (vector-push (activate-node (elt (elt network layer-number) node-number) prev-nodes) current-nodes)) ; push the activate-node result to the vector
      (setf prev-nodes current-nodes)))) ; moving to the next layer. This also returns current-nodes, meaning we don't need a dedicated return statement.

(defun get-activation-list (network input-nodes)
  "Like activate-network, but returns a list of every activation, including the input-nodes."
  (let ((activations (list input-nodes)) (prev-nodes input-nodes) (current-nodes) (nodes-in-layer))
    (dotimes (layer-number (length network) activations) ; returns the activations variable at the end of the dotimes
      (setf nodes-in-layer (length (elt network layer-number)))
      (setf current-nodes (map 'list #'activate-node (elt network layer-number) (make-array nodes-in-layer :initial-element prev-nodes))) ; map function is overpowered. no idea how this works
      (setf activations (append activations (list current-nodes)))
      (setf prev-nodes current-nodes))))

(defun print-network (network)
  "Prints information about a network in a human-readable format."
  (dotimes (layer (length network))
    (format t "~%LAYER ~a~%" layer)
    (dotimes (node (length (elt network layer)))
      (format t "~%Node ~a~%" node)
      (print-node (elt (elt network layer) node)))))    

(defun initialize-network-mono (sizes initializer node-function num-weights num-oparams)
  "Creates the network of nodes, sets every weight and bias according to the initializer.
Each layer is a list of node objects.
The entire network is a list containing every layer.
@sizes List of integers describing number of nodes in each layer.
@initializer Function that initializes value for each weight and bias. Currently using get-one-random.
@node-function The function called whenever the node is activated.
@num-weights How many weights per prev-node. Usually 1.
@num-oparams How many outer params in each node. Usually 2, bias and coefficient."
  (defparameter *norm-function* #'relu)
  (let ((sizes-no-input (cdr sizes)) (network NIL))
                                        ; NOTE: The input layer does NOT have dedicated node objects, because its values are given by an outside source.
                                        ; For example, an input layer might be how dark a pixel in an image is, for an image recognition network.
                                        ; The input layer itself goes through no function. This is why we use remove-first.
    
    (dotimes (layer (length sizes-no-input))
      (let* ((layer-size (elt sizes-no-input layer)) (layer-data (make-array layer-size :fill-pointer 0))) ; Make an empty vector for the layer.
        (dotimes (current-node layer-size)
          (vector-push (make-instance 'node ; Makes a new node to put in the vector.
                                      :prev-node-count (elt sizes layer) ; (elt sizes layer) is equal to (elt sizes-no-input layer) - 1, meaning it's a handy way of accessing the size of the previous layer.
                                        ; This is necessary to give the nodes input about how many nodes are in the previous layer; for example, how many weights they need.
                                      :weights (generate-weights initializer (elt sizes layer) num-weights)
                                      :outer-params (aops:generate initializer num-oparams)
                                      :node-function node-function)
                       layer-data))
        (setf network (append network (list layer-data)))))
    (return-from initialize-network-mono network)))

(defun make-gradient-list (network)
  "Makes a new gradient list with the dimensions of the given network. All values start at 0."
  (let ((glist NIL))
    (dolist (layer network)
      (setf glist (append glist (list (map 'list (lambda (node) (list (make-array (array-dimensions (weights node)) :initial-element 0) (make-array (array-dimensions (outer-params node)) :initial-element 0))) layer)))))
    glist))

(defun modify-glist (glist func)
  "Calls func on every number in the gradient list. Typically this involves multiplying it by a small negative number (descent rate).
@glist A gradient list. See make-gradient-list and add-gradients.
@func A function that takes one number as input."
  (dolist (layer glist)
    (dolist (node layer)
      (setf (elt node 0) (aops:each func (elt node 0))) ; weights
      (map-into (elt node 1) func (elt node 1))))
  glist)

(defun add-gradients (network activation-list error-list gradient-list)
  "Goes backwards through the network, computes gradients for each weight and parameter, and adds them to their corresponding spot in gradient-list.
@network The network being trained.
@activation-list The list of network activations. Needed for calculating derivatives that use zl or access prev-nodes.
@error-list The list of delta(cost)/delta(node) for each output node in the network.
@gradient-list The gradient list to be modified."
  ; (format t "~%Error list is ~a." error-list)
  (let ((layer-data) (layer-size) (current-dcdn error-list) (zl 2) (prev-nodes) (node) (weights) (outer-params))
                                        ; Goes through the network backwards.
                                        ; Starts by taking using the cost function derivative formulas.
    (do ((layer (- (length network) 1) (1- layer))) ((< layer 0)) ; for each layer
      (setf layer-data (elt network layer) layer-size (length layer-data) prev-nodes (elt activation-list layer))
      (if (< layer (- (length network) 1)) ;not the output layer, we need to compute new delta(cost)/delta(node)
          (let ((new-dcdn (make-array layer-size :initial-element 0)))
            (dotimes (node-number (length (elt network (+ 1 layer)))) ; loop through every output in the future layer
              (setf prev-nodes (elt activation-list layer)
                    node (elt (elt network (+ 1 layer)) node-number)
                    weights (weights node)
                    outer-params (outer-params node)
                    zl (if (zl-function node) (funcall (zl-function node) prev-nodes weights outer-params)))
              ; (print zl)
              (let ((len (prev-node-count node)))
                (print len)
                (map-into new-dcdn #'+ new-dcdn (map 'vector (lambda (old-dcdn pnd prev-nodes weights outer-params zl) (* old-dcdn (funcall pnd prev-nodes weights outer-params zl)))
                                                     (make-array len :initial-element (elt current-dcdn node-number)) (prev-nodes-derivatives node)
                                                     (make-array len :initial-element prev-nodes) (make-array len :initial-element weights)
                                                     (make-array len :initial-element outer-params) (make-array len :initial-element zl))))) ;this is kind of terrible. should be replaced with loop statement
            ; (format t "~%delta(cost)/delta(node) for layer ~a: ~a" layer new-dcdn)
            (setf current-dcdn new-dcdn)))
      (dotimes (node-number layer-size) ; Loops through every node in the layer.
        (setf node (elt layer-data node-number) weights (weights node) outer-params (outer-params node) zl (if (zl-function node) (funcall (zl-function node) prev-nodes weights outer-params))) ;sets up parameters
        (dotimes (prev-node (array-dimension weights 0)) ; Loops through every weight in the node.
          (dotimes (weight (array-dimension weights 1))
            ; (format t "~%delta(node)/delta(weight) for weight ~a ~a ~a ~a: ~a" layer node-number prev-node weight (funcall (aref (weights-derivatives node) prev-node weight) prev-nodes weights outer-params zl))
            (incf (aref (elt (elt (elt gradient-list layer) node-number) 0) prev-node weight) ; find the correct place in the gradient-list. Lot of lookups, might be more efficient to do a whole node or layer at a time then add it.
                  (*
                   (funcall (aref (weights-derivatives node) prev-node weight) prev-nodes weights outer-params zl) ; delta(node)/delta(weight)
                   (elt current-dcdn node-number))))) ; delta(cost)/delta(node)
        (dotimes (outer-param (length outer-params)) ; Loops through every outer-param in the node.
          (incf (elt (elt (elt (elt gradient-list layer) node-number) 1) outer-param) ; works similar to the weight one
                (*
                 (funcall (elt (outer-params-derivatives node) outer-param) prev-nodes weights outer-params zl)
                 (elt current-dcdn node-number)))) ; This elt statement can probably be stored in a variable for more efficiency.
        ))
    gradient-list))

(defun apply-glist (network glist)
  "Adds every element in the gradient list to its corresponding place in the network."
  (dotimes (layer-number (length network))
    (dotimes (node-number (length (elt network layer-number)))
      (let ((node (elt (elt network layer-number) node-number)) (glist-node (elt (elt glist layer-number) node-number)))
        (setf (weights node) (aops:each #'+ (weights node) (elt glist-node 0)))
        (setf (outer-params node) (map 'vector #'+ (outer-params node) (elt glist-node 1)))))))
    
(defun train (network cost-function descent-rate labelled-inputs labelled-outputs batch-size)
  "Trains a network on a set of labelled data. Goes through the entire list of data given. Prints stats along the way. Returns the network afterwards.
@network The network to be trained.
@cost-function The cost function, typically (reduce-map #'+ (lambda (labelled-output network-output) (* (+ network-output (* -1 labelled-output)) (+ network-output (* -1 labelled-output)))) labelled-outputs network-outputs).
I don't know if this works properly with the get-derivative helper functions so that will have to be tested. Also maybe we will have powers implemented in get-derivative one day.
@descent-rate The rate at which the weights and outer-params are modified. Must be a negative number. If it's too close to 0 the network will adjust slowly, if too far it will overshoot and you'll get garbage.
@labelled-inputs 2D vector containing 1D vectors of data to be sent through the input nodes of the network.
@labelled-outputs 2D vector containing 1D vectors of the desired outputs for said input data.
@batch-size The number of iterations per batch. After each batch, the network is modified by the gradient list. Small batches cost more computation time and can cause swings in the network if you get outlier data,
while large batches will obviously take forever to train unless you use a wacky wavy descent function."
  (let* ((last-layer-length (length (elt (last network) 0)))
         (cost-function (eval cost-function)) ; initial cost function will be quoted, otherwise the compiler will not give us access to the body and thus get-derivative
         (dcdn-formulas (map 'list #'get-derivative (make-array last-layer-length :initial-element (get-lambda-body cost-function)) (loop for i upto last-layer-length collect `(elt network-outputs ,i))))
         (dcdn-functions (map 'list #'evaluate-lambda-exp (make-array last-layer-length :initial-element '(labelled-outputs network-outputs)) dcdn-formulas))
         (descent-function (lambda (x) (* x descent-rate))) ; You can change this or modify the parameters if you want a non-linear descent function
         (batch-counter 0)
         (epoch-counter 0)
         (cost-sum 0)
         (glist (make-gradient-list network)))
    (dotimes (datum (aops:nrow labelled-inputs))
      (when (>= batch-counter batch-size) ; batch is finished
        (setf glist (modify-glist glist descent-function)) ; apply descent function to gradient list
        (print glist)
        (apply-glist network glist) ; apply gradient list to network
        (setf glist (make-gradient-list network)) ; make a new empty gradient list
        (incf epoch-counter)
        (format t "~%~%Epoch ~a. Average cost is ~a.~%" epoch-counter (/ cost-sum batch-size))
        (setf batch-counter 0 cost-sum 0)) ; reset the batch counter
      (let* ((labelled-input (coerce (slice-2d-array labelled-inputs datum) 'list)) ; awful
             (network-alist (get-activation-list network labelled-input))
             (network-outputs (elt (last network-alist) 0)) ; probably not even a good fix this is where the actual training part happens
             (labelled-output (coerce (slice-2d-array labelled-outputs datum) 'list)))
        ; (print (get-lambda-body (elt dcdn-functions 0)))
                                        ; (print dcdn-functions)
        (format t "~%Cost for this iteration was ~a.~%" (funcall cost-function labelled-output network-outputs))
        (format t "When given the input ~a, the network responded with ~a. The correct response was ~a." labelled-input network-outputs labelled-output)
        (incf cost-sum (funcall cost-function labelled-output network-outputs)) 
        (add-gradients network network-alist
                       (map 'list (lambda (dcdn-function labelled-outputs network-outputs) (funcall dcdn-function labelled-outputs network-outputs)) dcdn-functions
                                                  (make-array last-layer-length :initial-element labelled-output)
                                                  (make-array last-layer-length :initial-element network-outputs))
                       glist)
        (incf batch-counter)))))
