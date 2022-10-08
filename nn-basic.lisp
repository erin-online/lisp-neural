; Basic neural network intended for simple image recognition.
; Doesn't work with stuff like chess because I haven't figured that out yet.
; Chess is hard and neural networks are hard.
;
; NOTE: This program requires the array-operations or aops package. Load it with (ql:quickload :array-operations). Eventually I'll add something into the code to load it automatically.
;
; TO-DO:
;✓1. Node as an object, represent each layer as a list of nodes. Also, allow for the creation of multiple networks.
;✓2. Create an alternative to activate-network that returns the list of activations for every layer, not just the last ones.
; 3. Finish backpropagation using the function from 2
; 4. Work on file I/O for image recognition using data from the MNIST database http://yann.lecun.com/exdb/mnist/. This will result in a basic functioning neural network.
; 5. Allow for networks to be imported and exported via files.
; 6. Explore xenodes and other ideas

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
  Watch out for variable name conflicts--variable names are returned as-is."
  (elt (function-lambda-expression func) 2))

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

(defun remove-first (sequence)
  "Remove the first item in a sequence. This is basically the best function in the program."
  (remove (elt sequence 0) sequence :count 1))

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
  (if (not (equal (type-of expression) 'cons))
      (if (equal expression 'prev-node)
          (return-from replace-with-accesses `(elt prev-nodes ,n))
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

(defmacro reduce-map (func1 func2 &rest sequences)
  "Calls (reduce func1) on the result of (map func2 sequences)."
  `(reduce ,func1 (map 'list ,func2 ,@sequences)))

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

(defun get-derivative (func x &optional zl-function)
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
  (if (equal func x) (return-from get-derivative 1))
  (if (equal func 'zl) (return-from get-derivative (get-derivative zl-function x)))
  (if (or (not (equal (type-of func) 'cons)) (equal (elt func 0) 'aref) (equal (elt func 0) 'elt)) ; See below, except if the variable is not in a list
      (return-from get-derivative 0))
  (if (equal (length func) 1) ; One atom is in the provided lambda expression. This is a variable. If it is not, someone else screwed up.
      (if (equal (elt func 0) x) ; Are we finding the derivative with respect to this variable?
          (return-from get-derivative 1) ; Return 1 if yes
          (return-from get-derivative 0))) ; Return 0 if no (we are keeping this variable constant)
  (let ((first-atom (elt func 0)) ; The first item in the func list. This is a function, probably.
                                        ; (second-atom (if (eql (type-of (elt func 1)) 'cons) (elt (elt func 1) 0) (elt func 1)))
                                        ; I don't know if I need second-atom so I'm keeping it around in comment form
        
        (rest-atoms ; The rest of the atoms. Everything except first-atom.
                                        ; This works but needs to be cleaned up. I commented out Condition 1 in order to let in symbols, to stop the (sin (x)) issue.
                                        ; Assuming we are using this as a permanent fix (very possible), the comments need to be redone a bit.
          
          (if (and t ;eql (type-of (elt (remove-first func) 0)) 'cons) ; Condition 1: The second item in the func list is a cons cell (list).
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
                ,(get-derivative rest-atoms x zl-function))))) ; g'(x)
    (if (equal first-atom '+) ; sum
        (return-from get-derivative `(+ ,@(map 'list #'get-derivative rest-atoms                       ; Use ,@ to break the outer list so we can access the elements using the + function. Reduce doesn't work.
                                               (make-full-vector x rest-atoms)
                                               (make-full-vector zl-function rest-atoms)))))                     ; Call get-derivative on each rest-atom, so we need this array of xs to pass into map
    (if (equal first-atom '*) ; product rule
        (let* ((factors rest-atoms) (factor-derivs (map 'list #'get-derivative factors (make-full-vector x factors) (make-full-vector zl-function factors))) (indices))
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

    (if (equal first-atom 'reduce-map) ; reduce-map function. This can only be used when looping through all the prevnodes.
                                        ; (elt rest-atoms 1) WILL BE a lambda function that takes prev-node and weights as arguments (see locked-lambda). We can make assumptions based on this.
        (let ((replaced-func (replace-with-accesses (get-lambda-body (eval (elt rest-atoms 1))) (elt x 2))))
          (if (equal (elt rest-atoms 0) '#'+) ; adding up a list
              (return-from get-derivative (get-derivative replaced-func x zl-function)))
          (if (equal (elt rest-atoms 0) '#'*) ; multiplication (works; kinda weird)
              (return-from get-derivative `(* zl ,(get-derivative replaced-func x zl-function) (inv ,replaced-func))))))))

; PART 4: NODE OBJECT

(defclass node (standard-object)
                                        ; This is the node object. Every network is a list of layers, each layer is a list of nodes.
                                        ; The node stores four main things: prev-node-count, weights, outer-params, and function.

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
   (node-function :initarg :node-function :initform
                  `(lambda (prev-nodes weights outer-params) (+ (elt outer-params 0) (* (elt outer-params 1) (reduce-map #'* (locked-lambda (* prev-node (elt weights 0))) prev-nodes (aops:split weights 1))))) :accessor node-function
             :documentation "Function for the activation (output) of the node. Called every time the node is activated.")
   (weights-derivatives :accessor weights-derivatives
                        :documentation "2D array of functions for calculating the derivative of the node with respect to each weight.")
   (outer-params-derivatives :accessor outer-params-derivatives
                             :documentation "1D array of functions for calculating the derivative of the node with respect to each outer-param.")
   (prev-nodes-derivatives :accessor prev-nodes-derivatives
                           :documentation "List of functions for calculating the derivative of the node with respect to each prev-node.")
   (zl-function :accessor zl-function
                :documentation "Formula for the zl, which is anything that involves multiplying together a sequence. We calculate this separately to speed up derivative computation.")))

(defmethod initialize-instance :after ((node node) &key)
                                        ; This part handles finding the derivative equations for the node.
  (let* ((dims (array-dimensions (weights node))) (func (get-lambda-body (eval (node-function node)))) (wd-array (make-array dims)) (outer-params (outer-params node)) (zl-function (get-zl-function func))
         (deriv-lambda-list `(prev-nodes weights outer-params)))
    (setf func (replace-zl-function func))
    (when zl-function (setf (slot-value node 'zl-function) (evaluate-lambda-exp deriv-lambda-list zl-function)) (setf deriv-lambda-list (append deriv-lambda-list (list 'zl))))
                                        ; If there is a zl function (the node function uses reduce-map #'*), then store it in the node, and make sure every derivative formula in the node takes zl as an argument.
    (dotimes (prev-node (elt dims 0)) ; Loop through the 2D array of weights
      (dotimes (weight (elt dims 1))
        (setf (aref wd-array prev-node weight) (evaluate-lambda-exp deriv-lambda-list `(declare (ignorable ,@deriv-lambda-list)) ; This stops the compiler from giving us thousands of lines of warnings (good code alert)
                                                                    (get-derivative func `(aref weights ,prev-node ,weight) zl-function))))) ; Take derivative of formula with respect to each weight
    (setf (weights-derivatives node) wd-array)
                                        ; Loop through the 1D array of outer-params, and the list of prev-nodes
    (setf (outer-params-derivatives node) (map 'vector (lambda (func outer-param) (evaluate-lambda-exp deriv-lambda-list `(declare (ignorable ,@deriv-lambda-list)) (get-derivative func outer-param zl-function)))
                                               (make-full-vector func outer-params) (loop for i upto (length outer-params) collect `(elt outer-params ,i))))
    (setf (prev-nodes-derivatives node) (map 'list (lambda (func prev-node) (evaluate-lambda-exp deriv-lambda-list `(declare (ignorable ,@deriv-lambda-list)) (get-derivative func prev-node zl-function)))
                                             (make-array (elt dims 0) :initial-element func) (loop for i upto (elt dims 0) collect `(elt prev-nodes ,i))))
    (setf (node-function node) (evaluate-lambda-exp deriv-lambda-list `(declare (ignorable ,@deriv-lambda-list)) (replace-zl-function func)))))

(defgeneric activate-node (node prev-nodes)
  )

(defmethod activate-node ((node node) prev-nodes)
  "Takes in a node and a list of outputs in the previous layer. Returns the output (number) for that node."
  (let ((weights (weights node)) (outer-params (outer-params node))) ; Maybe a little unnecessary. This made more sense when we still had the separated function.
    (funcall (node-function node) prev-nodes weights outer-params)))

(defgeneric access-weight (node prev-node-index weight-index)
  )

(defmethod access-weight ((node node) prev-node-index weight-index)
  ; Helper function so I don't have to constantly use slice-2d-array and elt to access a specific weight.
  (elt (slice-2d-array (weights node) prev-node-index) weight-index))

(defun generate-weights (initializer prev-node-count weights-per-prev-node)
  "Generates a 2D weights array for a node using the aops library. Sets each weight to the result of a call to the (typically random) initializer function."
  (aops:generate initializer (list prev-node-count weights-per-prev-node)))

; PART 5: NETWORK OPERATIONS

(defun activate-network (network input-nodes)
  "Activates an entire network given a list of input nodes."
  (let ((prev-nodes input-nodes) (current-nodes) (nodes-in-layer))
    (dotimes (layer-number (length network) current-nodes)
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

(defun initialize-network-mono (sizes initializer)
  "Creates the network of nodes, sets every weight and bias according to the initializer.
Each layer is a list of node objects.
The entire network is a list containing every layer.
@sizes List of integers describing number of nodes in each layer.
@initializer Function that initializes value for each weight and bias. Currently using get-one-random."
  (defparameter *norm-function* #'relu)
  (let ((sizes-no-input (remove-first sizes)) (network NIL))
                                        ; NOTE: The input layer does NOT have dedicated node objects, because its values are given by an outside source.
                                        ; For example, an input layer might be how dark a pixel in an image is, for an image recognition network.
                                        ; The input layer itself goes through no function. This is why we use remove-first.
    
    (dotimes (layer (length sizes-no-input))
      (let* ((layer-size (elt sizes-no-input layer)) (layer-data (make-array layer-size :fill-pointer 0))) ; Make an empty vector for the layer.
        (dotimes (current-node layer-size)
          (vector-push (make-instance 'node ; Makes a new node to put in the vector.
                                      :prev-node-count (elt sizes layer) ; (elt sizes layer) is equal to (elt sizes-no-input layer) - 1, meaning it's a handy way of accessing the size of the previous layer.
                                        ; This is necessary to give the nodes input about how many nodes are in the previous layer; for example, how many weights they need.
                                      :weights (generate-weights initializer (elt sizes layer) 1)
                                      :outer-params (aops:generate initializer 2))
                       layer-data))
        (setf network (append network (list layer-data)))))
    (return-from initialize-network-mono network)))

(defun add-gradients (network activation-list error-list gradient-list)
  "Goes backwards through the network, computes gradients for each weight and parameter, and adds them to their corresponding spot in gradient-list.
@network The network being trained.
@activation-list The list of network activations. Needed for calculating derivatives that use zl or access prev-nodes.
@error-list The list of delta(cost)/delta(node) for each output node in the network.
@gradient-list The gradient list to be modified."
  (let ((layer-data) (layer-size) (current-dcdn error-list) (zl 2) (prev-nodes) (node) (weights))
                                        ; Goes through the network backwards.
                                        ; Starts by taking using the cost function derivative formulas.
    (do ((layer (- (length network) 1) (1- layer))) ((< layer 0))
      (setf layer-data (elt network layer) layer-size (length layer-data) prev-nodes (elt activation-list layer))
      (if (< layer (- (length network) 1)) ;not the output layer, we need to compute new delta(cost)/delta(node)
          (let ((new-dcdn (make-array layer-size :initial-element 0)))
            (dotimes (node-number (length (elt network (+ 1 layer)))) ; loop through every output in the future layer
              (setf node (elt (elt network (+ 1 layer)) node-number) weights (weights node) zl (get-zl-function (node-function node)))
              (print zl)
              (map-into new-dcdn #'+ new-dcdn (map 'vector #'eval (prev-nodes-derivatives node))))
            (map-into new-dcdn #'* new-dcdn current-dcdn)
            (print new-dcdn)
            (setf current-dcdn new-dcdn)))
      (dotimes (node-number layer-size) ; Loops through every node in the layer.
        (setf node (elt layer-data node-number) zl (get-zl-function (node-function node)))
        
        
    ))))
    
(defun train (network cost-function descent-rate labeled-data-list)
  (let ((dcdn-formulas (map 'list #'get-derivative (make-array (length (elt (last network) 0)) :initial-element cost-function) (loop for i upto (length (elt (last network) 0)) collect `(elt activations ,i)))))
    ; cost-function is typically defined as #'reduce-map f1 f2 activations correct-values.
    ))
