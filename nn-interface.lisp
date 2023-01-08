                                        ; Interface file meant for use by the user of the program to generate their own custom neural networks.
                                        ; Maybe in the future an in-program interface, perhaps with some sort of GUI, can be considered.
(defparameter *nf-presets* ; node function presets
  (list
   `(lambda (prev-nodes weights outer-params) (relu (+ (elt outer-params 0) (reduce-map #'+ (locked-lambda (* prev-node (elt weights 0))) prev-nodes (aops:split weights 1)))))
   `(lambda (prev-nodes weights outer-params) (+ (elt outer-params 0) (* (elt outer-params 1) (reduce-map #'+ (locked-lambda (* prev-node (elt weights 0))) prev-nodes (aops:split weights 1)))))
   `(lambda (prev-nodes weights outer-params) (+ (elt outer-params 0) (reduce-map #'+ (locked-lambda (* (elt weights 0) (sin (* (elt weights 1) prev-node)))) prev-nodes (aops:split weights 1))))
   `(lambda (prev-nodes weights outer-params) (exp (* (log (elt outer-params 0)) (reduce-map #'+ (locked-lambda (* prev-node (elt weights 0))) prev-nodes (aops:split weights 1)))))
   `(lambda (prev-nodes weights outer-params) (relu (+ (elt outer-params 0) (reduce-map #'+ (locked-lambda (* prev-node (elt weights 0) (elt weights 1))) prev-nodes (aops:split weights 1)))))))

(defparameter *cf-presets* ; cost function presets
  (list
   `(lambda (labelled-outputs network-outputs) (reduce-map #'+ (cost-lambda (* (+ network-output (* -1 labelled-output)) (+ network-output (* -1 labelled-output)))) labelled-outputs network-outputs))))

(defun make-net (sizes node-function)
  "Makes a network with size specified by sizes. For example, entering '(4 5 6) into the sizes field will give a network with 4 input nodes, 5 middle nodes, and 6 output nodes.
The node-function determines the output value of the node, given an input of a prev-node list. We have several presets here, which you can access using (elt *nf-presets* n) where n is between 0 and 2."
  (initialize-network-mono sizes #'get-one-random node-function))

(defun train-on-generated-data (network cost-function descent-rate batch-size generating-function)
  "Trains a network on procedurally generated data based on a function.
Basically, we have a simple function that takes in one number argument, such as sin or exp. You can also make your own with lambda.
We then send 50 random numbers between -5 and 5 into this function, saving both the inputs and outputs, then train the network on this data.
(This only works for networks with 1 input node and 1 output node. For others, refer to gen-pattern-random in nn-basic, or find another way to get data.)

Params:
network is the network to be trained
cost-function takes in a list of correct outputs and a list of the network's outputs, and gives a number based on how close they are. You want this to be lower. The standard cost function is available in the presets.
descent-rate is how much the network is modified after each batch. Must be negative to make sense. Don't make this too far from 0 or funny things will happen. (More then they already do, of course.)
batch-size is how many times you run the network before modifying it. Each batch has a gradient list that accumulates as more gradients are computed. The higher this is, the closer to 0 descent-rate must be.
generating-function is a function that takes in a number and spits out a number. The role of this is described in the first paragraph."
  (let ((data (gen-pattern-random 50 generating-function 'list 1 1 -5 5)))
    (train network cost-function descent-rate (elt data 0) (elt data 1) batch-size)))

                                        ; don't mind me, just keeping these around for testing
                                        ;(setf (outer-params (elt (elt my-net 0) 0)) #(-.5 69) (outer-params (elt (elt my-net 0) 1)) #(-.9 69) (outer-params (elt (elt my-net 0) 2)) #(-.2 69) (outer-params (elt (elt my-net 0) 3)) #(-.7 69))
                                        ;(setf (weights (elt (elt my-net 0) 0)) #2A((0.6)) (weights (elt (elt my-net 0) 1)) #2A((0.6)) (weights (elt (elt my-net 0) 2)) #2A((-0.2)) (weights (elt (elt my-net 0) 3)) #2A((0.8)))
                                        ;(setf (weights (elt (elt my-net 1) 0)) #2A((-.7) (-.7) (-.1) (-.3)) (outer-params (elt (elt my-net 1) 0)) #(-.9 420))
