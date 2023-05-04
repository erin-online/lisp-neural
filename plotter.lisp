(ql:quickload :lisp-stat)
(in-package :ls-user)

(defun train-draw ())

(defun draw-one (title network correct-fun &optional (min 0) (steps 500))
  ; first get the data and package it into a data-frame
  (let ((data (plist-df
		`(:x-coords ,(make-array (* 2 (+ 1 steps)) :initial-contents (append (loop for i upto steps collect (+ min (/ i steps))) (loop for i upto steps collect (+ min (/ i steps))))) ;x coords. you need 2 copies of them to hold both the network outputs and the correct outputs
		  :y-coords ,(make-array (* 2 (+ 1 steps)) :initial-contents (append (loop for i upto steps collect (elt (activate-network network (list (+ min (/ i steps)))) 0)) ;y coords for neural network
								              (loop for i upto steps collect (funcall correct-fun (+ min (/ i steps)))))) ;y coords for correct function
		  :symbols ,(make-array (* 2 (+ 1 steps)) :initial-contents (append (loop for i upto steps collect 'network-outputs) (loop for i upto steps collect 'shape-to-fit))))))) ;symbols
    ;(print-data data)
    (plot:plot (vega:defplot multi-series-line-chart `(:title ,title
						       :data ,data
						       :mark :line
						       :encoding (:x (:field :x-coords
								      :type :quantitative)
								  :y (:field :y-coords
								      :type :quantitative)
								  :color (:field :symbols
								          :type :nominal)))))))
						   
