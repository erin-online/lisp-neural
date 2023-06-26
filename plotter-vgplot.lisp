(defun draw-one (title network original-fun &key (min 0) (max 1) (step 0.01) (y-min 0) (y-max 1))
  (let* ((x-range (vgplot:range min max step))
	(original-function (map 'vector original-fun x-range))
	 (network-approximation (map 'vector (lambda (x) (elt (activate-network network (list x)) 0)) x-range)))
    (vgplot:plot x-range original-function "b;Original Function;" x-range network-approximation "r;Network Approximation;")
    (vgplot:axis (list min max y-min y-max))
    (vgplot:title title)))

(defun print-set (title network original-fun train-per-img images &key (path "lisp/plots/image") (descent-rate -0.002) (cost-function *cf0*) (batch-size 5) (graph-min 0) (graph-max 1) (graph-step 0.01))
  (dotimes (image images)
					; set up filename
    (let ((filename (concatenate 'string
				 (do ((filename path (concatenate 'string filename "0")) (image-index (+ 1 image) (* image-index 10))) ((>= image-index (expt 10 (floor (log images 10)))) filename))
				 (write-to-string (+ 1 image)) ".png")))
					; train the network
      (train-on-generated-data network cost-function descent-rate batch-size train-per-img original-fun graph-min graph-max)
					; call draw-one
      (draw-one title network original-fun :min graph-min :max graph-max :step graph-step)
					; add text showing how many examples the network has been trained on so far
      (vgplot:text 0.02 0.97 (concatenate 'string "Examples trained on: " (write-to-string (* image train-per-img batch-size)))) 
					; save plot under filename
      (vgplot:print-plot (pathname filename))
					; close plot
      (vgplot:close-plot)
      (print filename))))

(defun plot-sampler (network original-fun starting-batches factor-size &key (descent-rate -0.002) (cost-function *cf0*) (batch-size 5) (graph-min 0) (graph-max 1) (graph-step 0.01) (y-min 0) (y-max 1))
  (do ((response "a" (progn (format *query-io* "Type q to quit. ") (force-output *query-io*) (read-line *query-io*))) (next-batches starting-batches (* next-batches factor-size)) (total-batches starting-batches (+ total-batches (* next-batches factor-size)))) ((equal response "q"))
    (vgplot:close-plot)
    (train-on-generated-data network cost-function descent-rate batch-size next-batches original-fun graph-min graph-max)
    (format t "Plotting graph after ~a batches. The network has been trained on ~a batches in total. " next-batches total-batches)
    (draw-one "Fun fact: you're gay." network original-fun :min graph-min :max graph-max :step graph-step :y-min y-min :y-max y-max))
  (vgplot:close-all-plots))
