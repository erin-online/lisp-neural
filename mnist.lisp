(defun images-file-to-array (filename &optional (etype '(unsigned-byte 8)))
  "Takes in a file path and returns a 2D input array."
  (let ((filestream (open filename :element-type etype)))
                                        ; first two bytes are 0, third is data type
    (read-byte filestream)
    (read-byte filestream)
    (read-byte filestream)
    ; get dimension sizes
    (let ((dim-count (read-byte filestream)) (dim-sizes))
      (dotimes (dimension dim-count)
        (let ((dim-size 0))
          (dotimes (dim-byte 4)
            (setf (ldb (byte 8 (* 8 (- 3 dim-byte))) dim-size) (read-byte filestream)))
          (setf dim-sizes (append dim-sizes (list dim-size)))))
      ; data starts here
      ; flatten input (if using convnet strats then don't do this and rework your input arrays)
      (let ((data-array (make-array (list (car dim-sizes) (reduce #'* (cdr dim-sizes))))))
        ; read data
        (dotimes (img (car dim-sizes))
          (dotimes (data-byte (reduce #'* (cdr dim-sizes)))
            (setf (aref data-array img data-byte) (read-byte filestream))))
        data-array))))

(defun labels-file-to-array (filename &optional (etype '(unsigned-byte 8)) (slots 10))
  "Takes in a file path and returns a 2D labels array using one-hot encoding."
  (let ((filestream (open filename :element-type etype)))
    (read-byte filestream)
    (read-byte filestream)
    (read-byte filestream)
                                        ; one dimension. fuck you
    (read-byte filestream)
    (let ((dim-size 0))
      (dotimes (dim-byte 4)
        (setf (ldb (byte 8 (* 8 (- 3 dim-byte))) dim-size) (read-byte filestream)))
                                        ; data starts here
      (let ((data-array (make-array (list dim-size slots) :initial-element 0)))
                                        ; read data
        (dotimes (img dim-size)
          (setf (aref data-array img (read-byte filestream)) 1))
        data-array))))
