# TensorFlow Convolutional Neural Network TFRecords Based

Format Dataset Images
```
find . -name '*.png' -execdir mogrify -format jpg -resize 352x560! -quality 100 -type Grayscale {} \;
```