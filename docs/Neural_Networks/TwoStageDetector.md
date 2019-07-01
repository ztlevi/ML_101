# Two Stage Object Detector

Borrowed from [here](https://towardsdatascience.com/r-cnn-fast-r-cnn-faster-r-cnn-yolo-object-detection-algorithms-36d53571365e)

## R-CNN

### Selective Search

1. Generate initial sub-segmentation, we generate many candidate regions
2. Use greedy algorithm to recursively combine similar regions into larger ones
3. Use the generated regions to produce the final candidate region proposals

### Network Architecture

![img](../../assets/rcnn.png)

- At first, it uses selective search to generate about 2000 region proposals
- Region proposals are warped into a square and fed into a convolutional neural network that produces a 4096-dimensional feature vector as output.
- The CNN acts as a feature extractor and the output dense layer consists of the features extracted from the image and the extracted features are fed into an SVM to classify the presence of the object within that candidate region proposal.
- In addition to predicting the presence of an object within the region proposals, the algorithm also predicts four values which are offset values to increase the precision of the bounding box.

![rcnn2](../../assets/rcnn2.png)

### Problems with R-CNN

- It still takes a huge amount of time to train the network as you would have to classify 2000 region proposals per image.

- It cannot be implemented real time as it takes around 47 seconds for each test image.

- The selective search algorithm is a fixed algorithm. Therefore, no learning is happening at that stage. This could lead to the generation of bad candidate region proposals.

## Fast R-CNN

### ROI Pooling

The layer takes two inputs:

1. A fixed-size feature map obtained from a deep convolutional network with several convolutions and max pooling layers.
2. An N x 5 matrix of representing a list of regions of interest, where N is a number of RoIs. The first column represents the image index and the remaining four are the coordinates of the top left and bottom right corners of the region.

What does the RoI pooling actually do? For every region of interest from the input list, it takes a section of the input feature map that corresponds to it and scales it to some pre-defined size (e.g., 7×7). The scaling is done by:

1. Dividing the region proposal into equal-sized sections (the number of which is the same as the dimension of the output)
2. Finding the largest value in each section
3. Copying these max values to the output buffer

### Network Architecture

![fast rcnn](../../assets/fastrcnn.png)

- Instead of feeding the region proposals to the CNN, the author feeded the input image to the CNN to generate a convolutional feature map.
- From the convolutional feature map, the author identified the region of proposals and warp them into squares and by using a RoI pooling layer the author reshaped them into a fixed size so that it can be fed into a fully connected layer.
- From the RoI feature vector, the author used a softmax layer to predict the class of the proposed region and also the offset values for the bounding box.

### Why faster than R-CNN?

The reason “Fast R-CNN” is faster than R-CNN is because you don’t have to **feed 2000 region proposals to the convolutional neural network** every time. Instead, the convolution operation is done only once per image and a feature map is generated from it.

## Faster R-CNN

### Region Proposal Network

1. First, the picture goes through conv layers and feature maps are extracted.
2. Then a **sliding window** is used in RPN for each location over the feature map.
3. For each location, **k (k=9) anchor** boxes are used (**3 scales of 128, 256 and 512, and 3 aspect ratios of 1:1, 1:2, 2:1**) for generating region proposals.
4. A **cls** layer outputs $$2k$$ scores **whether there is object or not** for $$k$$ boxes.
5. A **reg** layer outputs $$4k$$ for the **coordinates** (box center coordinates, width and height) of _k_ boxes.
6. With a size of $$W \times H$$ feature map, there are $$WHk$$ anchors in total.

![rpn](../../assets/rpn.png)

### Network Architecture

![fast rcnn](../../assets/faster rcnn.png)

- Similar to Fast R-CNN, the image is provided as an input to a convolutional network which provides a convolutional feature map.
- Instead of using selective search algorithm on the feature map to identify the region proposals, a separate network is used to predict the region proposals(Region Proposal Network).

- The predicted region proposals are then reshaped using a RoI pooling layer which is then used to classify the image within the proposed region and predict the offset values for the bounding boxes.
