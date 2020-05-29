# Project Write-Up

## Model Selection and Custom Layers

The model I selected is [Faster R-CNN with Inception v2](https://github.com/opencv/open_model_zoo/blob/master/models/public/faster_rcnn_inception_v2_coco/faster_rcnn_inception_v2_coco.md) from the Open Model Zoo.

To convert the model to IR, I used:

`python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json`

There were a few unsupported layers, so I added a cpu extension using:

`-l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so`


## Comparing Model Performance

The size of the model before conversion and after conversion has changed drastically. The pb file which stores the original Tensorflow model has a size of 55,815KB while the converted xml file is only 124KB. Before conversion, based on this [paper](https://arxiv.org/pdf/1506.01497v3.pdf), the model was able to achieve an mAP of 78.8% on the PASCAL VOC 2007 test set. Since we're optimizing the model using the Model Optimizer, it's likely that the accuracy has dropped, but the drop in accuracy should only be minimal. The [paper](https://arxiv.org/pdf/1506.01497v3.pdf) also stated that it had an inference speed of 200ms per image. Meanwhile, the converted model takes about 880ms to perform inference on a frame of the video, so it's significantly slower.

## Assess Model Use Cases

Some of the potential use cases of the people counter app are counting the total number of people that pass by in a given time frame, detecting when a certain number of people in the same area at a given time, and keeping track of the duration that people are present in an area.

Each of these use cases would be useful because the first use case can be used to access total number of customers in a given day at a store, the second use case can be used to track if people are following Covid-19 guidelines staying 6 feet apart, and the third use case can be used to, perhaps in a store, track the average amount of time customers spend at a particular shelf.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. This model is trained on a wide variety of objects and is relatively accurate. However, while performing inference on the test video, it cannot recognize the person until they move fully into the frame. Since this model is trained on a general training set, it the images in the training set are likely in pretty well-lit settings, making this model function best in natural lighting. Similarly, this model probably does best with close to frontal angles of people and image sizes close to 600x600. 

## App Results

The app works nearly as expected. The app successfully puts bounding boxes around the people once most of their upper body comes into the frame, counts the number of people present, and starts recording the direction of time they are in frame.

![img1](https://github.com/miche2lec/edge_ai_people_counter/blob/master/images/app1.png)
__First person that appears in the frame__.

However, there are a few issues with the app display, namely when duration is zero or higher than a certain threshold, the app displays invalid, and total counted does not actually seem to display the total variable. A mentor on Knowledge has remarked that perhaps total counted is displaying the number of requests made to the server.

By looking at the server outputs, it is clear that the correct values are being sent to the server. 

![img2](https://github.com/miche2lec/edge_ai_people_counter/blob/master/images/backend1.png)
__Server output while first person is in the frame__.

![img3](https://github.com/miche2lec/edge_ai_people_counter/blob/master/images/app2.png)
__First person leaving frame__.

![img4](https://github.com/miche2lec/edge_ai_people_counter/blob/master/images/app3.png)
__First person left frame__.

![img5](https://github.com/miche2lec/edge_ai_people_counter/blob/master/images/app5.png)
__Second person entered frame__.

![img6](https://github.com/miche2lec/edge_ai_people_counter/blob/master/images/backend3.png)
__Server output while second person is entering frame__.

Note the value of "total" increases correctly to the value of 2. Duration stops being sent until second person enters frame.

![img7](https://github.com/miche2lec/edge_ai_people_counter/blob/master/images/backend5.png)
__Server output while second person is in frame__.

![img9](https://github.com/miche2lec/edge_ai_people_counter/blob/master/images/app7.png)
__Third person is in frame__.

![img8](https://github.com/miche2lec/edge_ai_people_counter/blob/master/images/backend7.png)
__Server output while third person is in frame__.

At some point when duration > 100, the app starts outputing invalid date. When the third person enters the frame, the app continues to display invalid date, even though the server shows the duration resetting again from zero to count the number of seconds that the new person is in the frame. Unfortunately, editing how the app displays the inputs that are sent are outside of my hands.