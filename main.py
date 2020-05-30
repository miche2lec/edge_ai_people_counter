"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client

def draw_outputs(coords, image, w, h):
        '''
        TODO: This method needs to be completed by you
        '''
        x1=int(coords[0]*w)
        x2=int(coords[2]*w)
        y1=int(coords[1]*h)
        y2=int(coords[3]*h)
        image = cv2.rectangle(image,(x1, y1),(x2, y2),(0,255,0),3)
        return image


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    plugin = infer_network.load_model(args.model, args.device, args.cpu_extension)
    net_input_shape = infer_network.get_input_shape()
    

    ## TODO: Handle the input stream ###
    # Create a flag for single images
    image_flag = False
    # Check if the input is a webcam
    if args.input == 'CAM':
        args.input = 0
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp') or args.input.endswith('.png'):
        image_flag = True
    
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)
    
    # Create a video writer for the output video
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = 0
    duration = 0
    person_flag = False
    total_people_count = 0
    people_on_screen = 0
    no_person_time = 0


    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        

        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        ### TODO: Pre-process the image as needed ###
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
      
        ### TODO: Start asynchronous inference for specified request ###
        net_input = {'image_info': p_frame.shape[1:], 'image_tensor': p_frame}
        #infer_time_start = time.time()
        infer_network.exec_net(net_input)

        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:
            #infer_time_end = time.time()
            result = infer_network.get_output()
            
            
            ### TODO: Extract any desired stats from the results ###
            obj = result[0][0][0]
            if obj[2] > args.prob_threshold:
                frame = draw_outputs(obj[3:], frame, int(cap.get(3)), int(cap.get(4)))
                if not person_flag:
                    person_flag = True
                    people_on_screen += 1
                    total_people_count += people_on_screen
                num_frames += 1
                no_person_time = 0
            else:
                if person_flag:
                    no_person_time += 1
                if no_person_time > 3*int(fps):
                    person_flag = False
                    people_on_screen = 0
                    duration = num_frames/fps
                    no_person_time = 0
                    num_frames = 0
                    client.publish('person/duration',
                                   payload=json.dumps({'duration': duration}),
                                   qos=0, retain=False)
                    duration = 0

            ## TODO: Calculate and send relevant information on ###
            ## current_count, total_count and duration to the MQTT server ###
            ## Topic "person": keys of "count" and "total" ###
            ## Topic "person/duration": key of "duration" ###
            client.publish('person',
                           payload=json.dumps({
                               'count': people_on_screen, 'total': total_people_count}),
                           qos=0, retain=False)
                

                

        ### TODO: Send the frame to the FFMPEG server ###
        #frame = cv2.resize(frame, (768, 432))
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()

        ### TODO: Write an output image if `single_image_mode` ###
    cap.release()
    cv2.destroyAllWindows()


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
