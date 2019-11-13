# IoT Access Control

#### Project for access control through edge computing

In this project, I used a Convolutional Neural Network (CNN) to make a face recognition application. It runs on a Raspberry Pi and the faces are detect through Haar Cascade Classifiers. Since this system is meant to be used in remote areas, the communication system used LoRa.

This work is the Final Project for Control and Automation Engineering Bachelor Degree at Federal University of Itajubá (UNIFEI).

___
## Hardware requisites:

 - [Raspberry Pi](https://www.raspberrypi.org/)
	 - In this project, I used a RP 4 B+, but a 3 B+ will work as well. A 2 B+ may work but since we're gonna use a good amount of processing power, I would not recommend using it. 
 - USB Camera
 - [HAT Dragino GPS LoRa](https://www.dragino.com/products/lora/item/106-lora-gps-hat.html)
 - [WiFi LoRa 32 ESP32 OLED board](https://heltec.org/project/wifi-lora-32/)

## Dependencies:
This project was made on Ubuntu 16.04 LTS using a laptop Intel Core i5 8GB RAM and no GPU.
Considering you're using a Linux, you'll need:

 - Python 3.5
 - OpenCV python
 - Pytorch

> This section needs to be improved by letting all commands needed to install all dependencies.

On Raspberry Pi, I used the [Raspbian Buster Full](https://www.raspberrypi.org/downloads/raspbian/). You'll need:

 - Python 3.7
 - OpenCV python
 - Pytorch

> To save some time, I'll let the image I used available here very soon :)

## Contents:

### Host
Host has all the code I used to gather the training set data and generate the CNN.

### Raspberry
Raspberry has all the code that'll run on the edge device, which is the facial detection & recognition and LoRa communication.