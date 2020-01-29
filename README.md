<h1 align="center"> ~ Rock - Paper - Scissor with Edge TPU ~ </h1>

Today, 28/01/2020, I really wanted to loose some time and explore a little bit these new edge AI devices made by [Google](https://coral.ai/).
In order to check their performance, I trained a very simple and dumb [CNN](https://github.com/EscVM/RPS_with_Edge_TPU/blob/master/media/baby_cnn_arch.png) (feel free to improve it) on the [Rock-Paper-Scissor](https://www.tensorflow.org/datasets/catalog/rock_paper_scissors) dataset and I made it run on a Coral Dev Board and a Raspberry 3 B+ with the USB Coral Accelerator. 

# 1.0 Getting Started

Clone this repository

   ```bash
   git clone https://github.com/EscVM/RPS_with_Edge_TPU
   ```
Python3 is required. I used TensorFlow 2.x for the training, but I uploaded also all converted and original weights. So, if you don't want to re-train the network you can simply use the inference code.

## 1.1 Installations for the hosting device

Install on the hosting device to make the inference code work the following libraries:

- [opencv-python](https://pypi.org/project/opencv-python/)
- numpy
- [TensorFlow Lite Interpreter](https://www.tensorflow.org/lite/guide/python). If you're using the Coral USB Accelerator with the Raspberry 3 B+ or 4 download ARM32
**N.B.** If you are using the Dev Board the Interpreter and also the TFLite Converter are already installed


# 2.0 Run the Interpreter
Open your terminal in the project folder and launch:

   ```bash
   python3 rpc_webcame.py
   ```
   
Enjoy the network predicting the shape of your beautifull hands :)

# 3.0 Improve the CNN Network 

As I already wrote in the introduction, I made this project very quickly to check the performance of my two Coral devices. So, I didn't spend time building a cool network. If you want to improve the CNN structure or using transfer learning to retrain your prefered architecture, in the project folder you can find the two jupyter notebook I used to **train** the network and **convert** from TensorFlow to TFLite. Then you have to use the [TPU compiler](https://coral.ai/docs/edgetpu/compiler/) to make your TFLite file TPU compatible. It's a long, but not difficult process. [Here](https://coral.ai/docs/edgetpu/models-intro/) you can find a beutifull summary of the entire chain.
