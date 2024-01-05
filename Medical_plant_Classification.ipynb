{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/abin0001/Anveshion_project/blob/master/Medical_plant_Classification.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "JZc6IhsamhyB"
      },
      "source": [
        "\n",
        "# **Medicinal Plant Classification**\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "qms-lsqMzTGy"
      },
      "outputs": [],
      "source": [
        "# !nvidia-smi"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ac75Dlj0yAax",
        "outputId": "2f642217-c01b-43e6-c760-596296225ccf"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Drive already mounted at /content/drive/; to attempt to forcibly remount, call drive.mount(\"/content/drive/\", force_remount=True).\n"
          ]
        }
      ],
      "source": [
        "from google.colab import drive\n",
        "drive.mount(\"/content/drive/\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Ou1akV3uFq1p"
      },
      "source": [
        "### **Importing Required Packages**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "gkDSVdXD_gqu"
      },
      "outputs": [],
      "source": [
        "# def module():\n",
        "#Importing the Libraries.\n",
        "import os  #To access file directory.\n",
        "import cv2 # To handle the images.\n",
        "import matplotlib.pyplot as plt # To visualise the data.\n",
        "import pandas as pd # To handle the dataframesl Performance.\n",
        "from ast import literal_eval # To Evaluate string  fro the arrays.\n",
        "import pickle #To create a image dataset.\n",
        "import numpy as np # To handle with arrays.\n",
        "from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input # Pretrained Model and Input processor.\n",
        "from tensorflow.keras import layers, models, optimizers # For NueralNetworks\n",
        "from tensorflow.keras.callbacks import TensorBoard  #To analyse the mode\n",
        "from sklearn.model_selection import train_test_split #To split the data for train and d test.\n",
        "from sklearn.preprocessing import LabelEncoder,StandardScaler #\n",
        "from tensorflow.keras.utils import to_categorical #"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "tTXT-rmaTcWp"
      },
      "source": [
        "### **Creating our own dataset**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "g8hTCqN0iCDs"
      },
      "outputs": [],
      "source": [
        "path = \"/content/drive/MyDrive/Anveshon project/Plants Images/cropped images Resized Images\"\n",
        "# trg_path=\"/content/drive/MyDrive/Anveshon project/Plants Images\"\n",
        "\n",
        "class CreateImageDataset:\n",
        "    def __init__(self, dir_path):\n",
        "        # module()\n",
        "        self.img_ = []\n",
        "        self.label_= []\n",
        "        self.dir_path = dir_path\n",
        "        self.class_list = os.listdir(self.dir_path)\n",
        "\n",
        "    def load_data(self):\n",
        "        for self.class_folder in self.class_list:\n",
        "            self.class_path = os.path.join(self.dir_path, self.class_folder)\n",
        "            if os.path.isdir(self.class_path):\n",
        "                for self.img_filename in os.listdir(self.class_path):\n",
        "                    self.img_path = os.path.join(self.class_path, self.img_filename)\n",
        "                    self.img = cv2.imread(self.img_path)\n",
        "                    if self.img.shape != (768, 576, 3):\n",
        "                      self.img = cv2.resize(self.img,(576,768))\n",
        "                    else:\n",
        "                      self.img = self.img\n",
        "                    self.label_.append(self.class_folder)\n",
        "                    self.img_.append(self.img)\n",
        "        self.X=np.array(self.img_)\n",
        "\n",
        "        return self.X,self.label_\n",
        "\n",
        "\n",
        "# if __name__ == \"__main__\":\n",
        "#     source_folder = path\n",
        "#     dataset_creator = CreateImageDataset(source_folder)\n",
        "#     x,y=dataset_creator.load_data()\n",
        "# print(x.shape)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "s0vcDEG87PXK"
      },
      "outputs": [],
      "source": [
        "source_folder = path\n",
        "dataset_creator = CreateImageDataset(source_folder)\n",
        "x,y=dataset_creator.load_data()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "WUBCQeM2yfiI"
      },
      "outputs": [],
      "source": [
        "labelencode=LabelEncoder()\n",
        "y=labelencode.fit_transform(y)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "BRmpsn-t00xY",
        "outputId": "fd89a250-b147-4529-d51e-74ed5432f4e1"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "array([6, 6, 6, ..., 2, 2, 2])"
            ]
          },
          "execution_count": 9,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "y"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "wugY8JLnywAv"
      },
      "outputs": [],
      "source": [
        "# y=np.array([y])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "rhuIpCYF0VVc"
      },
      "outputs": [],
      "source": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "egLtsj1mjGtZ"
      },
      "outputs": [],
      "source": [
        "# img_list=[]\n",
        "\n",
        "# for i in range(len(x)):\n",
        "#   img=x[i]\n",
        "#   # print(img.shape)\n",
        "#   img_list.append(img.flatten())\n",
        "\n",
        "# img_arry=np.array(img_list)\n",
        "\n",
        "# data=pd.DataFrame({\n",
        "#     'Image':img_list,\n",
        "#     'label':y\n",
        "# })\n",
        "\n",
        "# data.head()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "r1F45ybhoZDz"
      },
      "outputs": [],
      "source": [
        "# data.info()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "sr9g55r6olaa"
      },
      "outputs": [],
      "source": [
        "# x=np.array(data['Image'])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "hESou6HPk-Op"
      },
      "outputs": [],
      "source": [
        "# img_list"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "EuL1ph4mlGFd"
      },
      "outputs": [],
      "source": [
        "# img_sam=img_list[0].reshape((768, 576, 3))\n",
        "# img_arr_sam=img_arry[0].reshape((768, 576, 3))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "9Do-xTfMl3Dy"
      },
      "outputs": [],
      "source": [
        "# plt.imshow(img_sam)\n",
        "# plt.show()\n",
        "# plt.imshow(img_arr_sam)\n",
        "# plt.show()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "I2NWiAfbcXh_"
      },
      "outputs": [],
      "source": [
        "x,_,y,_=train_test_split(x,y,test_size=0.9,random_state=42)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "LkVyUf524Slh"
      },
      "outputs": [],
      "source": [
        "x,_,y,_=train_test_split(x,y,test_size=0.9,random_state=42)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "la0J8wBYqw1V"
      },
      "outputs": [],
      "source": [
        "x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)\n",
        "\n",
        "\n",
        "x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,test_size=0.2,random_state=42)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "ltffwHBS2287"
      },
      "outputs": [],
      "source": [
        "from keras.utils import to_categorical\n",
        "\n",
        "y_train= to_categorical(y_train, num_classes=9)\n",
        "y_val = to_categorical(y_val, num_classes=9)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "uT1bHA6CtSHE",
        "outputId": "fee2dbd9-27d4-49d1-d326-7fefea73979a"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "(20, 768, 576, 3)"
            ]
          },
          "execution_count": 21,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "x_train.shape"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "2clq6GC1uKsy",
        "outputId": "b37e26c3-c24e-458c-fdbe-b140b20a09db"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "(11, 768, 576, 3)"
            ]
          },
          "execution_count": 22,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "x_test.shape"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Pzwkkq8Z3VQC",
        "outputId": "7a956a22-ba68-4e0d-c5b3-c2698d3a948c"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "(5, 768, 576, 3)"
            ]
          },
          "execution_count": 23,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "x_val.shape"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "8YLqVkIjLhN6"
      },
      "outputs": [],
      "source": [
        "# plt.rcParams[\"figure.figsize\"] = (10,10)\n",
        "# fig,axes = plt.subplots(10,10)\n",
        "# for i,ax in enumerate(axes.flatten()):\n",
        "#   ax.imshow(x_train[i])\n",
        "#   ax.set_title(f'{y_train[i]}')\n",
        "#   ax.set_xticks([])\n",
        "#   ax.set_yticks([])\n",
        "# fig.tight_layout()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "aRHuuUYfLllj"
      },
      "outputs": [],
      "source": [
        "# scaler = StandardScaler()\n",
        "# x_train=scaler.fit_transform(x_train)\n",
        "# x_test=scaler.fit_transform(x_test)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "abeMQ6XRvAV2"
      },
      "outputs": [],
      "source": [
        "x_train=x_train/255.0"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "fUGTS0r6wDPB"
      },
      "outputs": [],
      "source": [
        "# print(x_train)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "C2W-x7nVvqUN"
      },
      "outputs": [],
      "source": [
        "x_test=x_test/255.0"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "IY9OLvrhvvON"
      },
      "outputs": [],
      "source": [
        "x_val=x_val/255.0"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "D8iRLCj06YTx"
      },
      "outputs": [],
      "source": [
        "train_x_preprocessed = preprocess_input(x_train)\n",
        "val_x_preprocessed = preprocess_input(x_val)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "KjsU5jjHwaHq",
        "outputId": "39e3e114-a002-402b-fa7b-dd38ec0378f9"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "(20, 768, 576, 3)"
            ]
          },
          "execution_count": 31,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "train_x_preprocessed.shape"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Cf9D1py5z-hz",
        "outputId": "2f284a12-0cc9-479e-9f1f-ad41d49510eb"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "(5, 768, 576, 3)"
            ]
          },
          "execution_count": 32,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "val_x_preprocessed.shape"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Symg5iCh0CPI",
        "outputId": "0aa84c44-72c5-4da2-95e0-000a3ff2646d"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "[[0. 0. 0. 0. 0. 0. 1. 0. 0.]\n",
            " [0. 0. 0. 1. 0. 0. 0. 0. 0.]\n",
            " [0. 0. 0. 0. 0. 0. 1. 0. 0.]\n",
            " [0. 0. 0. 1. 0. 0. 0. 0. 0.]\n",
            " [0. 0. 0. 0. 1. 0. 0. 0. 0.]\n",
            " [0. 0. 1. 0. 0. 0. 0. 0. 0.]\n",
            " [0. 0. 0. 0. 1. 0. 0. 0. 0.]\n",
            " [0. 0. 0. 0. 0. 0. 0. 1. 0.]\n",
            " [0. 0. 0. 0. 0. 1. 0. 0. 0.]\n",
            " [0. 1. 0. 0. 0. 0. 0. 0. 0.]\n",
            " [0. 0. 0. 0. 0. 0. 0. 0. 1.]\n",
            " [0. 0. 0. 0. 0. 1. 0. 0. 0.]\n",
            " [0. 1. 0. 0. 0. 0. 0. 0. 0.]\n",
            " [0. 0. 0. 0. 1. 0. 0. 0. 0.]\n",
            " [1. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
            " [1. 0. 0. 0. 0. 0. 0. 0. 0.]\n",
            " [0. 0. 0. 0. 0. 0. 1. 0. 0.]\n",
            " [0. 1. 0. 0. 0. 0. 0. 0. 0.]\n",
            " [0. 0. 0. 0. 0. 0. 0. 1. 0.]\n",
            " [0. 0. 0. 0. 0. 0. 0. 1. 0.]]\n"
          ]
        }
      ],
      "source": [
        "y_train.shape\n",
        "\n",
        "print(y_train)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "9U3h6Izs_Klb"
      },
      "source": [
        "### **Building Vgg16 model**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "lYsfBoP9Rj4u",
        "outputId": "f8a386eb-9e46-4e60-c2a1-dc6f3462c4fa"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Model: \"vgg16\"\n",
            "_________________________________________________________________\n",
            " Layer (type)                Output Shape              Param #   \n",
            "=================================================================\n",
            " input_1 (InputLayer)        [(None, 768, 576, 3)]     0         \n",
            "                                                                 \n",
            " block1_conv1 (Conv2D)       (None, 768, 576, 64)      1792      \n",
            "                                                                 \n",
            " block1_conv2 (Conv2D)       (None, 768, 576, 64)      36928     \n",
            "                                                                 \n",
            " block1_pool (MaxPooling2D)  (None, 384, 288, 64)      0         \n",
            "                                                                 \n",
            " block2_conv1 (Conv2D)       (None, 384, 288, 128)     73856     \n",
            "                                                                 \n",
            " block2_conv2 (Conv2D)       (None, 384, 288, 128)     147584    \n",
            "                                                                 \n",
            " block2_pool (MaxPooling2D)  (None, 192, 144, 128)     0         \n",
            "                                                                 \n",
            " block3_conv1 (Conv2D)       (None, 192, 144, 256)     295168    \n",
            "                                                                 \n",
            " block3_conv2 (Conv2D)       (None, 192, 144, 256)     590080    \n",
            "                                                                 \n",
            " block3_conv3 (Conv2D)       (None, 192, 144, 256)     590080    \n",
            "                                                                 \n",
            " block3_pool (MaxPooling2D)  (None, 96, 72, 256)       0         \n",
            "                                                                 \n",
            " block4_conv1 (Conv2D)       (None, 96, 72, 512)       1180160   \n",
            "                                                                 \n",
            " block4_conv2 (Conv2D)       (None, 96, 72, 512)       2359808   \n",
            "                                                                 \n",
            " block4_conv3 (Conv2D)       (None, 96, 72, 512)       2359808   \n",
            "                                                                 \n",
            " block4_pool (MaxPooling2D)  (None, 48, 36, 512)       0         \n",
            "                                                                 \n",
            " block5_conv1 (Conv2D)       (None, 48, 36, 512)       2359808   \n",
            "                                                                 \n",
            " block5_conv2 (Conv2D)       (None, 48, 36, 512)       2359808   \n",
            "                                                                 \n",
            " block5_conv3 (Conv2D)       (None, 48, 36, 512)       2359808   \n",
            "                                                                 \n",
            " block5_pool (MaxPooling2D)  (None, 24, 18, 512)       0         \n",
            "                                                                 \n",
            "=================================================================\n",
            "Total params: 14714688 (56.13 MB)\n",
            "Trainable params: 14714688 (56.13 MB)\n",
            "Non-trainable params: 0 (0.00 Byte)\n",
            "_________________________________________________________________\n"
          ]
        }
      ],
      "source": [
        "Vgg16_base_mdl = VGG16(weights='imagenet', include_top=False, input_shape=(x_train[0].shape),classes=9)\n",
        "Vgg16_base_mdl.summary()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "zFeWKkl8Wv4F"
      },
      "outputs": [],
      "source": [
        "#changing the model in to non_trainable.\n",
        "for layer in Vgg16_base_mdl.layers:\n",
        "    layer.trainable = False"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "8LD0SRCTXhV2",
        "outputId": "8b5913ad-8d02-4949-d253-829cee44f759"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Model: \"vgg16\"\n",
            "_________________________________________________________________\n",
            " Layer (type)                Output Shape              Param #   \n",
            "=================================================================\n",
            " input_1 (InputLayer)        [(None, 768, 576, 3)]     0         \n",
            "                                                                 \n",
            " block1_conv1 (Conv2D)       (None, 768, 576, 64)      1792      \n",
            "                                                                 \n",
            " block1_conv2 (Conv2D)       (None, 768, 576, 64)      36928     \n",
            "                                                                 \n",
            " block1_pool (MaxPooling2D)  (None, 384, 288, 64)      0         \n",
            "                                                                 \n",
            " block2_conv1 (Conv2D)       (None, 384, 288, 128)     73856     \n",
            "                                                                 \n",
            " block2_conv2 (Conv2D)       (None, 384, 288, 128)     147584    \n",
            "                                                                 \n",
            " block2_pool (MaxPooling2D)  (None, 192, 144, 128)     0         \n",
            "                                                                 \n",
            " block3_conv1 (Conv2D)       (None, 192, 144, 256)     295168    \n",
            "                                                                 \n",
            " block3_conv2 (Conv2D)       (None, 192, 144, 256)     590080    \n",
            "                                                                 \n",
            " block3_conv3 (Conv2D)       (None, 192, 144, 256)     590080    \n",
            "                                                                 \n",
            " block3_pool (MaxPooling2D)  (None, 96, 72, 256)       0         \n",
            "                                                                 \n",
            " block4_conv1 (Conv2D)       (None, 96, 72, 512)       1180160   \n",
            "                                                                 \n",
            " block4_conv2 (Conv2D)       (None, 96, 72, 512)       2359808   \n",
            "                                                                 \n",
            " block4_conv3 (Conv2D)       (None, 96, 72, 512)       2359808   \n",
            "                                                                 \n",
            " block4_pool (MaxPooling2D)  (None, 48, 36, 512)       0         \n",
            "                                                                 \n",
            " block5_conv1 (Conv2D)       (None, 48, 36, 512)       2359808   \n",
            "                                                                 \n",
            " block5_conv2 (Conv2D)       (None, 48, 36, 512)       2359808   \n",
            "                                                                 \n",
            " block5_conv3 (Conv2D)       (None, 48, 36, 512)       2359808   \n",
            "                                                                 \n",
            " block5_pool (MaxPooling2D)  (None, 24, 18, 512)       0         \n",
            "                                                                 \n",
            "=================================================================\n",
            "Total params: 14714688 (56.13 MB)\n",
            "Trainable params: 0 (0.00 Byte)\n",
            "Non-trainable params: 14714688 (56.13 MB)\n",
            "_________________________________________________________________\n"
          ]
        }
      ],
      "source": [
        "#model summary after Change non_trainable\n",
        "Vgg16_base_mdl.summary()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "lYUs4CC1Xsr_"
      },
      "source": [
        "### **Creating our Neural network**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "56QZRIi5XqV8"
      },
      "outputs": [],
      "source": [
        "X = layers.Flatten()(Vgg16_base_mdl.output)\n",
        "X = layers.Dense(256,activation='relu')(X)\n",
        "X = layers.Dropout(0.5)(X)\n",
        "predictions = layers.Dense(9,activation='softmax')(X)\n",
        "model=models.Model(inputs=Vgg16_base_mdl.input,outputs=predictions)\n",
        "model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "background_save": true,
          "base_uri": "https://localhost:8080/"
        },
        "id": "BfkNtp57uxbv",
        "outputId": "ff40ee02-46e8-421c-81a1-2d250ab72182"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Epoch 1/10\n",
            "1/1 [==============================] - 126s 126s/step - loss: 3.4251 - accuracy: 0.1000 - val_loss: 19.4061 - val_accuracy: 0.0000e+00\n",
            "Epoch 2/10\n",
            "1/1 [==============================] - 140s 140s/step - loss: 20.6954 - accuracy: 0.2000 - val_loss: 19.3096 - val_accuracy: 0.0000e+00\n",
            "Epoch 3/10\n",
            "1/1 [==============================] - 140s 140s/step - loss: 23.8693 - accuracy: 0.1000 - val_loss: 10.2647 - val_accuracy: 0.0000e+00\n",
            "Epoch 4/10\n",
            "1/1 [==============================] - 137s 137s/step - loss: 17.3789 - accuracy: 0.1500 - val_loss: 6.5547 - val_accuracy: 0.2000\n",
            "Epoch 5/10\n",
            "1/1 [==============================] - 139s 139s/step - loss: 13.6076 - accuracy: 0.2000 - val_loss: 3.8559 - val_accuracy: 0.2000\n",
            "Epoch 6/10\n",
            "1/1 [==============================] - 138s 138s/step - loss: 12.0674 - accuracy: 0.1500 - val_loss: 3.7968 - val_accuracy: 0.2000\n",
            "Epoch 7/10\n",
            "1/1 [==============================] - 121s 121s/step - loss: 8.5790 - accuracy: 0.1000 - val_loss: 3.7957 - val_accuracy: 0.0000e+00\n",
            "Epoch 8/10\n",
            "1/1 [==============================] - 137s 137s/step - loss: 5.2064 - accuracy: 0.3000 - val_loss: 3.9549 - val_accuracy: 0.0000e+00\n",
            "Epoch 9/10\n",
            "1/1 [==============================] - 138s 138s/step - loss: 4.6556 - accuracy: 0.1500 - val_loss: 4.0044 - val_accuracy: 0.0000e+00\n",
            "Epoch 10/10\n",
            "1/1 [==============================] - 139s 139s/step - loss: 4.4850 - accuracy: 0.1000 - val_loss: 3.3161 - val_accuracy: 0.0000e+00\n"
          ]
        },
        {
          "data": {
            "text/plain": [
              "<keras.src.callbacks.History at 0x79aa146eae30>"
            ]
          },
          "execution_count": 38,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "model.fit(train_x_preprocessed,y_train,epochs=10,validation_data=(val_x_preprocessed,y_val))"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "model.save('model.h5')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "key_PpJw-n42",
        "outputId": "09dfc26b-e3c2-4c61-85e4-42238b8b977e"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.10/dist-packages/keras/src/engine/training.py:3103: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.\n",
            "  saving_api.save_model(\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "KERAS_MODEL_PATH = 'tf_keras_c2f.tf'"
      ],
      "metadata": {
        "id": "olfpXWle_RI1"
      },
      "execution_count": 41,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model.save(KERAS_MODEL_PATH, save_format='tf')"
      ],
      "metadata": {
        "id": "6u6u5EwJ_Ub_"
      },
      "execution_count": 42,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import tensorflow as tf"
      ],
      "metadata": {
        "id": "yoM3bvLG_XLh"
      },
      "execution_count": 40,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "converter = tf.lite.TFLiteConverter.from_saved_model(KERAS_MODEL_PATH)\n",
        "tflite_model = converter.convert()\n",
        "\n",
        "# Save the TFLITE model\n",
        "with open('modelc2f.tflite', 'wb') as f:\n",
        "    f.write(tflite_model)"
      ],
      "metadata": {
        "id": "kH6JnoDI_WYW"
      },
      "execution_count": 43,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "y_pred=model.predict(x_test)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "WYUOEMDx_zpu",
        "outputId": "126be7e3-dd50-4974-92e1-97f7907b3a00"
      },
      "execution_count": 44,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "1/1 [==============================] - 60s 60s/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "y_test"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "DEOBTEFR_7rM",
        "outputId": "3a6a1f6d-451c-458f-ba6c-dad6de3d29be"
      },
      "execution_count": 45,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([6, 8, 5, 7, 4, 3, 7, 2, 7, 4, 1])"
            ]
          },
          "metadata": {},
          "execution_count": 45
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "y_pred"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "vkeAPD3TAKbF",
        "outputId": "e7507a2f-f38e-4083-91a8-4da1b4a01476"
      },
      "execution_count": 46,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[6.5613375e-03, 7.2392452e-01, 1.0669976e-03, 2.7330578e-03,\n",
              "        3.4715398e-04, 5.7461131e-03, 1.0525974e-03, 2.5633082e-01,\n",
              "        2.2372771e-03],\n",
              "       [1.2756873e-02, 6.9391614e-01, 2.7782014e-03, 3.8121317e-03,\n",
              "        9.1168453e-04, 9.3949093e-03, 2.4287894e-03, 2.7055115e-01,\n",
              "        3.4499965e-03],\n",
              "       [9.9177174e-03, 7.0803130e-01, 3.8819911e-03, 6.3894140e-03,\n",
              "        1.4291478e-03, 1.5745172e-02, 4.1584643e-03, 2.4491586e-01,\n",
              "        5.5309539e-03],\n",
              "       [5.9166215e-03, 6.7749679e-01, 9.4959402e-04, 2.3455522e-03,\n",
              "        4.1240282e-04, 3.5946188e-03, 9.7494101e-04, 3.0652848e-01,\n",
              "        1.7808912e-03],\n",
              "       [7.0814150e-03, 5.5693591e-01, 1.3585713e-03, 2.2833683e-03,\n",
              "        5.3915841e-04, 5.4947236e-03, 1.1372131e-03, 4.2316538e-01,\n",
              "        2.0041415e-03],\n",
              "       [1.1913427e-02, 7.4215901e-01, 4.0325169e-03, 4.2332420e-03,\n",
              "        1.3073264e-03, 1.2640350e-02, 3.8435161e-03, 2.1337879e-01,\n",
              "        6.4918343e-03],\n",
              "       [9.9967131e-03, 7.3222595e-01, 2.8583598e-03, 4.3246262e-03,\n",
              "        1.0679006e-03, 1.2878864e-02, 3.0795773e-03, 2.2936894e-01,\n",
              "        4.1991095e-03],\n",
              "       [9.1381567e-03, 6.6895407e-01, 2.3643705e-03, 3.7465168e-03,\n",
              "        8.9980569e-04, 1.0844340e-02, 2.5829254e-03, 2.9884213e-01,\n",
              "        2.6277814e-03],\n",
              "       [8.5847490e-03, 6.3709861e-01, 2.4311927e-03, 3.2913163e-03,\n",
              "        5.8154884e-04, 8.7485844e-03, 3.1909430e-03, 3.3175731e-01,\n",
              "        4.3156375e-03],\n",
              "       [7.8266216e-03, 6.9315761e-01, 1.4540753e-03, 1.8255861e-03,\n",
              "        7.3282060e-04, 8.9512011e-03, 1.0170859e-03, 2.8140175e-01,\n",
              "        3.6332351e-03],\n",
              "       [7.1254866e-03, 7.1880263e-01, 2.3676301e-03, 2.8187127e-03,\n",
              "        8.1451191e-04, 6.4157001e-03, 1.4077143e-03, 2.5732499e-01,\n",
              "        2.9225680e-03]], dtype=float32)"
            ]
          },
          "metadata": {},
          "execution_count": 46
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "np.argmax(y_pred,axis=1)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "PZMqigzFA9nd",
        "outputId": "59afa9f4-3d84-415b-f041-94a278114566"
      },
      "execution_count": 54,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])"
            ]
          },
          "metadata": {},
          "execution_count": 54
        }
      ]
    }
  ],
  "metadata": {
    "colab": {
      "provenance": [],
      "mount_file_id": "1UDVgbqqgrWf48mVS0ok2OIc8BsImLQxk",
      "authorship_tag": "ABX9TyNhg1P5W2/H7/Fp1uSYxdDI",
      "include_colab_link": true
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}