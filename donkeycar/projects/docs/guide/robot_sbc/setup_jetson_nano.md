# Get Your Jetson Nano Working

![donkey](/assets/logos/nvidia_logo.png)

* [Step 1: Flash Operating System](#step-1-flash-operating-system)
* [Step 2: Install Dependencies](#step-2-install-dependencies)
* [Step 3: Setup Virtual Env](#step-3-setup-virtual-env)
* [Step 4: Install Donkeycar Python Code](#step-4-install-donkeycar-python-code)
* Then [Create your Donkeycar Application](/guide/create_application/)

## Step 1: Flash Operating System

Visit the official [Nvidia Jetson Nano Getting Started Guide](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#prepare). Work through the __Prepare for Setup__, __Writing Image to the microSD Card__, and __Setup and First Boot__ instructions, then return here.

## Step 2: Install Dependencies

ssh into your vehicle. Use the the terminal for Ubuntu or Mac. [Putty](https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html) for windows.

```bash
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install build-essential python3 python3-dev python3-pip python3-pandas python3-opencv python3-h5py libhdf5-serial-dev hdf5-tools nano ntp
```

Optionally, you can install RPi.GPIO clone for Jetson Nano from [here](https://github.com/NVIDIA/jetson-gpio). This is not required for default setup, but can be useful if using LED or other GPIO driven devices.

##  Step 3: Setup Virtual Env

```bash
pip3 install virtualenv
python3 -m virtualenv -p python3 env --system-site-packages
echo "source env/bin/activate" >> ~/.bashrc
source ~/.bashrc
```

##  Step 4: Install Donkeycar Python Code

* Change to a dir you would like to use as the head of your projects.

```bash
mkdir -p ~/projects; cd ~/projects
```

* Get the latest donkeycar from Github.

```bash
git clone https://github.com/autorope/donkeycar
cd donkeycar
git checkout master
pip install -e .[nano]
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v42 tensorflow-gpu==1.13.1+nv19.3
```

Note: This last command can take some time to compile grpcio.

##  Step 5: Install PyGame (Optional)

If you plan to use a USB camera, you will also want to setup pygame:

```bash
sudo apt-get install python-dev libsdl1.2-dev libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev libsdl1.2-dev libsmpeg-dev python-numpy subversion libportmidi-dev ffmpeg libswscale-dev libavformat-dev libavcodec-dev libfreetype6-dev

pip install pygame

```

Later on you can add the `CAMERA_TYPE="WEBCAM"` in myconfig.py.

----

### Next, [create your Donkeycar application](/guide/create_application/).
