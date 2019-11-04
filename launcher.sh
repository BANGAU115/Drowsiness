#!/bin/sh
# launcher.sh
# navigate to home directory, then to this directory, then execute python script, then back home
#sudo xauth add aml/unix:0  MIT-MAGIC-COOKIE-1  e4eb9c8f3cdd42c166fd5cd35744dbe7
#xhost +localhost
#export ICETHORITY=/home/bang/.ICEauthority
export DISPLAY=:0.0 
sudo mount /dev/mmcblk0p1 /home/bang/Bangau/sdcard

cd /
cd home/bang/Bangau
#sudo python3 Cam1.py & sudo python3 Cam2.py 
#sudo python3 camera.py & sudo python3 Drowsiness_USB.py
sudo python3 Dr.py & sudo python3 Cam.py 
cd /
