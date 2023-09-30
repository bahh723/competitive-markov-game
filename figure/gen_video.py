import os

os.chdir(".")
os.system("ffmpeg -framerate 8 -i tmp/%03d.png -r 30 -pix_fmt yuv420p tmp/video.mp4")
