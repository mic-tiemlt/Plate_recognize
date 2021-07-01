# Plate_recognize
Plate OCR
1, cd plate_recognize
2, download file .zip https://drive.google.com/file/d/1bBDa0Ig99-9tw8RTJlWCEz1Wkhw8SvbH/view?usp=sharing
3, unzip file downloaded
4, install requirement: (dowload and install anaconda)
    # Tensorflow CPU
    conda env create -f conda-cpu.yml
    conda activate yolov4-cpu

    # Tensorflow GPU
    conda env create -f conda-gpu.yml
    conda activate yolov4-gpu
5, run in terminal: 
  python detect_plate.py --weights ./checkpoints/custom-416 --size 416 --model yolov4 --video ./data/video/test2.mp4
