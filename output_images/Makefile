
PYTHONC = ../../miniconda3/envs/carnd-term1/bin/python
PYTHONG = LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64 ../miniconda3/envs/carnd-term1-gpu/bin/python

UNAME = ${shell uname}
ifeq ($(UNAME),Darwin)
PYTHONC = ../../../../src/miniconda3/envs/carnd-term1/bin/python
PYTHONG = ../../../../src/miniconda3/envs/carnd-term1/bin/python
endif


all: 
	${PYTHONC} project_pipeline.py

run: 
	${PYTHONC} project_pipeline_video.py

calib:
	${PYTHONC} project_camera_caliblation.py
