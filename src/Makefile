
PYTHONC = ../../miniconda3/envs/carnd-term1/bin/python
PYTHONG = LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64 ../../miniconda3/envs/carnd-term1-gpu/bin/python

UNAME = ${shell uname}
ifeq ($(UNAME),Darwin)
PYTHONC = ../../../../src/miniconda3/envs/carnd-term1/bin/python
PYTHONG = ../../../../src/miniconda3/envs/carnd-term1/bin/python
endif


all: 
	${PYTHONC} project_code04.py

run: 
	${PYTHONC} project_code04_1.py

calib:
	${PYTHONC} project_code01.py

test:
#	${PYTHONC} test02.py
#	${PYTHONC} test01.py

