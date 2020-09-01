FROM python:3.7.8

WORKDIR /home
ADD . /home

RUN pip install tensorflow && \
    pip install numpy keras pillow tk

# win32gui not supported on linux


CMD ["python","/home/2.0_Zafrin_Dhali_HandwrittenDigit_GUITest.py"]
