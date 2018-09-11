FROM python:3
ARG params_file
ADD server.py .
RUN pip install requirements.txt
ENTRYPOINT ["python", "./server.py", $params_file]
