FROM apache/airflow:2.6.1rc2
USER airflow
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --user -r /tmp/requirements.txt