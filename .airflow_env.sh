export AIRFLOW_HOME=/home/david/coin/.airflow
export AIRFLOW__CORE__DAGS_FOLDER=/home/david/coin/dags
export AIRFLOW__CORE__LOAD_EXAMPLES=False

# NOT 8080: on the Windows side a `netsh interface portproxy` rule has IP Helper
# holding 0.0.0.0:8080 (forwarding to a WSL IP from an older boot), which stops
# wslrelay.exe from binding it -- so localhost:8080 never reaches this VM. 8090 is
# free on both sides. Overridable: the makefile's `airflow` target passes its own.
export AIRFLOW__WEBSERVER__WEB_SERVER_PORT=${AIRFLOW__WEBSERVER__WEB_SERVER_PORT:-8090}

source /home/david/coin/.venv-airflow/bin/activate
