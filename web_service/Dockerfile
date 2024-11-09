FROM python:3.12-bookworm

WORKDIR /app

RUN pip install pipenv

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --deploy --system

COPY ["predict.py", "final_model.bin", "./"]

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind", "0.0.0.0:9696", "predict:app" ]