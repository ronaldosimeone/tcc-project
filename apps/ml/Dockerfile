FROM python:3.11

WORKDIR /app
COPY . .

RUN pip install jupyter pandas scikit-learn

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root"]