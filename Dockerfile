FROM python:3.7

WORKDIR /AIDEA 

ADD ./requirements.txt /AIDEA/requirements.txt
RUN pip3 install -r requirements.txt

ADD . .

EXPOSE 8501

CMD ["streamlit", "run", "dashboard.py"]