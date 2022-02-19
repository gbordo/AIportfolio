FROM python:3.9.7
COPY . ./
RUN pip3 install -r requirements.txt
CMD ["assign_4_bordoniqi.py"]
ENTRYPOINT ["python"]