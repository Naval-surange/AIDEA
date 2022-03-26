import pandas as pd
import dask
from io import StringIO 
import csv
import dask.dataframe as dd

class dataloader:
    def load_data(method,aws_bucket=None,access_key=None,secret_access=None,session_token=None,file_name=None):

        if method=="aws":
            AWS_S3_BUCKET = aws_bucket
            AWS_ACCESS_KEY_ID = access_key
            AWS_SECRET_ACCESS_KEY = secret_access
            AWS_SESSION_TOKEN = session_token

            key = f"files/{file_name}"

            df = pd.read_csv(
                        f"s3://{AWS_S3_BUCKET}/{key}",
                        storage_options={
                            "key": AWS_ACCESS_KEY_ID,
                            "secret": AWS_SECRET_ACCESS_KEY,
                            "token": AWS_SESSION_TOKEN,
                        },
                    )

        elif method=="upload":
            if file_name.name.endswith("csv"):
                df = pd.read_csv(file_name,header=0)
            elif file_name.name.endswith("tsv"):
                df = pd.read_csv(file_name,sep="\t",header=0)
            elif file_name.name.endswith("xls") or file_name.name.endswith("xlsx"):
                df = pd.read_excel(file_name,header=0)

        elif method=="url":
            if file_name.endswith("csv") or file_name.endswith("txt"):
                df = dd.read_csv(str(file_name),error_bad_lines=False,quoting=csv.QUOTE_NONE, blocksize=None).compute()
            elif file_name.endswith("tsv"):
                df = dd.read_csv(str(file_name),sep="\t",error_bad_lines=False,quoting=csv.QUOTE_NONE).compute()
            elif file_name.endswith("xls") or file_name.endswith("xlsx"):
                parts = dask.delayed(pd.read_excel)(str(file_name), sheet_name=0)
                df = dd.from_delayed(parts).compute()



        return df