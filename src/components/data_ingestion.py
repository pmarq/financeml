import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import yfinance as yf
from datetime import datetime
import numpy as np

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def getApiData(self):

        try: 

            #BALANCE#--------

            list_symbols = pd.read_excel(r'src\data\ibrx.xls')['Código'].values

            # Lista para armazenar os símbolos filtrados
            balance_annual_stocks = []

            # Iterar sobre cada símbolo de estoque
            for symbol in list_symbols:
                try:
                    # Obter os dados para o símbolo da ação atual
                    stock_data = yf.Ticker(symbol + '.SA')
                    
                    # Obter os dados do balanço patrimonial da ação
                    balance_sheet_data = stock_data.balancesheet
                    
                    # Verificar se todas as datas de 2023 a 2020 estão presentes nas colunas
                    if all(str(year) in balance_sheet_data.columns for year in range(2023, 2019, -1)):
                        balance_annual_stocks.append(symbol)
                        logging.info(f"Encontrados dados para {symbol} de 2023 a 2020:")                        
                        logging.info("="*50)
                
                except Exception as e:
                    logging.info(f"Erro ao buscar dados para {symbol}: {e}")

            # Exibir os símbolos filtrados
            logging.info("Balanços das ações:")

          #DRE#-----------------
            
            # Lista para armazenar os símbolos filtrados
            dre_annual_stocks = []

            # Iterar sobre cada símbolo de estoque
            for symbol in list_symbols:
                try:
                    # Obter os dados para o símbolo de ação atual
                    stock_data_dre = yf.Ticker(symbol + '.SA')
                    
                    # Obter os dados do DRE para a ação
                    dre_sheet_data = stock_data.financials
                    
                    # Verificar se todas as datas de 2023 a 2020 estão presentes nas colunas
                    if all(str(year) in balance_sheet_data.columns for year in range(2023, 2019, -1)):
                        dre_annual_stocks.append(symbol)
                        logging.info(f"Encontrados dados para {symbol} de 2023 a 2020:")                     
                        logging.info("="*50)
                
                except Exception as e:
                    print(f"Erro ao buscar dados para {symbol}: {e}")

            # Exibir os símbolos filtrados
            logging.info("DRE das ações:")

            #FUNDMENTOS#----------------

            fundamentals = {}

            # Assumindo que balance_annual_stocks e dre_annual_stocks já foram definidos
            # e contêm os símbolos das ações que passaram pelos critérios de filtro.

            # Iterar sobre os símbolos das ações que possuem tanto dados de balanço patrimonial quanto DRE disponíveis
            for symbol in set(balance_annual_stocks).intersection(dre_annual_stocks):
                try:
                    # Obter os dados para o símbolo da ação atual
                    stock_data = yf.Ticker(symbol + '.SA')
                    
                    # Obter os dados do balanço patrimonial e DRE da ação
                    balance_sheet_data = stock_data.balancesheet
                    dre_sheet_data = stock_data.financials
                    
                    # Adicionar os dados ao dicionário de fundamentos
                    fundamentals[symbol] = {
                        "Balance Sheet": balance_sheet_data,
                        "DRE": dre_sheet_data
                    }
                    
                    logging.info(f"Dados adicionados para {symbol}.")
                
                except Exception as e:
                    logging.info(f"Erro ao buscar dados para {symbol}: {e}")

        except Exception as e:
            raise CustomException(e,sys)       
                    
          




           


    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            """ //df=pd.read(???) """
                        

            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Inmgestion of the data iss completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    