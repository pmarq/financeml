import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import yfinance as yf
from datetime import datetime
import numpy as np
import wget
from zipfile import ZipFile

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

path_data = "src/data/"

def ingestion_cvm(): 

# acessando a base de dados e crando arquivos históricos.
    url_base = 'https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/ITR/DADOS/'
    arquivos_zip = []
    for ano in range(2014,2025):
        arquivos_zip.append(f'itr_cia_aberta_{ano}.zip')

 # Baixando e extraindo os arquivos
    for arq in arquivos_zip:
        wget.download(url_base+arq, path_data)

    for arq in arquivos_zip:
        ZipFile(path_data+arq, 'r').extractall(f'{path_data}CVM_Data')

 # Concatenação e criação dos arquivos únicos por tipo de relatório
    nomes = ['BPA_con', 'BPA_ind', 'BPP_con', 'BPP_ind', 'DRE_con', 'DRE_ind', "DRA_ind", "DRA_con"]
    for nome in nomes:
        arquivo = pd.DataFrame()
        for ano in range(2014, 2025):
            arquivo = pd.concat([arquivo, pd.read_csv(f'{path_data}CVM_Data/itr_cia_aberta_{nome}_{ano}.csv', sep=";", decimal=",", encoding="ISO-8859-1")])
        arquivo.to_csv(f'{path_data}data_itr/itr_cia_aberta_{nome}_2014-2023.csv', index=False)


##----------------------------------##   
    ## COTAÇÃO
    ## Séries históricas disponíveis em
    ## http://www.b3.com.br/pt_br/market-data-e-indices/servicos-de-dados/market-data/historico/mercado-a-vista/series-historicas/
    ## Estrutura do arquivo disponível em 
    ## http://www.b3.com.br/data/files/33/67/B9/50/D84057102C784E47AC094EA8/SeriesHistoricas_Layout.pdf

# src/components/data_ingestion_.py


def processar_arquivo_bovespa(arquivo):
    """
    Processa o arquivo de cotações da Bovespa para um determinado ano.

    Parâmetros:
        arquivo (str): Caminho para o arquivo de cotações da Bovespa.

    Retorna:
        pd.DataFrame: DataFrame contendo os dados processados do arquivo.
    """
    tamanho_campos = [
        2, 8, 2, 12, 3, 12, 10, 3, 4, 13, 13, 13, 13, 13, 13, 13, 5, 18, 18, 13, 
        1, 8, 7, 13, 12, 3
    ]
    nomes_colunas = [
        "tipo_registro", "data_pregao", "cod_bdi", "cod_negociacao", "tipo_mercado", 
        "noma_empresa", "especificacao_papel", "prazo_dias_merc_termo", "moeda_referencia", 
        "preco_abertura", "preco_maximo", "preco_minimo", "preco_medio", "preco_ultimo_negocio", 
        "preco_melhor_oferta_compra", "preco_melhor_oferta_venda", "numero_negocios",
        "quantidade_papeis_negociados", "volume_total_negociado", "preco_exercicio", 
        "ìndicador_correcao_precos", "data_vencimento", "fator_cotacao", 
        "preco_exercicio_pontos", "codigo_isin", "num_distribuicao_papel"
    ]

    try:
        # Ler o arquivo com a codificação apropriada e larguras definidas
        print(f"Lendo arquivo: {arquivo}")
        dados_acoes = pd.read_fwf(arquivo, widths=tamanho_campos, encoding='latin1', header=1)
        dados_acoes.columns = nomes_colunas

        # Eliminar a última linha
        dados_acoes = dados_acoes.drop(len(dados_acoes) - 1)
        print(f"Última linha removida do arquivo: {arquivo}")

        # Ajustar valores com vírgula (dividir os valores dessas colunas por 100)
        lista_virgula = [
            "preco_abertura", "preco_maximo", "preco_minimo", "preco_medio", 
            "preco_ultimo_negocio", "preco_melhor_oferta_compra", 
            "preco_melhor_oferta_venda", "volume_total_negociado", 
            "preco_exercicio", "preco_exercicio_pontos"
        ]

        for coluna in lista_virgula:
            # Garantir que os valores são float antes da divisão
            dados_acoes[coluna] = pd.to_numeric(dados_acoes[coluna], errors='coerce') / 100.0
            print(f"Coluna {coluna} ajustada dividindo por 100.")

        print(f"Arquivo processado com sucesso: {arquivo}")
        return dados_acoes

    except FileNotFoundError:
        print(f"Arquivo {arquivo} não encontrado.")
        return None
    except Exception as e:
        print(f"Erro ao processar o arquivo {arquivo}: {e}")
        return None

def transformation_cotacaohist_b3(path_data):
    """
    Consolida os dados de cotações históricas da B3 de vários anos em um único DataFrame.

    Parâmetros:
        path_data (str): Caminho base para o diretório onde estão armazenados os arquivos de cotações.

    Retorna:
        None
    """
    diretorio = os.path.join(path_data, "cotacoes_historicas")
    lista_dados_acoes = []

    for ano in range(2014, 2024 + 1):
        arquivo = os.path.join(diretorio, f"COTAHIST_A{ano}.TXT")
        print(f"Processando arquivo: {arquivo}")
        dados_ano = processar_arquivo_bovespa(arquivo)
        if dados_ano is not None:
            lista_dados_acoes.append(dados_ano)
        else:
            print(f"Arquivo {arquivo} não foi adicionado à lista.")

    if lista_dados_acoes:
        try:
            # Concatenar todos os DataFrames em um único DataFrame
            dados_acoes_todos_anos = pd.concat(lista_dados_acoes, ignore_index=True)
            print("Todos os arquivos foram concatenados com sucesso.")

            # Verifica e cria o diretório de saída, se não existir
            output_dir = os.path.join(path_data, "consolidated_cotacao_hist")
            os.makedirs(output_dir, exist_ok=True)
            print(f"Diretório de saída verificado/criado: {output_dir}")
            
            # Salva o DataFrame consolidado
            output_path = os.path.join(output_dir, "COTAHIST_TODOS_ANOS.csv")
            dados_acoes_todos_anos.to_csv(output_path, index=False, encoding='latin1')
            print(f"Dados consolidados salvos em: {output_path}")

        except Exception as e:
            print(f"Erro ao concatenar ou salvar os dados: {e}")
    else:
        print("Nenhum dado foi consolidado. Verifique se os arquivos estão no diretório correto.")
 

##-------------------------------------------##
## TRASFORMAÇÃO

""" 
def transformation_cvm_balance():

# Função para consolidar dados de balanço (ativo e passivo      
    consolidate_balance_data() """

def consolidate_balance_data():
    # Caminhos para os arquivos de ativo e passivo
    caminho_ativo = f"{path_data}data_itr/itr_cia_aberta_BPA_ind_2014-2023.csv"
    caminho_passivo = f"{path_data}data_itr/itr_cia_aberta_BPP_ind_2014-2023.csv"
    
    # Carrega os arquivos de ativo e passivo sem configurações adicionais
    balance_tri_at = pd.read_csv(caminho_ativo)
    balance_tri_ps = pd.read_csv(caminho_passivo)
    
    # Concatena os DataFrames de ativo e passivo
    balance_tri = pd.concat([balance_tri_at, balance_tri_ps])
    
    # Filtra para manter apenas o último exercício
    balance_tri = balance_tri[balance_tri['ORDEM_EXERC'] == 'ÚLTIMO']
    
    # Verifica e cria o diretório consolidated_active_passive se não existir
    output_dir = f"{path_data}consolidated_active_passive"
    os.makedirs(output_dir, exist_ok=True)
    
    # Salva o DataFrame consolidado
    resume_balancedata_tri = balance_tri
    resume_balancedata_tri.to_csv(f"{output_dir}/resume_balancedata_tri.csv", index=False)

def balance_resume_data():
    # Caminhos para os arquivos de ativo e passivo
    caminho_df_balancedata_tri = f"{path_data}consolidated_active_passive/resume_balancedata_tri.csv"
    
    
    # Carrega os arquivos de ativo e passivo sem configurações adicionais
    df_balancedata_tri= pd.read_csv(caminho_df_balancedata_tri)

    # Unificar a remoção de colunas em uma única linha de código
    df_balancedata_tri = df_balancedata_tri.drop(columns=[
    'ESCALA_MOEDA', 'DT_REFER', 'GRUPO_DFP', 
    'VERSAO', 'MOEDA', 'ORDEM_EXERC', 'ST_CONTA_FIXA'])

    # Lista completa de contas, incluindo "Ativo Não Circulante" e "Ativo Realizável a Longo Prazo"
    full_cd_conta_list = [    
        ('1', 'Ativo Total'),
        ('1.01', 'Ativo Circulante'),
        ('1.02', 'Ativo Não Circulante'),  # Adicionando o Ativo Não Circulante
        ('1.02', 'Ativo Realizável a Longo Prazo'),  # Adicionando o Ativo Realizável a Longo Prazo
        ('1.02.01', 'Ativo Realizável a Longo Prazo'),  # Subconta de Ativo Realizável a Longo Prazo
        ('2.01', 'Passivo Circulante'),
        ('2.02', 'Passivo Não Circulante'),
        ('2.03', 'Patrimônio Líquido'),
        ('2.03.08.02', 'Lucro/Prejuízo do Exercício'),
        ('2.05', 'Patrimônio Líquido'),
        ('2.07', 'Patrimônio Líquido')]

    # Convertendo a lista em um DataFrame
    df_resume_list = pd.DataFrame(full_cd_conta_list, columns=['CD_CONTA', 'DS_CONTA'])

    # Realizando o filtro no DataFrame original (unique_accounts_df)
    df_balancedata_tri_resume = df_balancedata_tri.merge(df_resume_list, on=['CD_CONTA', 'DS_CONTA'])

    # Criar uma nova coluna que combina 'CD_CONTA' e 'DS_CONTA'
    df_balancedata_tri_resume['CD_DS_CONTA'] = df_balancedata_tri_resume['CD_CONTA'].astype(str) + ' - ' + df_balancedata_tri_resume['DS_CONTA']

    # Pivotar o DataFrame para transformar 'CD_DS_CONTA' em colunas, com 'VL_CONTA' como valores
    df_balance_tri_resume_col= df_balancedata_tri_resume.pivot_table(
        index=['CNPJ_CIA', 'DENOM_CIA', 'CD_CVM', 'DT_FIM_EXERC'],
        columns='CD_DS_CONTA',
        values='VL_CONTA',
        aggfunc='sum'  # Caso haja entradas duplicadas, somamos os valores
    ).reset_index()

    # Atualizando o DataFrame com as colunas principais, incluindo o Ativo Realizável a Longo Prazo e Lucro/Prejuízo do Exercício
    df_balance_tri_resume_col_res = df_balance_tri_resume_col[[
        'CNPJ_CIA', 
        'DENOM_CIA', 
        'CD_CVM', 
        'DT_FIM_EXERC', 
        '1 - Ativo Total', 
        '1.01 - Ativo Circulante', 
        '1.02 - Ativo Não Circulante',  # Mantendo Ativo Não Circulante
        '1.02 - Ativo Realizável a Longo Prazo',  # Adicionando Ativo Realizável a Longo Prazo
        '1.02.01 - Ativo Realizável a Longo Prazo',  # Adicionando subconta de Ativo Realizável a Longo Prazo
        '2.01 - Passivo Circulante', 
        '2.02 - Passivo Não Circulante', 
        '2.03 - Patrimônio Líquido', 
        '2.03.08.02 - Lucro/Prejuízo do Exercício',  # Adicionando Lucro/Prejuízo do Exercício
        '2.05 - Patrimônio Líquido', 
        '2.07 - Patrimônio Líquido']]
    
    # Unificar as colunas de Patrimônio Líquido em uma só, seguindo a ordem de precedência
    df_balance_tri_resume_col_res['Patrimonio_Liquido_Unificado'] = (
        df_balance_tri_resume_col_res['2.03 - Patrimônio Líquido']
        .fillna(df_balance_tri_resume_col_res['2.05 - Patrimônio Líquido'])
        .fillna(df_balance_tri_resume_col_res['2.07 - Patrimônio Líquido'])
    )

    # Unificar as colunas de Ativo Não Circulante, usando Ativo Realizável a Longo Prazo como substituto quando necessário
    df_balance_tri_resume_col_res['Ativo_Nao_Circulante_Unificado'] = (
        df_balance_tri_resume_col_res['1.02 - Ativo Não Circulante']
        .fillna(df_balance_tri_resume_col_res['1.02 - Ativo Realizável a Longo Prazo'])
        .fillna(df_balance_tri_resume_col_res['1.02.01 - Ativo Realizável a Longo Prazo'])
    )

    # Exibir as primeiras linhas para verificar o resultado
    df_balance_tri_resume_col_res[['CNPJ_CIA', 'DENOM_CIA', 'CD_CVM', 'DT_FIM_EXERC', 
                                '1 - Ativo Total', '1.01 - Ativo Circulante', 
                                'Ativo_Nao_Circulante_Unificado',  # Coluna unificada de Ativo Não Circulante
                                '2.01 - Passivo Circulante', '2.02 - Passivo Não Circulante', 
                                'Patrimonio_Liquido_Unificado',  # Coluna unificada de Patrimônio Líquido
                                '2.03.08.02 - Lucro/Prejuízo do Exercício']]  # Incluindo Lucro/Prejuízo do Exercício
        
    # Selecionando as colunas principais, incluindo Ativo_Nao_Circulante_Unificado e Patrimonio_Liquido_Unificado
    df_balance_tri_final = df_balance_tri_resume_col_res[[
        'CNPJ_CIA', 
        'DENOM_CIA', 
        'CD_CVM', 
        'DT_FIM_EXERC', 
        '1 - Ativo Total',
        '1.01 - Ativo Circulante', 
        'Ativo_Nao_Circulante_Unificado',  # Coluna unificada de Ativo Não Circulante
        '2.01 - Passivo Circulante',
        '2.02 - Passivo Não Circulante',
        'Patrimonio_Liquido_Unificado'  # Coluna unificada de Patrimônio Líquido
    ]]

    df_cnpj_acoes_b3 = pd.read_csv(f"{path_data}cnpj_acoes_b3.csv")

    # Renomear colunas no segundo dataframe para facilitar a fusão dos datasets
    df_cnpj_acoes_b3_clean = df_cnpj_acoes_b3.rename(columns={
        "CNPJ das empresas listadas na B3": "Ticker",
        "Unnamed: 1": "Denominação social",
        "Unnamed: 2": "Nome de pregão",
        "Unnamed: 3": "CNPJ"
    })

    # Remover a primeira linha que contém os cabeçalhos duplicados
    df_cnpj_acoes_b3_clean = df_cnpj_acoes_b3_clean.drop(index=0)

    # Corrigir o formato do CNPJ em ambos os dataframes
    df_cnpj_acoes_b3_clean['CNPJ'] = df_cnpj_acoes_b3_clean['CNPJ'].str.replace(r'\D', '', regex=True)
    df_balance_tri_final['CNPJ_CIA'] = df_balance_tri_final['CNPJ_CIA'].str.replace(r'\D', '', regex=True)

    # Realizar o merge dos datasets com base no CNPJ, mantendo o Ticker
    df_merged_with_ticker = pd.merge(
        df_balance_tri_final, 
        df_cnpj_acoes_b3_clean[['Ticker', 'CNPJ']], 
        left_on='CNPJ_CIA', 
        right_on='CNPJ', 
        how='inner'
    )

    # Atualizar o df_balance_tri_final com os dados resultantes do merge
    df_balance_tri_final = df_merged_with_ticker

    # Reordenar as colunas conforme o que foi solicitado, incluindo Ativo_Nao_Circulante_Unificado
    df_balance_tri_final = df_balance_tri_final[[
        'CNPJ', 
        'Ticker', 
        'DENOM_CIA', 
        'CD_CVM', 
        'DT_FIM_EXERC', 
        '1 - Ativo Total', 
        '1.01 - Ativo Circulante', 
        'Ativo_Nao_Circulante_Unificado',  # Incluindo a coluna de Ativo Não Circulante Unificado
        '2.01 - Passivo Circulante', 
        '2.02 - Passivo Não Circulante', 
        'Patrimonio_Liquido_Unificado']]
    
    # Verifica e cria o diretório consolidated_active_passive se não existir
    output_dir = f"{path_data}balance_resume"
    os.makedirs(output_dir, exist_ok=True)
    
    # Salva o DataFrame consolidado    
    df_balance_tri_final.to_csv(f"{output_dir}/df_balancedata_tri_final.csv", index=False)

    ## Nova Função ---- Balanço Estudo


    ...
## -----------## 


    ...
def transformation_cvm_dre():
    # Função para consolidar dados de DRE (Demonstração de Resultados) individual
    consolidate_dre_data()

def consolidate_dre_data():
    # Caminho para o arquivo de DRE individual
    caminho_dre_ind = f"{path_data}data_itr/itr_cia_aberta_DRE_ind_2014-2023.csv"
    
    # Carrega o arquivo de DRE individual
    dre_ind = pd.read_csv(caminho_dre_ind)
    
    # Filtra para manter apenas o último exercício
    dre_tri = dre_ind[dre_ind['ORDEM_EXERC'] == 'ÚLTIMO']
    
    # Define o nome e caminho do arquivo de saída, mantendo o diretório consolidated_dre
    output_dir = f"{path_data}consolidated_dre"
    os.makedirs(output_dir, exist_ok=True)
    
    # Salva o DataFrame filtrado
    resume_dredata_tri = dre_tri
    resume_dredata_tri.to_csv(f"{output_dir}/resume_dredata_tri.csv", index=False)

def dre_resume_data():
    # Caminhos para os arquivos de ativo e passivo
    caminho_df_dredata_tri = f"{path_data}consolidated_dre/resume_dredata_tri.csv"
        
        
    # Carrega os arquivos de ativo e passivo sem configurações adicionais
    df_dredata_tri= pd.read_csv(caminho_df_dredata_tri)

    df_dredata_tri = df_dredata_tri.drop(columns=['VERSAO', 'MOEDA', 'ORDEM_EXERC', 'ST_CONTA_FIXA','ESCALA_MOEDA','DT_REFER','DT_INI_EXERC'])

    # Contas de interesse que queremos filtrar
    selected_accounts = [
        "Receitas",
        "Resultado Operacional",
        "Lucro/Prejuízo do Período",
        "Lucro por Ação - (R$ / Ação)",
        "Resultado Bruto Intermediação Financeira",
        "Resultado Bruto de Serviços",
        "Outras Despesas/Receitas Operacionais",
        "Despesas Operacionais",
        "Resultado Antes do Resultado Financeiro e dos Tributos"
    ]
    # Filtrando o dataframe para incluir apenas essas contas
    df_dredata_tri_resume = df_dredata_tri[df_dredata_tri['DS_CONTA'].isin(selected_accounts)]

    df_pivot_dre_tri_resume_res = df_dredata_tri_resume[['CNPJ_CIA', 'DENOM_CIA', 'CD_CVM',
        'DT_FIM_EXERC', 'CD_CONTA', 'DS_CONTA', 'VL_CONTA']]
    
    # Criar a coluna 'CD_DS_CONTA' combinando 'CD_CONTA' e 'DS_CONTA'
    df_pivot_dre_tri_resume_res['CD_DS_CONTA'] = df_pivot_dre_tri_resume_res['CD_CONTA'] + ' - ' + df_pivot_dre_tri_resume_res['DS_CONTA']

    # Realizar a pivotagem para transformar as contas em colunas
    df_pivot_dre_tri_resume_res = df_pivot_dre_tri_resume_res.pivot_table(
        index=['CNPJ_CIA', 'DENOM_CIA', 'CD_CVM', 'DT_FIM_EXERC'], 
        columns='CD_DS_CONTA', 
        values='VL_CONTA'
    ).reset_index()

    # Identificando as colunas que contêm as variações de Receita, Lucro/Prejuízo e Resultado Operacional (3.05)
    receita_columns = [col for col in df_pivot_dre_tri_resume_res.columns if 'Receita' in col]
    lucro_prejuizo_columns = [col for col in df_pivot_dre_tri_resume_res.columns if 'Lucro/Prejuízo' in col]
    resultado_operacional_columns = [col for col in df_pivot_dre_tri_resume_res.columns if col.startswith('3.05')]

    # Criando as colunas consolidadas, pegando o primeiro valor não nulo para cada linha
    df_pivot_dre_tri_resume_res['Receita Consolidada'] = df_pivot_dre_tri_resume_res[receita_columns].bfill(axis=1).iloc[:, 0]
    df_pivot_dre_tri_resume_res['Lucro/Prejuízo Consolidado'] = df_pivot_dre_tri_resume_res[lucro_prejuizo_columns].bfill(axis=1).iloc[:, 0]
    df_pivot_dre_tri_resume_res['Resultado Operacional Consolidado'] = df_pivot_dre_tri_resume_res[resultado_operacional_columns].bfill(axis=1).iloc[:, 0]

    df_cnpj_acoes_b3 = pd.read_csv(f"{path_data}cnpj_acoes_b3.csv")

    # Renomear colunas no segundo dataframe para facilitar a fusão dos datasets
    df_cnpj_acoes_b3_clean = df_cnpj_acoes_b3.rename(columns={
        "CNPJ das empresas listadas na B3": "Ticker",
        "Unnamed: 1": "Denominação social",
        "Unnamed: 2": "Nome de pregão",
        "Unnamed: 3": "CNPJ"
    })

    # Remover a primeira linha que contém os cabeçalhos duplicados
    df_cnpj_acoes_b3_clean = df_cnpj_acoes_b3_clean.drop(index=0)

    # Corrigir o formato do CNPJ para garantir a correspondência entre os dataframes
    df_cnpj_acoes_b3_clean['CNPJ'] = df_cnpj_acoes_b3_clean['CNPJ'].str.replace(r'\D', '', regex=True)
    df_pivot_dre_tri_resume_res['CNPJ_CIA'] = df_pivot_dre_tri_resume_res['CNPJ_CIA'].str.replace(r'\D', '', regex=True)

    # Realizar o merge dos datasets com base no CNPJ e mantendo o Ticker
    df_merged_with_ticker = pd.merge(
        df_pivot_dre_tri_resume_res, 
        df_cnpj_acoes_b3_clean[['Ticker', 'CNPJ']], 
        left_on='CNPJ_CIA', 
        right_on='CNPJ', 
        how='inner'
    )

    df_pivot_dre_tri_resume_res = df_merged_with_ticker

    # Reordenar as colunas conforme o que foi solicitado
    df_pivot_dre_tri_resume_res = df_pivot_dre_tri_resume_res[[
        'CNPJ', 'Ticker', 'DENOM_CIA', 'CD_CVM', 'DT_FIM_EXERC','Resultado Operacional Consolidado', 'Lucro/Prejuízo Consolidado'
    ]]

     # Verifica e cria o diretório consolidated_active_passive se não existir
    output_dir = f"{path_data}dre_resume"
    os.makedirs(output_dir, exist_ok=True)
    
    # Salva o DataFrame consolidado    
    df_pivot_dre_tri_resume_res.to_csv(f"{output_dir}/dre_tri_res.csv", index=False)

## Nova Função ---- Dre Estudo



##----------------## 
## Resume Cotações


def consolidate_cot_hist_resume():
    # Caminho para o arquivo de DRE individual
    caminho_cotacao_hist_consol = f"{path_data}consolidated_cotacao_hist/COTAHIST_TODOS_ANOS.csv"

    # Ler o arquivo CSV com a codificação apropriada
    preco_acoes_todos_os_anos = pd.read_csv(caminho_cotacao_hist_consol, encoding='latin1')     

    # Filtrando apenas as colunas desejadas
    colunas_desejadas = ['data_pregao', 'cod_negociacao', 'noma_empresa', 'preco_ultimo_negocio']
    preco_acoes = preco_acoes_todos_os_anos[colunas_desejadas]

    # Filtrando apenas as empresas com cod_negociacao terminando em '3'
    preco_acoes = preco_acoes[preco_acoes['cod_negociacao'].str.endswith('3')]

    preco_acoes_final3 = preco_acoes     
    
    # Define o nome e caminho do arquivo de saída, mantendo o diretório consolidated_dre
    output_dir = f"{path_data}resume_cotacao_hist"
    os.makedirs(output_dir, exist_ok=True)
    
    # Salva o DataFrame filtrado
    preco_acoes_final3 = preco_acoes
    preco_acoes_final3.to_csv(f"{output_dir}/resume_cotacao_hist.csv", index=False)


## Nova Função - Estudo da Cotação







    





           

    

