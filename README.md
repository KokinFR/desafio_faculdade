# Assistente Ada — Ciência da Computação

Projeto da disciplina **Evolução de Software**, referente às aulas 01, 02 e 03. O sistema combina um modelo local executado pelo Ollama, busca RAG com embeddings, técnicas de engenharia de prompt, resposta JSON e contagem de tokens.

## 1. Pré-requisitos

- Python 3.10 ou superior;
- Ollama instalado e aberto.

No terminal, baixe o modelo:

```bash
ollama pull phi3
```

## 2. Instalação

Abra o terminal dentro da pasta do projeto e execute:

```bash
python -m venv .venv
```

No Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Na primeira execução, os modelos de embeddings e tokenização serão baixados. Depois disso, o assistente e o Phi-3 funcionam localmente.

## 3. Rodar no terminal

```bash
python assistente.py
```

Digite `sair` para encerrar.

## 4. Rodar a interface Streamlit

```bash
streamlit run app.py
```

O navegador deverá abrir em `http://localhost:8501`.

## 5. Gerar o log das cinco perguntas

Com o Ollama aberto, execute:

```bash
python executar_testes.py
```

O arquivo `log_testes.txt` será criado com perguntas, respostas, JSON, similaridades e tokens. Esse arquivo deve ser gerado no computador da apresentação, pois as respostas do modelo podem variar. Tire também um print do terminal ou da interface após os cinco testes.

## Requisitos atendidos

- Modelo Phi-3 local via Ollama;
- comentário das camadas do ecossistema no topo do script;
- base própria com 10 trechos;
- embeddings com `all-MiniLM-L6-v2`;
- RAG com similaridade de cosseno/produto escalar em NumPy;
- persona Ada definida em mensagem `system`;
- few-shot e raciocínio passo a passo interno;
- classificação estruturada usando `format="json"`;
- prompt final criado por função reutilizável;
- tokens de pergunta e resposta exibidos;
- explicação, em comentário, da diferença entre LLM e agente;
- terminal e front-end Streamlit;
- script para registrar cinco testes.

## Observação acadêmica

O agente não é o modelo Phi-3 isoladamente. O modelo apenas gera texto. O agente é a combinação das decisões programadas: classificar a pergunta, gerar o embedding, selecionar o trecho, montar o prompt e solicitar a resposta ao LLM.
