"""
Camadas do ecossistema usadas neste projeto:
- Modelo: Phi-3, executado localmente pelo Ollama.
- Framework/biblioteca de IA: sentence-transformers (gera embeddings).
- Base vetorial: NumPy (armazena vetores e calcula similaridade de cosseno).
- Ferramenta de desenvolvimento: transformers/AutoTokenizer (conta tokens).
- Interface opcional: Streamlit (front-end web local).

LLM x agente:
O Phi-3/Ollama é o LLM, isto é, o motor que gera texto. O agente é a lógica
deste programa: classificar a pergunta, buscar contexto, montar o prompt,
decidir o que enviar ao modelo e devolver a resposta com seus metadados.
"""

import json
from typing import Any

import numpy as np
import ollama
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


MODELO_OLLAMA = "phi3"
MODELO_EMBEDDING = "all-MiniLM-L6-v2"
MODELO_TOKENIZER = "microsoft/Phi-3-mini-4k-instruct"

BASE_CONHECIMENTO = [
    "Algoritmos são sequências finitas e ordenadas de passos para resolver problemas. Sua qualidade pode ser analisada pelo tempo de execução e pelo uso de memória, frequentemente descritos pela notação Big O.",
    "Estruturas de dados organizam informações para facilitar operações. Listas, pilhas, filas, árvores, grafos e tabelas hash possuem características diferentes de acesso, inserção, remoção e busca.",
    "Programação orientada a objetos organiza software em objetos que combinam estado e comportamento. Seus conceitos principais incluem classes, objetos, encapsulamento, herança, polimorfismo e abstração.",
    "Bancos de dados relacionais armazenam dados em tabelas relacionadas. SQL permite consultar e modificar os dados, enquanto chaves primárias identificam registros e chaves estrangeiras representam relacionamentos.",
    "Sistemas operacionais administram processador, memória, arquivos e dispositivos. Eles oferecem serviços aos programas e coordenam processos, threads, escalonamento, memória virtual e controle de acesso.",
    "Redes de computadores conectam dispositivos para compartilhar dados e recursos. Modelos como TCP/IP dividem a comunicação em camadas, e protocolos como HTTP, TCP, IP e DNS cumprem funções específicas.",
    "Engenharia de software aplica processos, métodos e ferramentas ao desenvolvimento e à manutenção de sistemas. Requisitos, projeto, implementação, testes, versionamento e evolução fazem parte do ciclo de vida.",
    "Inteligência artificial cria sistemas capazes de executar tarefas associadas à inteligência humana. Aprendizado de máquina aprende padrões a partir de dados, enquanto aprendizado profundo utiliza redes neurais com várias camadas.",
    "Segurança da informação busca preservar confidencialidade, integridade e disponibilidade. Autenticação, autorização, criptografia, atualizações e cópias de segurança reduzem riscos, mas não eliminam todas as ameaças.",
    "Compiladores traduzem código-fonte para outra representação, geralmente código de máquina. A compilação envolve análise léxica, sintática e semântica, otimização e geração de código; interpretadores executam instruções durante a execução.",
]

PERSONA = (
    "Você é Ada, uma tutora paciente e didática de Ciência da Computação. "
    "Explique com linguagem clara para estudantes de graduação, use exemplos curtos "
    "quando ajudarem e não invente informações ausentes no contexto. "
    "Faça o raciocínio passo a passo internamente, mas apresente somente a explicação final."
)


def extrair_conteudo(resposta: Any) -> str:
    """Aceita objetos e dicionários retornados por diferentes versões do Ollama."""
    if isinstance(resposta, dict):
        return resposta["message"]["content"]
    return resposta.message.content


class AssistenteEstudos:
    def __init__(self) -> None:
        print("Carregando modelos locais...")
        self.modelo_embeddings = SentenceTransformer(MODELO_EMBEDDING)
        self.tokenizer = AutoTokenizer.from_pretrained(MODELO_TOKENIZER)
        self.embeddings_base = self.modelo_embeddings.encode(
            BASE_CONHECIMENTO, normalize_embeddings=True
        )

    def contar_tokens(self, texto: str) -> int:
        return len(self.tokenizer.encode(texto, add_special_tokens=False))

    def buscar_contexto(self, pergunta: str) -> tuple[str, float, int]:
        embedding_pergunta = self.modelo_embeddings.encode(
            pergunta, normalize_embeddings=True
        )
        # Com vetores normalizados, produto escalar equivale ao cosseno.
        similaridades = np.dot(self.embeddings_base, embedding_pergunta)
        indice = int(np.argmax(similaridades))
        return BASE_CONHECIMENTO[indice], float(similaridades[indice]), indice

    def classificar_pergunta(self, pergunta: str) -> dict[str, Any]:
        """Resposta estruturada obrigatória usando format='json'."""
        prompt = f"""
Classifique a pergunta de Ciência da Computação.
Responda exclusivamente como JSON válido neste formato:
{{"categoria": "algoritmos|dados|programacao|sistemas|redes|software|ia|seguranca|compiladores|outros", "palavras_chave": ["..."], "nivel": "basico|intermediario|avancado"}}
Pergunta: {pergunta}
"""
        resposta = ollama.chat(
            model=MODELO_OLLAMA,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0},
        )
        return json.loads(extrair_conteudo(resposta))

    def montar_prompt(
        self, pergunta: str, contexto: str, classificacao: dict[str, Any]
    ) -> str:
        """Template reutilizável com few-shot e instrução passo a passo."""
        return f"""
Use prioritariamente o CONTEXTO RECUPERADO para responder.
Se ele não for suficiente, diga claramente que a base não contém detalhes suficientes.
Analise internamente passo a passo antes de formular uma resposta curta e didática.

EXEMPLOS (few-shot):
Pergunta: O que é uma pilha?
Resposta: Uma pilha é uma estrutura LIFO: o último elemento inserido é o primeiro a sair. Um exemplo é uma pilha de pratos.

Pergunta: Para que serve uma chave primária?
Resposta: Ela identifica de forma única cada registro de uma tabela, evitando ambiguidade na localização dos dados.

CLASSIFICAÇÃO: {json.dumps(classificacao, ensure_ascii=False)}
CONTEXTO RECUPERADO: {contexto}
PERGUNTA DO ESTUDANTE: {pergunta}
RESPOSTA:
""".strip()

    def responder(self, pergunta: str) -> dict[str, Any]:
        contexto, similaridade, indice = self.buscar_contexto(pergunta)
        classificacao = self.classificar_pergunta(pergunta)
        prompt = self.montar_prompt(pergunta, contexto, classificacao)
        resposta_ollama = ollama.chat(
            model=MODELO_OLLAMA,
            messages=[
                {"role": "system", "content": PERSONA},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.2},
        )
        resposta = extrair_conteudo(resposta_ollama).strip()
        return {
            "pergunta": pergunta,
            "resposta": resposta,
            "tokens_pergunta": self.contar_tokens(pergunta),
            "tokens_resposta": self.contar_tokens(resposta),
            "contexto": contexto,
            "indice_contexto": indice,
            "similaridade": similaridade,
            "classificacao": classificacao,
        }


def executar_terminal() -> None:
    assistente = AssistenteEstudos()
    print("\nAssistente Ada iniciado. Digite 'sair' para encerrar.\n")
    while True:
        pergunta = input("Você: ").strip()
        if pergunta.lower() in {"sair", "exit", "quit"}:
            print("Até a próxima sessão de estudos!")
            break
        if not pergunta:
            continue
        try:
            resultado = assistente.responder(pergunta)
            print(f"\nAda: {resultado['resposta']}")
            print(f"\nCategoria (JSON): {json.dumps(resultado['classificacao'], ensure_ascii=False)}")
            print(f"Similaridade do contexto: {resultado['similaridade']:.4f}")
            print(f"Tokens da pergunta: {resultado['tokens_pergunta']}")
            print(f"Tokens da resposta: {resultado['tokens_resposta']}\n")
        except Exception as erro:
            print(f"\nNão foi possível responder: {erro}\n")


if __name__ == "__main__":
    executar_terminal()
