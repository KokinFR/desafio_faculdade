import json
from datetime import datetime

from assistente import AssistenteEstudos


PERGUNTAS = [
    "O que é um algoritmo e como podemos avaliar sua eficiência?",
    "Qual é a diferença entre pilha e fila?",
    "Para que servem as chaves primária e estrangeira em um banco de dados?",
    "Qual é a função de um sistema operacional?",
    "Qual é a diferença entre inteligência artificial e aprendizado de máquina?",
]


def main() -> None:
    assistente = AssistenteEstudos()
    linhas = ["LOG DE TESTES — ASSISTENTE ADA", f"Data: {datetime.now():%d/%m/%Y %H:%M}", ""]
    for numero, pergunta in enumerate(PERGUNTAS, start=1):
        resultado = assistente.responder(pergunta)
        bloco = [
            f"TESTE {numero}",
            f"Pergunta: {pergunta}",
            f"Resposta: {resultado['resposta']}",
            f"Classificação JSON: {json.dumps(resultado['classificacao'], ensure_ascii=False)}",
            f"Similaridade: {resultado['similaridade']:.4f}",
            f"Tokens da pergunta: {resultado['tokens_pergunta']}",
            f"Tokens da resposta: {resultado['tokens_resposta']}",
            "-" * 70,
            "",
        ]
        linhas.extend(bloco)
        print("\n".join(bloco))
    with open("log_testes.txt", "w", encoding="utf-8") as arquivo:
        arquivo.write("\n".join(linhas))
    print("Log salvo em log_testes.txt")


if __name__ == "__main__":
    main()
