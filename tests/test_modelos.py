from modelos import ClassificacaoPergunta


def test_classificacao_pergunta_valida_dados_estruturados():
    classificacao = ClassificacaoPergunta(
        categoria="dados",
        palavras_chave=["chave primária", "chave estrangeira"],
        nivel="basico",
    )

    assert classificacao.categoria == "dados"
    assert classificacao.palavras_chave == ["chave primária", "chave estrangeira"]
    assert classificacao.nivel == "basico"


def test_classificacao_pergunta_rejeita_categoria_invalida():
    try:
        ClassificacaoPergunta(
            categoria="categoria_invalida",
            palavras_chave=[],
            nivel="basico",
        )
    except ValueError:
        return

    raise AssertionError("A categoria inválida deveria ser rejeitada.")
