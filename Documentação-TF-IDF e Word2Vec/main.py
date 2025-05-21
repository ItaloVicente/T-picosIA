from sklearn.datasets import fetch_20newsgroups

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re

# 1. Seleção de categorias
categorias = ['sci.space', 'rec.sport.baseball', 'talk.religion.misc', 'comp.graphics']

# 2. Carregando os dados
dados = fetch_20newsgroups(subset='train', categories=categorias, remove=('headers', 'footers', 'quotes'))

# 3. Limpeza básica dos textos
def limpar_texto(texto):
    texto = texto.lower()  # minúsculas
    texto = re.sub(r'\W+', ' ', texto)  # remove pontuação
    texto = re.sub(r'\d+', '', texto)  # remove números
    palavras = texto.split()
    palavras_filtradas = [p for p in palavras if p not in ENGLISH_STOP_WORDS]
    return ' '.join(palavras_filtradas)

# Aplicar limpeza em todos os textos
textos_limpos = [limpar_texto(texto) for texto in dados.data]

# Ver um exemplo
print("Texto original:\n", dados.data[0][:300])
print("\nTexto limpo:\n", textos_limpos[0][:300])