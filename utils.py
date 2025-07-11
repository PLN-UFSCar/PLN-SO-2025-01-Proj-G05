import re
import unicodedata
import string
import emoji
import spacy
import nltk
from nltk.corpus import stopwords
from typing import List
from sklearn.metrics import f1_score, hamming_loss, accuracy_score
import numpy as np

nltk.download('stopwords')
stopwords_pt = set(stopwords.words('portuguese'))
nlp = spacy.load('pt_core_news_sm')

ABBREVIATIONS = {
    "vc": "você", "td": "tudo", "pq": "porque", "blz": "beleza",
    "q": "que", "n": "não", "ñ":"não", "ta": "tá", "eh": "é", "aki": "aqui",
    "msg": "mensagem", "tbm": "também", "hj": "hoje", "obg": "obrigado", "tmj": "estamos_juntos", "fpd": "filho_da_puta"
}

CONTRACTIONS = {
    "tá": "está", "tô": "estou", "cê": "você", "num": "em um", "dum": "de um"
}

SWEAR_WORDS = {
    "foda", "merda", "caralho", "puta", "porra", "viado", "bicha", "desgraçado", "escroto"
}

MWES = [
    # Insultos diretos e compostos
    "filho da mãe",
    "filha da mãe",
    "filho da puta",
    "filha da puta",
    "vai se ferrar",
    "tomar no cu",
    "vai tomar no cu",
    "encher o saco",
    "encher a porra do saco",
    "enfia no cu",
    "vai pra puta que pariu",
    "mala sem alça",
    "pau no cu",
    "boca de bosta",
    "idiota do caralho",
    "retardado mental",
    "bando de lixo",
    "cabeça de vento",
    "mente pequena",
    "mau caráter",
    "falso moralista",
]

def expand_abbreviations(text):
    return ' '.join([ABBREVIATIONS.get(w, w) for w in text.split()])

def expand_contractions(text):
    return ' '.join([CONTRACTIONS.get(w, w) for w in text.split()])

def normalize_laughter(text):
    return re.sub(r"(k|rs|ha|he){3,}", "risada", text)

def remove_accents(text):
    # Substitui apenas acentos, mantendo caracteres especiais como ç e pontuação
    normalized = unicodedata.normalize('NFD', text)
    # Remove apenas marcas de acento (não remove letras como ç, ñ, etc.)
    return ''.join([c for c in normalized if unicodedata.category(c) != 'Mn'])

def handle_negation(tokens):
    result = []
    negate = False
    for token in tokens:
        if token in {"não", "nunca", "jamais"}:
            negate = True
            result.append(token.text)
        elif token.is_punct or token.is_space:
            negate = False
            result.append(token.text)
        elif negate:
            result.append("não_" + token.lemma_)
        else:
            result.append(token.lemma_)
    return result

def merge_mwes(text: str, mwe_list: list) -> str:
    """
    Substitui expressões multipalavras por formas unificadas com underscore.
    Exemplo: "filho da mãe" -> "filho_da_mãe"
    """
    for expr in sorted(mwe_list, key=len, reverse=True):
        # Cria regex segura, insensível a maiúsculas
        pattern = re.compile(r'\b' + re.escape(expr) + r'\b', flags=re.IGNORECASE)
        replacement = expr.replace(" ", "_")
        text = pattern.sub(replacement, text)
    return text

def replace_named_entities(text: str) -> str:
    """
    Substitui entidades nomeadas por categorias genéricas (e.g. PESSOA, ORG).
    Ex: "Bolsonaro é um lixo" -> "ENTIDADE_PESSOA é um lixo"
    """
    doc = nlp(text)
    new_text = text
    offset = 0

    # Mapeamento personalizado
    category_map = {
        "PER": "ENTIDADE_PESSOA",
        "ORG": "ENTIDADE_ORGANIZACAO",
        "LOC": "ENTIDADE_LOCAL",
        "MISC": "ENTIDADE_MISC"
    }

    for ent in doc.ents:
        label = ent.label_
        if label in category_map:
            start = ent.start_char + offset
            end = ent.end_char + offset
            replacement = category_map[label]
            new_text = new_text[:start] + replacement + new_text[end:]
            # Atualiza o offset caso o replacement tenha tamanho diferente
            offset += len(replacement) - (end - start)

    return new_text

def preprocess_text(text: str, config) -> str:
    
    if config.get("lowercase"):
        text = text.lower()

    if config.get("remove_urls"):
        text = re.sub(r"http\S+|www\S+", "", text)

    if config.get("remove_mentions_hashtags"):
        if config.get("split_hashtags"):
            text = re.sub(r"#", " ", text)
        else:
            text = re.sub(r"@\w+|#\w+", "", text)

    if config.get("remove_emojis"):
        text = emoji.replace_emoji(text, replace='')

    if config.get("remove_numbers"):
        text = re.sub(r"\d+", "", text)
    
    if config.get("replace_named_entities"):
        text = replace_named_entities(text)

    if config.get("merge_mwes"):
        text = merge_mwes(text, MWES)

    doc = nlp(text)
    filtered_tokens = []

    negation = False  # controle de escopo de negação

    for token in doc:
        word = token.text.lower()

        if config.get("remove_punctuation") and token.is_punct:
            continue

        if config.get("remove_stopwords") and word in stopwords_pt:
            continue

        if config.get("min_token_length") and len(word) < config["min_token_length"]:
            continue

        if config.get("pos_filter") and token.pos_ not in {"NOUN", "ADJ"}:
            continue

        if config.get("expand_abbreviations") and word in ABBREVIATIONS:
            word = ABBREVIATIONS[word]
        if config.get("expand_contractions") and word in CONTRACTIONS:
            word = CONTRACTIONS[word]

        if config.get("normalize_laughter"):
            word = normalize_laughter(word)

        if config.get("replace_swears") and word in SWEAR_WORDS:
            filtered_tokens.append("palavra_ofensiva")
            continue

        # Negation scope handling
        if config.get("negation_scope"):
            if word in {"não", "nunca", "jamais", "nem"}:
                negation = True
                continue  # opcional: podemos omitir o marcador
            elif token.is_punct:
                negation = False

            if negation:
                word = "NEG_" + word

        # Lematização
        if config.get("lemmatize"):
            lemma = token.lemma_.strip()
            word = lemma if lemma else word

        if config.get("remove_accents"):
            word = remove_accents(word)

        filtered_tokens.append(word)

    return ' '.join(filtered_tokens)

def format_lime_output(texto, predict_proba, target, explicador):
    print("Categorias de discurso de ódio:")
    probs = predict_proba([texto])[0]
    
    categorias_relevantes = [(classe, prob) for classe, prob in zip(target, probs) if prob >= 0.355]
    
    if categorias_relevantes:
        for classe, prob in categorias_relevantes:
            print(f"{classe}({int(prob * 100)}%)")
        
        print("\n🔍 Explicação - presença das palavras:")

        # Filtra palavras com peso > 0.05 (positivo ou negativo)
        palavras_filtradas = [(palavra, peso) for palavra, peso in explicador.as_list() if abs(peso) > 0.12]
        
        # Ordena por peso decrescente (absoluto)
        palavras_ordenadas = sorted(palavras_filtradas, key=lambda x: abs(x[1]), reverse=True)

        if palavras_ordenadas:
            for palavra, peso in palavras_ordenadas:
                print(f"{palavra}({peso:+.3f})")
        else:
            # Se nenhuma palavra com peso relevante, pega a com maior peso
            palavra, peso = max(explicador.as_list(), key=lambda x: abs(x[1]))
            print(f"{palavra}({peso:+.3f})")
    
    else:
        print("O texto não é discurso de ódio 🕊️")

def print_multilabel_metrics(y_true, y_pred):
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    hamming = hamming_loss(y_true, y_pred)
    exact = accuracy_score(y_true, y_pred)

    print("\n📊 Avaliação Multilabel")
    print("=" * 30)
    print(f"✔️ F1 Score (Micro):     {f1_micro:.4f}")
    print(f"✔️ F1 Score (Macro):     {f1_macro:.4f}")
    print(f"✔️ F1 Score (Weighted):  {f1_weighted:.4f}")
    print(f"⚠️ Hamming Loss:         {hamming:.4f}")
    print(f"✅ Subset Accuracy:      {exact:.4f}")
    print("=" * 30)