## 📂 Dataset Utilizado
### **TuPyE-Dataset**
🔗 [Link oficial no Hugging Face](https://huggingface.co/datasets/Silly-Machine/TuPyE-Dataset)  

### Autores

- **Lucas Zito**  
  UFSCar  
  Sorocaba - SP  
  RA: 802626

- **Marcelo Pirro**  
  UFSCar  
  Sorocaba - SP  
  RA: 800510

- **Nicolas Benitiz**  
  UFSCar  
  Sorocaba - SP  
  RA: 813037

- **Rafael Campos**  
  UFSCar  
  Sorocaba - SP  
  RA: 801968

- **Rafael Penido**  
  UFSCar  
  Sorocaba - SP  
  RA: 802726
  
# 📚 Repositório de Estratégias de PLN

Este repositório contém uma coleção de notebooks que exploram diferentes estratégias de **Processamento de Linguagem Natural (PLN)**, bem como variações de cada abordagem. O objetivo é comparar técnicas e avaliar seu desempenho em tarefas como classificação de texto e análise de sentimentos.

## 📁 Notebooks

### 🔹 TF-IDF
1. **1-A-tf_idf.ipynb** , **1-B-tf_idf.ipynb** ,  **1-C-tf_idf.ipynb**  
   Implementação da técnica **TF-IDF** para extração de características textuais e variações como ajustes nos parâmetros de vetorização.
   
### 🔹 Word2Vec
2. **2-A-word2vec.ipynb**  
   Utiliza o modelo **Word2Vec** para representar palavras como vetores densos, aplicando-o em tarefas de PLN.
   
### 🔹 BERT
3. **3-A-Bert.ipynb**  
   Aplicação do modelo **BERT** para tarefas como classificação de texto, análise de sentimentos e categorização multilabel. Além de variações do notebook anterior como ajustes nos hiperparâmetros do modelo por exemplo.

### 🔹 Multimodal
4. **Multimodal.ipynb**  
   Explora técnicas **multimodais**, combinando texto e imagens para tarefas específicas de PLN.
   
### 🔹 Explicações e Avaliações
5. **aval_explain.ipynb**  
   Explica as métricas utilizadas na avaliação dos modelos desenvolvidos.

6. **modelo_base.ipynb**  
    Contém um modelo base utilizado como referência para comparação com outras abordagens.

7. **regras.ipynb**  
    Abordagem de detecção de discurso de ódio e categorização utilizando regras.
   
8. **preprocessamento.ipynb**  
    Preprocessamento utilizado para tratamento dos textos em todos as estratégias

## 📄 Arquivos de Dados

- **HateBR.csv**  
  Conjunto de dados utilizado para treinar e avaliar o modelo multimodal.

- **explain.csv**  
  Análise realizadas pelos membros do grupo para comparar com com as palavras apontadas pelo explain.

- **analise.zip**  
  Arquivo compactado com análises adicionais realizadas durante o desenvolvimento.

## 🖼️ Imagens

- **imagebolsonaro.jpg**  
- **imageteletubbies.jpg**  
  Imagens utilizadas no notebook multimodal para demonstrações práticas.
  
## 📑 Documentação

- **Relatório Final.pdf**  
  Documento com a descrição dos resultados obtidos nas diferentes abordagens testadas.


