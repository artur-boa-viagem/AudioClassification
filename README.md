# AudioClassification
Projeto da disciplina de Sinais e Sistemas no curso de Engenharia da Computação do CIn-UFPE

# Clone o Projeto
Clone o projeto com:
```console
$ git clone https://github.com/luanjaardim/AudioClassification.git
```

Depois disso você deve instalar alguns pacotes de python que são utilizados nesse projeto, se quiser você pode criar um python enviroment antes de instalá-los:
```console
$ cd AudioClassification/
$ pip install -r requirements.txt
```
(Atenção! Os pacotes acima possuem a versão mais atual para um arch Linux x86-64 em 20/03/2024, caso a instalação de algum pacote falhe, procure a versão correta para o pacote no seu sistema)

# Organização do Projeto
O projeto possui o diretório 'src/' que contém os códigos fontes do projeto, e os diretórios 'Animal-Sound-Dataset/' e 'Spectrogram-DB/' que contém os datasets utilizados no projeto, no primeiro estão os arquivos de áudio e no segundo estão os espectrogramas gerados a partir dos arquivos de áudio, estes que por sua vez alimentam os modelos utilizados no projeto.

# Treinando um modelo
Para treinar um modelo, você pode utilizar o script `train.py`, por exemplo:
```console
$ python train.py train my_model HOG
```
O comando acima vai treinar um modelo que utiliza do extrator HOG, LBP é a outra opção, para a classificação de espectrogramas.
O modelo treinado será salvo no diretório `src/models/` com o nome `HOG_my_model.keras`.

# Testando um modelo treinado
Para testar um modelo treinado, você pode utilizar o script `test.py`, por exemplo:
```console
$ python test.py test models/HOG_my_model.keras
```
Desta forma você irá testar o modelo `HOG_my_model.keras` com o conjunto de testes.
