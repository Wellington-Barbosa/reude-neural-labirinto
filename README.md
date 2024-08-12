# Expression-direct - Rede Neural de Movimentação

Esta aplicação permite controlar uma bolinha em um labirinto utilizando movimentos da mão capturados pela webcam. A aplicação utiliza MediaPipe para detecção de mãos e um modelo de rede neural treinado para reconhecer gestos que indicam movimentos para a esquerda ou direita.

## Funcionalidades

- **Detecção de Mãos**: Utiliza MediaPipe para detectar pontos de referência das mãos em tempo real.
- **Treinamento do Modelo**: Coleta dados de movimentos de mãos e treina um modelo de rede neural para reconhecer gestos.
- **Movimento da Bolinha**: Baseado nos gestos reconhecidos, a bolinha no labirinto se move para a esquerda ou direita.

## Requisitos

- Python 3.7 ou superior
- Webcam

### Bibliotecas Necessárias

Instale as bibliotecas necessárias com o seguinte comando:

```bash
pip install opencv-python mediapipe tensorflow pandas scikit-learn
```
## Como Usar

1. Coletar Dados de Referência
Execute a aplicação para coletar dados de movimentos de mão. Pressione a tecla 'a' para coletar dados para o movimento à esquerda e 'd' para o movimento à direita. Pressione 'q' para finalizar a coleta.

2. Treinar o Modelo
Se os dados foram coletados com sucesso, o modelo será treinado automaticamente após finalizar a coleta pressionando 'q'. O modelo será salvo para uso posterior.

3. Jogar o Jogo do Labirinto
Após coletar e treinar o modelo, ou se um modelo pré-existente for carregado, a aplicação entrará no modo de jogo. Utilize os movimentos de mão para mover a bolinha no labirinto.

## Estrutura do Projeto

* app.py: Arquivo principal da aplicação que coleta dados, treina o modelo e controla o jogo.
* hand_gesture_model.h5: Arquivo de modelo treinado salvo.
* hand_gesture_data.csv: Arquivo CSV que armazena os dados de treinamento coletados.
* labirinto.png: Imagem do labirinto utilizada como fundo no jogo.

## Exemplo de Uso

1. Execute a aplicação para coletar dados de referência:

```bash
python app.py
```

* Coloque sua mão na frente da webcam e pressione 'a' para coletar dados de movimentos para a esquerda.
* Pressione 'd' para coletar dados de movimentos para a direita.
* Pressione 'q' para finalizar a coleta e treinar o modelo.

2. Após o treinamento, a aplicação entrará no modo de jogo. Use os movimentos da mão para mover a bolinha no labirinto.

## Nota

Se a imagem do labirinto não for exibida corretamente, verifique se o arquivo labirinto.png está no mesmo diretório que o arquivo app.py.

## Problemas Conhecidos

* A precisão do modelo pode variar dependendo da quantidade e qualidade dos dados coletados.
* Certifique-se de que a webcam está funcionando corretamente e que há iluminação adequada para uma melhor detecção das mãos.