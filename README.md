# Reconhecimento de Gestos Manuais e Jogo de Labirinto

Este projeto implementa um sistema de reconhecimento de gestos manuais utilizando a captura de vídeo por webcam. Ele permite que os usuários controlem uma bola dentro de um labirinto com base nos movimentos da mão. O sistema é construído usando técnicas de visão computacional e aprendizado de máquina, incluindo OpenCV, MediaPipe e TensorFlow.

## Índice
- [Instalação](#instalação)
- [Uso](#uso)
- [Como Funciona](#como-funciona)
- [Treinamento do Modelo](#treinamento-do-modelo)
- [Coleta de Dados](#coleta-de-dados)
- [Dependências](#dependências)
- [Licença](#licença)

## Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/seuusuario/hand-gesture-maze-game.git
   
2. Navegue até o diretório do projeto:
    ```bash
    cd hand-gesture-maze-game

3. Instale os pacotes Python necessários:
    ```bash
    pip install -r requirements.txt

## Uso

Para executar o sistema de reconhecimento de gestos manuais e iniciar o jogo de labirinto, execute o seguinte comando:
    
1. Execução:
    ```bash
    python app.py

## Como Funciona

1. **Detecção de Mãos**: O sistema utiliza o MediaPipe para detectar e rastrear os pontos de referência das mãos em tempo real.

2. **Coleta de Dados**: Os pontos de referência das mãos (coordenadas x e y) são coletados quando teclas específicas são pressionadas para rotular os movimentos (esquerda, direita, cima, baixo).

3. **Treinamento do Modelo**: Um modelo de rede neural é treinado com os dados coletados para prever a direção do movimento da mão.

4. **Controle do Jogo**: O modelo treinado é usado para controlar o movimento de uma bola dentro da imagem de um labirinto. O movimento da bola é determinado pela direção do movimento da mão.

## Treinamento do Modelo

Se você não tiver um modelo pré-treinado, precisará coletar dados e treinar o modelo:

1. **Coleta de Dados**:

    * Execute o programa e pressione as seguintes teclas para coletar dados de movimento:
        * 'a': Coleta dados para o movimento à esquerda.
        * 'd': Coleta dados para o movimento à direita.
        * 'w': Coleta dados para o movimento para cima.
        * 's': Coleta dados para o movimento para baixo.
        * Pressione 'q' para parar a coleta de dados.

2. **Treinamento do Modelo**:

    * Se não houver um modelo existente, o programa automaticamente treinará um modelo usando os dados coletados e o salvará como hand_gesture_model.h5.

    * O modelo será avaliado nos dados de teste, e sua precisão será exibida.
    
3. Jogando o Jogo:

    Após o treinamento do modelo, a bola dentro do labirinto poderá ser controlada com base nos movimentos da sua mão em frente à webcam.
    
## Coleta de Dados

A coleta de dados é crucial para a precisão do modelo. Certifique-se de coletar amostras suficientes para cada direção a fim de treinar o modelo de forma eficaz. Os dados coletados são salvos em um arquivo CSV (hand_gesture_data.csv), que é usado para o treinamento.

## Dependências

O projeto depende das seguintes bibliotecas:

* OpenCV
* MediaPipe
* NumPy
* Pandas
* TensorFlow
* Scikit-learn

    Você pode instalar todas as dependências usando:
    ```bash
    pip install -r requirements.txt

## Licença

Este projeto é licenciado sob a Licença MIT - veja o arquivo LICENSE para mais detalhes.