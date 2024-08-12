import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Função para coletar dados de referência da mão
def collect_data(points, direction, data, labels):
    DIRECTIONS = {'left': 0, 'right': 1, 'up': 2, 'down': 3}
    hand_data = [point.x for point in points.landmark] + [point.y for point in points.landmark]
    data.append(hand_data)
    labels.append(DIRECTIONS[direction])
    return data, labels

# Função para prever a direção do movimento da mão
def predict_direction(points, model, scaler):
    hand_data = [point.x for point in points.landmark] + [point.y for point in points.landmark]
    hand_data = np.array(hand_data).reshape(1, -1)
    hand_data = scaler.transform(hand_data)
    prediction = model.predict(hand_data)
    return np.argmax(prediction)

# Função para treinar o modelo
def train_model(data_path, model_path):
    # Carrega os dados
    data = pd.read_csv(data_path)
    if data.empty:
        print("Erro: Dados para treinamento estão vazios.")
        return None, None

    X = data.drop('label', axis=1)
    y = data['label']

    # Normaliza os dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Divide os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Cria e treina o modelo
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    # Avalia o modelo
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Model accuracy: {accuracy * 100:.2f}%')

    # Salva o modelo treinado
    model.save(model_path)
    return model, scaler

# Função principal
def main():
    # Caminho do modelo e arquivo de dados
    model_path = 'hand_gesture_model.h5'
    data_path = 'hand_gesture_data.csv'

    # Inicia a captura de vídeo
    video = cv2.VideoCapture(0)

    # Inicia o detector de mãos e o desenhador
    mp_hands = mp.solutions.hands.Hands(max_num_hands=1)
    mp_drawing = mp.solutions.drawing_utils

    # Inicializa a posição da bolinha
    ball_position = (100, 100)
    ball_radius = 20

    # Carrega a imagem do labirinto
    labirinto_img = cv2.imread('labirinto.png')
    if labirinto_img is None:
        print("Erro ao carregar a imagem do labirinto")
        return

    # Dados para treinamento
    data = []
    labels = []

    # Verifica se o modelo já existe
    if not os.path.exists(model_path):
        print("Colete dados de referência pressionando 'a' para esquerda, 'd' para direita, 'w' para cima, e 's' para baixo. Pressione 'q' para terminar a coleta.")
        while True:
            _, frame = video.read()
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_hands.process(img_rgb)
            hand_points = results.multi_hand_landmarks
            h, w, _ = frame.shape
            if hand_points:
                for points in hand_points:
                    # Desenha os pontos de referência das mãos
                    mp_drawing.draw_landmarks(frame, points, mp.solutions.hands.HAND_CONNECTIONS)
                    # Coleta dados com base em comandos do teclado
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('a'):
                        data, labels = collect_data(points, 'left', data, labels)
                        print("Coletado: esquerda")
                    elif key == ord('d'):
                        data, labels = collect_data(points, 'right', data, labels)
                        print("Coletado: direita")
                    elif key == ord('w'):
                        data, labels = collect_data(points, 'up', data, labels)
                        print("Coletado: cima")
                    elif key == ord('s'):
                        data, labels = collect_data(points, 'down', data, labels)
                        print("Coletado: baixo")

            # Mostra a imagem na tela
            cv2.imshow('Imagem', frame)

            # Sai do loop se a tecla 'q' for pressionada
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Salva os dados coletados
        if data and labels:
            df = pd.DataFrame(data)
            df['label'] = labels
            df.to_csv(data_path, index=False)

            # Treina o modelo
            model, scaler = train_model(data_path, model_path)
        else:
            print("Nenhum dado foi coletado.")
            return
    else:
        # Carrega o modelo existente
        model = tf.keras.models.load_model(model_path)
        scaler = StandardScaler()

        # Carrega e normaliza os dados para ajustar o scaler
        data = pd.read_csv(data_path)
        X = data.drop('label', axis=1)
        scaler.fit(X)

    # Inicializa a posição da bolinha
    ball_position = (100, 100)
    ball_radius = 20

    while True:
        _, frame = video.read()
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(img_rgb)
        hand_points = results.multi_hand_landmarks
        h, w, _ = frame.shape

        if hand_points:
            for points in hand_points:
                # Desenha os pontos de referência das mãos
                mp_drawing.draw_landmarks(frame, points, mp.solutions.hands.HAND_CONNECTIONS)

                # Prediz a direção do movimento da mão
                direction = predict_direction(points, model, scaler)
                if direction == 0:  # esquerda
                    ball_position = (ball_position[0] - 5, ball_position[1])
                elif direction == 1:  # direita
                    ball_position = (ball_position[0] + 5, ball_position[1])
                elif direction == 2:  # cima
                    ball_position = (ball_position[0], ball_position[1] - 5)
                elif direction == 3:  # baixo
                    ball_position = (ball_position[0], ball_position[1] + 5)

        # Mostra a imagem do labirinto
        labirinto_resized = cv2.resize(labirinto_img, (w, h))
        frame = cv2.addWeighted(frame, 0.5, labirinto_resized, 0.5, 0)

        # Desenha a bolinha
        cv2.circle(frame, ball_position, ball_radius, (0, 255, 0), -1)

        # Mostra a imagem na tela
        cv2.imshow('Labirinto', frame)

        # Sai do loop se a tecla 'q' for pressionada
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
