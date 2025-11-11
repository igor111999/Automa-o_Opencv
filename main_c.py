'''
Automação de gestos com opencv-python - Professor: Igor Carvalho
Explicação do  projeto
Automação de maquina com captura de imagens pela webcam, após a captura de imagem ele 
verifica qual gesto foi encontrado para executar um comando.
'''
import cv2 #cv2: biblioteca OpenCV para usar a webcam.
import mediapipe as mp #usada para detectar mãos e pontos de referência (landmarks).
import time #time: usado para medir tempo e realizar pausas.
import os #os: usado para executar comandos do sistema (abrir e fechar programas).


mp_hands = mp.solutions.hands #guarda o módulo de detecção de mãos do MediaPipe.
mp_draw = mp.solutions.drawing_utils #fornece funções para desenhar as conexões e landmarks da mão na tela.

hands = mp_hands.Hands(
    max_num_hands=1,           # detecta apenas 1 mão
    min_detection_confidence=0.7, #confiança mínima para considerar uma detecção válida.
    min_tracking_confidence=0.7 #confiança mínima para acompanhar o movimento da mão.
)

cap = cv2.VideoCapture(0) # Inicializa a captura de vídeo

tempoinicial = time.time() # Controle de tempo
tempo_limite = 300  # 5 minutos


def contar_dedos(hand_landmarks): # Função auxiliar para contar dedos levantados
    dedos = [] #lista vazia

    tips_ids = [4, 8, 12, 16, 20] #Índices dos pontos das pontas dos dedos (polegar → mindinho) segundo o padrão do MediaPipe.

#Verifica se o polegar está levantado.
#Compara a posição x da ponta (landmark 4) com o ponto anterior (3).
    if hand_landmarks.landmark[tips_ids[0]].x < hand_landmarks.landmark[tips_ids[0] - 1].x:
        dedos.append(1) #se nao estiver levantado adiciona a lista vazia o proximo dedo a lista
    else:
        dedos.append(0) #caso esteja levantado adiciona o polegar

    # Para os outros dedos
    #o que diferencia é que o dedo polegar e avaliado pelo indice x e os outros
    # a seguir é avaliado pelo indice y
    for id in range(1, 5):
        if hand_landmarks.landmark[tips_ids[id]].y < hand_landmarks.landmark[tips_ids[id] - 2].y:
            dedos.append(1)
        else:
            dedos.append(0)

    return sum(dedos)  # após o reconhecimento, verifica a quantidade de dedos levantados


while True: #cria um loop quando a tela e capturada
    ret, frame = cap.read()
    if not ret: # se não conseguir acesso a camera ele para o codigo
        break

    # Espelha a imagem
    frame = cv2.flip(frame, 1)
    # h- altura
    # w - largura
    # c - pontos na imagem
    h, w, c = frame.shape

    # Converte para RGB (MediaPipe usa RGB)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #cor dos pontos e das linhas do media pipe
    results = hands.process(rgb)

    dedos_levantados = 0 #inicia o codigo com os dedos zerados

    if results.multi_hand_landmarks: #identifica quantas mãos estão levantadas
        for hand_landmarks in results.multi_hand_landmarks: #para cada mão levatada
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)# Desenha as conexões da mão
            dedos_levantados = contar_dedos(hand_landmarks)# Conta os dedos levantados
            cv2.putText(frame, f"Dedos: {dedos_levantados}", (10, 50), #Coloca na tela a quantidade de dedos levantados
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)#Desenha os pontos e linhas da mão sobre o frame.
    # Exibe a imagem
    cv2.imshow("Indentificação de gestos - prof Igor Carvalho", frame)

    # Sai com ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

    # Verifica tempo limite
    if time.time() - tempoinicial >= tempo_limite:
        print("Tempo limite atingido. Encerrando...")
        break

# Libera os recursos
cap.release()
cv2.destroyAllWindows()
