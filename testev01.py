import cv2
import numpy as np
from inference import get_model
import supervision as sv
import time
import mysql.connector  # Biblioteca para conexão MySQL

# Configurações do banco de dados
db_config = {
    'host': 'xxx.xxx.x.xx',
    'database': 'detect_car',
    'user': 'xxxxxx',
    'password': 'xxxxx'
}

# Função para atualizar o valor no banco de dados
def update_database(detected_value):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        query = "UPDATE sinal SET detected = %s WHERE id = 1"
        cursor.execute(query, (detected_value,))
        conn.commit()
    except mysql.connector.Error as err:
        print("Erro ao conectar ao banco de dados:", err)
    finally:
        cursor.close()
        conn.close()

# Função para capturar os cliques do mouse e salvar os pontos
points = []

def click_event(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        if len(points) == 4:
            cv2.destroyWindow('Selecione a Área')

# Carregar o modelo pré-treinado
model = get_model(api_key="YPn2PFW06xNY8hY7Kc56", model_id="inovatec-semaforo/1")

video = r"C:\Users\faelf\Downloads\inovatec-main\inovatec-main\istockphoto-1810467996-640_adpp_is.mp4"
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

ret, frame = cap.read()
if not ret:
    print("Erro ao capturar o primeiro frame.")
    exit()

cv2.imshow('Selecione a Área', frame)
cv2.setMouseCallback('Selecione a Área', click_event)
cv2.waitKey(0)

if len(points) != 4:
    print("É necessário selecionar exatamente 4 pontos para a ROI.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

roi_contour = np.array(points, dtype=np.int32)
mask = np.zeros_like(frame[:, :, 0])
cv2.fillPoly(mask, [roi_contour], 255)

confidence_threshold = 0.4
nms_threshold = 0.5

car_count = 0
motorcycle_count = 0

signal_state = "red"
last_car_detection_time = 0
signal_duration = 1

while True:
    ret, frame = cap.read()
    if not ret:
        break

    car_count = 0
    motorcycle_count = 0

    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    hsv = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
    adjusted_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
    fg_mask = backSub.apply(adjusted_frame)
    blurred_frame = cv2.GaussianBlur(fg_mask, (5, 5), 0)

    sobelx = cv2.Sobel(blurred_frame, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(blurred_frame, cv2.CV_64F, 0, 1, ksize=5)
    sobel = cv2.sqrt(sobelx**2 + sobely**2)

    sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX)
    sobel = np.uint8(sobel)
    _, thresh = cv2.threshold(sobel, 50, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            if cv2.pointPolygonTest(roi_contour, (x + w // 2, y + h // 2), False) >= 0:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    results = model.infer(masked_frame)
    detections = sv.Detections.from_inference(results[0].dict(by_alias=True, exclude_none=True))
    boxes = detections.xyxy.tolist()
    confidences = detections.confidence.tolist()
    class_names = detections.data['class_name']
    
    boxes_np = np.array(boxes, dtype=np.float32)
    confidences_np = np.array(confidences, dtype=np.float32)
    indices = cv2.dnn.NMSBoxes(boxes_np.tolist(), confidences_np.tolist(), confidence_threshold, nms_threshold)
    
    if len(indices) > 0:
        indices = indices.flatten()
        boxes = [boxes[i] for i in indices]
        confidences = [confidences[i] for i in indices]
        class_names = [class_names[i] for i in indices]
    
        car_count = sum(1 for class_name in class_names if class_name == 'Carro')
        motorcycle_count = sum(1 for class_name in class_names if class_name == 'motorbike')

    current_time = time.time()
    if car_count > 0:
        signal_state = "green"
        last_car_detection_time = current_time
        update_database(1)  # Atualiza para 1 quando carro detectado (sinal verde)
    elif current_time - last_car_detection_time > signal_duration:
        signal_state = "red"
        update_database(0)  # Atualiza para 0 quando nenhum carro é detectado (sinal vermelho)

    annotated_frame = frame.copy()
    for box, confidence, class_name in zip(boxes, confidences, class_names):
        x1, y1, x2, y2 = map(int, box)
        padding = 10
        x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
        x2, y2 = min(frame.shape[1], x2 + padding), min(frame.shape[0], y2 + padding)

        if cv2.pointPolygonTest(roi_contour, ((x1 + x2) // 2, (y1 + y2) // 2), False) >= 0:
            label = f'{class_name} {confidence:.2f}'
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.putText(annotated_frame, f'Carros: {car_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f'Motos: {motorcycle_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    signal_color = (0, 255, 0) if signal_state == "green" else (0, 0, 255)
    cv2.putText(annotated_frame, f'Sinal: {signal_state}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, signal_color, 2)

    cv2.imshow('Detecção de Veículos e Sinal Virtual', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
