import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model('emnist_model.h5')

# Daftar huruf A-Z (sesuai EMNIST Balanced)
class_names = [chr(i) for i in range(65, 91)]  # 'A' to 'Z'

# Buka kamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip biar gak mirror
    frame = cv2.flip(frame, 1)

    # Tentukan posisi kotak input
    x, y, w, h = 180, 100, 200, 200
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    roi = frame[y:y + h, x:x + w]

    # Preprocess: grayscale, resize, normalisasi
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    normalized = resized / 255.0
    reshaped = normalized.reshape(1, 28, 28, 1)

    # Prediksi huruf
    prediction = model.predict(reshaped, verbose=0)
    predicted_class = np.argmax(prediction)
    predicted_letter = class_names[predicted_class]

    # Tampilkan prediksi
    cv2.putText(frame, f'Prediction: {predicted_letter}', (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Tampilkan hasil kamera
    cv2.imshow("AI Huruf Kamera", frame)

    # Tombol keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan
cap.release()
cv2.destroyAllWindows()
