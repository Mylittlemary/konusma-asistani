import time                                                     # Zaman damgası almak için
import cv2                                                      # Kamera ve görüntü işleme
import mediapipe as mp                                          # Yüz noktası tespiti ("mp" kısa adıyla)
from mediapipe.tasks import python as mp_python                 # MediaPipe'ın genel görev ayarları
from mediapipe.tasks.python import vision                       # Görüntüyle ilgili görevler

MODEL_YOLU = "app/backend/face_landmarker.task"                 # İndirdiğimiz hazır model dosyası

# --- Yüz noktası bulucuyu hazırla (döngüden önce, bir kez) ---
ayarlar = vision.FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_YOLU),  # Hangi model kullanılacak
    running_mode=vision.RunningMode.VIDEO,                      # Görüntüler video karesi olarak gelecek
    num_faces=1,                                                # Sadece bir yüz ara (hasta tek kişi)
)
yuz_bulucu = vision.FaceLandmarker.create_from_options(ayarlar)  # Ayarlara göre bulucuyu oluştur

kamera = cv2.VideoCapture(0, cv2.CAP_DSHOW)                     # Kamerayı aç (Windows DirectShow)

while True:
    basarili, kare = kamera.read()                              # Kameradan bir kare al
    if not basarili: break                                      # Kare yoksa döngüden çık
    kare = cv2.flip(kare, 1)                                    # Ayna görüntüsü

    # --- MediaPipe'a kareyi ver ---
    rgb = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)                 # OpenCV BGR verir, MediaPipe RGB ister
    mp_goruntu = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)  # MediaPipe'ın anladığı biçim
    zaman_ms = int(time.time() * 1000)                          # Şu anki zaman (milisaniye)
    sonuc = yuz_bulucu.detect_for_video(mp_goruntu, zaman_ms)   # Yüz noktalarını bul

    # --- Bulunan noktaları ekrana çiz ---
    if sonuc.face_landmarks:                                    # En az bir yüz bulunduysa
        noktalar = sonuc.face_landmarks[0]                      # İlk (tek) yüzün noktaları: bir liste
        yukseklik, genislik, _ = kare.shape                     # Karenin boyutları (yükseklik, genişlik, renk)
        for nokta in noktalar:                                  # Listedeki her nokta için
            x = int(nokta.x * genislik)                         # 0-1 arası konumu piksele çevir
            y = int(nokta.y * yukseklik)
            cv2.circle(kare, (x, y), 1, (0, 255, 0), -1)        # O noktaya küçük yeşil bir daire çiz
        cv2.putText(kare, f"Nokta sayisi: {len(noktalar)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)  # Sol üste nokta sayısını yaz

    cv2.imshow("Yuz Noktalari", kare)                           # Kareyi pencerede göster
    tus = cv2.waitKey(1) & 0xFF                                 # Basılan tuşu al
    if tus in (ord("q"), ord("Q"), 27): break                   # q, Q veya Esc ile çık
    if cv2.getWindowProperty("Yuz Noktalari", cv2.WND_PROP_VISIBLE) < 1: break  # X ile kapatıldıysa çık

kamera.release(); cv2.destroyAllWindows()                       # Kamerayı ve pencereleri kapat