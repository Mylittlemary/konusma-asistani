import time                                                     # Zaman ölçümü
import cv2                                                      # Kamera ve ekran
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_YOLU = "app/backend/face_landmarker.task"
GOZLER = [
    {"bebek": 468, "kose1": 33,  "kose2": 133},
    {"bebek": 473, "kose1": 362, "kose2": 263},
]
DURUMLAR = ["ORTA", "SOL", "SAG", "YUKARI"]                     # Sırayla ölçülecek bakışlar
HAZIRLIK_SN = 2                                                 # Her durumdan önce hazırlanma süresi
OLCUM_SN = 3                                                    # Her durumda ölçüm süresi


def goz_degerleri(noktalar):
    """Her göz için (yatay, dikey) oranlarını liste olarak döndürür."""
    degerler = []
    for goz in GOZLER:
        bebek = noktalar[goz["bebek"]]
        k1, k2 = noktalar[goz["kose1"]], noktalar[goz["kose2"]]
        sol_x, sag_x = min(k1.x, k2.x), max(k1.x, k2.x)
        genislik = sag_x - sol_x
        if genislik == 0:
            return None
        yatay = (bebek.x - sol_x) / genislik
        dikey = (bebek.y - (k1.y + k2.y) / 2) / genislik
        degerler.append((yatay, dikey))
    return degerler


ayarlar = vision.FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_YOLU),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
)
yuz_bulucu = vision.FaceLandmarker.create_from_options(ayarlar)
kamera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

olcumler = {durum: [] for durum in DURUMLAR}                    # Her durum için toplanan değerler
durum_sirasi = 0
durum_baslangic = time.time()

while durum_sirasi < len(DURUMLAR):
    basarili, kare = kamera.read()
    if not basarili: break
    kare = cv2.flip(kare, 1)

    durum = DURUMLAR[durum_sirasi]
    gecen = time.time() - durum_baslangic                       # Bu durumda geçen süre
    olcuyor_mu = gecen > HAZIRLIK_SN                            # Hazırlık bitti mi?

    rgb = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)
    sonuc = yuz_bulucu.detect_for_video(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(time.time() * 1000))

    if sonuc.face_landmarks and olcuyor_mu:
        degerler = goz_degerleri(sonuc.face_landmarks[0])
        if degerler:
            olcumler[durum].append(degerler)                    # Bu karenin değerlerini kaydet

    yazi = f"{durum} BAK" + ("  (olculuyor)" if olcuyor_mu else "  (hazirlan)")
    cv2.putText(kare, yazi, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)
    cv2.imshow("Kalibrasyon", kare)
    if cv2.waitKey(1) & 0xFF in (ord("q"), 27): break

    if gecen > HAZIRLIK_SN + OLCUM_SN:                          # Bu durum bitti, sıradakine geç
        durum_sirasi += 1
        durum_baslangic = time.time()

kamera.release(); cv2.destroyAllWindows()

# --- Sonuçları özetle ---
print("\nDurum    | Goz1 yatay | Goz2 yatay | Ort. yatay | Ort. dikey | Kare")
ortalar = {}
for durum, liste in olcumler.items():
    if not liste:
        print(f"{durum:<8} | olcum yok")
        continue
    g1 = sum(d[0][0] for d in liste) / len(liste)
    g2 = sum(d[1][0] for d in liste) / len(liste)
    dk = sum((d[0][1] + d[1][1]) / 2 for d in liste) / len(liste)
    ortalar[durum] = ((g1 + g2) / 2, dk)
    print(f"{durum:<8} | {g1:10.3f} | {g2:10.3f} | {(g1 + g2) / 2:10.3f} | {dk:10.3f} | {len(liste)}")

if len(ortalar) == 4:
    print("\nOnerilen esikler (iki durumun ortasi):")
    print(f"SOL_ESIK = {(ortalar['ORTA'][0] + ortalar['SOL'][0]) / 2:.3f}")
    print(f"SAG_ESIK = {(ortalar['ORTA'][0] + ortalar['SAG'][0]) / 2:.3f}")
    print(f"YUKARI_ESIK = {(ortalar['ORTA'][1] + ortalar['YUKARI'][1]) / 2:.3f}")