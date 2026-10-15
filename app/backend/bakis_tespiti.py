import time                                                     # Zaman ölçümü
from collections import deque                                   # Son N değeri tutan liste (yumuşatma için)
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_YOLU = "app/backend/face_landmarker.task"

# --- MediaPipe nokta numaraları ---
GOZLER = [
    {"bebek": 468, "kose1": 33,  "kose2": 133, "ust": 159, "alt": 145},
    {"bebek": 473, "kose1": 362, "kose2": 263, "ust": 386, "alt": 374},
]

# --- Ayarlar ---
KALIBRASYON_DURUMLARI = ["ORTA", "SOL", "SAG", "YUKARI"]
HAZIRLIK_SN = 2        # Her durumdan önce hazırlanma süresi
OLCUM_SN = 2           # Her durumda ölçüm süresi
YUMUSATMA_KARE = 5     # Son kaç karenin ortalaması alınacak (titremeyi azaltır)
SECIM_KARE = 20        # Bir yön kaç kare (~0.7 sn) sabit kalırsa SEÇİLMİŞ sayılır
ESIK_ORANI = 0.6       # Eşik, ortadan uç değere doğru yolun yüzde kaçında (0.5 = tam orta)
BASLAMA_BEKLEME_SN = 2 # Kalibrasyondan sonra takipten önce ortaya bakma süresi


def olcum(noktalar):
    """İki gözün ortalaması olarak (yatay oran, göz açıklığı) döndürür."""
    yataylar, acikliklar = [], []
    for goz in GOZLER:
        bebek = noktalar[goz["bebek"]]
        k1, k2 = noktalar[goz["kose1"]], noktalar[goz["kose2"]]
        sol_x, sag_x = min(k1.x, k2.x), max(k1.x, k2.x)
        genislik = sag_x - sol_x
        if genislik == 0:
            return None
        yataylar.append((bebek.x - sol_x) / genislik)                 # Göz bebeği gözün neresinde (0-1)
        aciklik = noktalar[goz["alt"]].y - noktalar[goz["ust"]].y      # Alt ve üst göz kapağı arası
        acikliklar.append(aciklik / genislik)                          # Göz genişliğine oranla
    return sum(yataylar) / len(yataylar), sum(acikliklar) / len(acikliklar)


def esikleri_hesapla(ort):
    """Kalibrasyon ortalamalarından eşikleri hesaplar (ortadan uca doğru ESIK_ORANI kadar)."""
    orta_yatay, orta_aciklik = ort["ORTA"]
    return {
        "sol": orta_yatay + ESIK_ORANI * (ort["SOL"][0] - orta_yatay),
        "sag": orta_yatay + ESIK_ORANI * (ort["SAG"][0] - orta_yatay),
        "yukari": orta_aciklik + ESIK_ORANI * (ort["YUKARI"][1] - orta_aciklik),  # Yukarı bakınca göz daha çok açılır
    }


def yon_bul(yatay, aciklik, esik):
    """Ölçümlere ve eşiklere bakarak bakış yönünü döndürür."""
    if aciklik > esik["yukari"]:
        return "YUKARI"
    if yatay < esik["sol"]:
        return "SOL"
    if yatay > esik["sag"]:
        return "SAG"
    return "ORTA"


def yaz(kare, metin, y, boyut=1.0, renk=(0, 255, 255), kalinlik=2):
    """Karenin sol tarafına yazı yazar."""
    cv2.putText(kare, metin, (20, y), cv2.FONT_HERSHEY_SIMPLEX, boyut, renk, kalinlik)


# --- Hazırlık ---
ayarlar = vision.FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_YOLU),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
)
yuz_bulucu = vision.FaceLandmarker.create_from_options(ayarlar)
kamera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

mod = "kalibrasyon"                                     # Önce kalibrasyon, sonra takip
durum_no, durum_baslangic = 0, time.time()
toplanan = {d: [] for d in KALIBRASYON_DURUMLARI}
esik = None
son_olcumler = deque(maxlen=YUMUSATMA_KARE)
aday_yon, aday_sayac, secilen = None, 0, "-"
secime_hazir = True                                     # Seçimden sonra ortaya dönülünce tekrar True olur
takip_baslangic = 0.0

while True:
    basarili, kare = kamera.read()
    if not basarili: break
    kare = cv2.flip(kare, 1)

    rgb = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)
    sonuc = yuz_bulucu.detect_for_video(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(time.time() * 1000))
    deger = olcum(sonuc.face_landmarks[0]) if sonuc.face_landmarks else None

    if mod == "kalibrasyon":
        # ---------- KALİBRASYON MODU ----------
        durum = KALIBRASYON_DURUMLARI[durum_no]
        gecen = time.time() - durum_baslangic
        olcuyor = gecen > HAZIRLIK_SN
        if olcuyor and deger:
            toplanan[durum].append(deger)
        yaz(kare, f"{durum} BAK" + ("  (olculuyor)" if olcuyor else "  (hazirlan)"), 60, 1.2)
        yaz(kare, "Kalibrasyon", 100, 0.7, (255, 255, 255))

        if gecen > HAZIRLIK_SN + OLCUM_SN:                      # Bu durum bitti
            durum_no += 1
            durum_baslangic = time.time()
            if durum_no == len(KALIBRASYON_DURUMLARI):          # Tüm durumlar bitti
                ort = {d: (sum(v[0] for v in liste) / len(liste), sum(v[1] for v in liste) / len(liste))
                       for d, liste in toplanan.items() if liste}
                if len(ort) == len(KALIBRASYON_DURUMLARI):
                    esik = esikleri_hesapla(ort)
                    mod = "takip"
                    takip_baslangic = time.time()
                    secime_hazir, aday_yon, aday_sayac = True, None, 0
                    print("Kalibrasyon tamam. Esikler:",
                          {k: round(v, 3) for k, v in esik.items()})
                    if ort["YUKARI"][1] <= ort["ORTA"][1]:
                        print("Uyari: Yukari bakista goz acikligi artmadi; yukari tespiti zayif olabilir.")
                else:                                           # Bir durumda yüz bulunamadı, baştan al
                    print("Bazi durumlarda yuz bulunamadi, kalibrasyon yeniden basliyor.")
                    durum_no, durum_baslangic = 0, time.time()
                    toplanan = {d: [] for d in KALIBRASYON_DURUMLARI}

    else:
        # ---------- TAKİP MODU ----------
        if time.time() - takip_baslangic < BASLAMA_BEKLEME_SN:     # Kalibrasyondan hemen sonra bekle
            yaz(kare, "ORTAYA BAK, basliyor...", 70, 1.2)
        elif deger:
            son_olcumler.append(deger)                          # Son 5 ölçümü tut
            yatay = sum(v[0] for v in son_olcumler) / len(son_olcumler)    # Yumuşatılmış yatay
            aciklik = sum(v[1] for v in son_olcumler) / len(son_olcumler)  # Yumuşatılmış açıklık
            yon = yon_bul(yatay, aciklik, esik)

            if yon == aday_yon:                                 # Aynı yöne bakmaya devam ediyor
                aday_sayac += 1
            else:                                               # Yön değişti, saymaya baştan başla
                aday_yon, aday_sayac = yon, 1
            if yon == "ORTA":                                   # Ortaya dönüldü: yeni seçime hazır
                secime_hazir = True
            if aday_sayac == SECIM_KARE and yon != "ORTA" and secime_hazir:   # Yeterince uzun baktı: SEÇİM
                secilen = yon
                secime_hazir = False                            # Ortaya dönmeden yeni seçim yok
                print(f"SECIM: {yon}")

            yaz(kare, yon, 70, 2, (0, 255, 255), 4)
            if yon != "ORTA" and secime_hazir:                  # Seçim ilerleme çubuğu
                oran = min(aday_sayac / SECIM_KARE, 1)
                cv2.rectangle(kare, (20, 90), (20 + int(300 * oran), 110), (0, 255, 0), -1)
                cv2.rectangle(kare, (20, 90), (320, 110), (255, 255, 255), 2)
            yaz(kare, f"Secilen: {secilen}" + ("" if secime_hazir else "  (ortaya don)"), 150, 0.9, (0, 255, 0))
            yaz(kare, f"yatay: {yatay:.3f}  aciklik: {aciklik:.3f}", 185, 0.6, (255, 255, 255))
        else:
            yaz(kare, "Yuz bulunamadi", 70, 1, (0, 0, 255))
        yaz(kare, "c: yeniden kalibre et   q: cikis", kare.shape[0] - 20, 0.6, (255, 255, 255), 1)

    cv2.imshow("Bakis Tespiti", kare)
    tus = cv2.waitKey(1) & 0xFF
    if tus in (ord("q"), ord("Q"), 27):
        break
    if tus in (ord("c"), ord("C")):                             # Yeniden kalibrasyon
        mod, durum_no, durum_baslangic = "kalibrasyon", 0, time.time()
        toplanan = {d: [] for d in KALIBRASYON_DURUMLARI}
        son_olcumler.clear()
    if cv2.getWindowProperty("Bakis Tespiti", cv2.WND_PROP_VISIBLE) < 1:
        break

kamera.release(); cv2.destroyAllWindows()