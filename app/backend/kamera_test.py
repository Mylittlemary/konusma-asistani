import cv2 #OpenCV kütüphanesini kullanacağımızı söylüyoruz.
kamera= cv2.VideoCapture(0)#Bilgisayarın kamerasını aç.0=İlk kamera
while True: #Durana kadar sürekli tekrarla
    basarili,kare=kamera.read() #kamereadan bir kare al.Başarılı:alındı mı?(True/False)
    if not basarili:break #Kare alınmadıysa görüntüden çık.
    kare=cv2.flip(kare,1)#Görüntüyü yatay çevir (ayna gibi görünsün)
    cv2.imshow("Kamera",kare)# Kareyi "Kamera" adlı pencerede göster.
    if cv2.waitKey(1)& 0xFF ==ord("q"):break #1 ms klavyeyi dinle.Q ya basıldıysa çık.
kamera.release();cv2.destroyAllWindows()#Kamerayı serbest bırak.pencereyi kapat