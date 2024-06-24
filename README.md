Bu projenin temel amacı kütüphanelerin getirdiği avantajlardan yararlanmadan bir yapay sinir ağı modeli oluşturarak çarpım tahmini yapabilmesini sağlamaktadır.

Yalnızca Numpy kütüphanesi kullanılmış ve bu projede kullanılan matris çarpımları, transpoz işlemleri , rastgele sayı üretilmesini ve diğer
matematiksel operasyonları çok karmaşık hale getirmeden kullanılabilmesini sağlamıştır.

Sigmoid Aktivasyon Fonksiyonu:

Geriye doğru yayılım (backpropagation) sırasında gradyanlar üzerinde çalışabilmemizi sağlar.
Sigmoid fonksiyonu, girişin büyük veya küçük olması durumunda çıkışın hassas bir şekilde değişmesi eğilimindedir.

Veri Seti:

1 ila 10 arasındaki iki sayının çarpım sonuçlarından oluşan bir
çarpım tablosu veri setini oluşturur. Bu veri seti, yapay sinir ağını
eğitmek ve test etmek için kullanılabilir.
İki iç içe döngü (for döngüsü) kullanılarak, 1 ila 10 arasındaki iki
sayının çarpım sonuçlarından oluşan örnekler ([i/100, j/100,
(i*j)/100]) data listesine eklenir.
Veri setine sayılar yüze bölünerek eklenmesi ile normalizasyon
işlemi gerçekleştirilir. 
Oluşturulan liste, bir NumPy dizisine dönüştürülür.
Bu şekilde elde edilen veri seti, her bir satırında çarpanlar (i, j) ve
çarpım sonuçlarını içeren örneklerden oluşur. Örneğin, i=2 ve
j=3 için, [0.02, 0.03, 0.06] örneği elde edilir. Bu veri seti, 
yapay sinir ağının öğrenme sürecinde kullanılabilecek girdi (input) ve çıkış (output) değerlerini içerir.


Eğitim parametreleri:


learning_rate: Öğrenme oranıdır. Bu, ağırlıkların
güncellenme adımlarında kullanılan bir hiperparametredir ve
ağın öğrenme hızını kontrol eder.
epochs: Eğitim iterasyonlarının sayısını belirler. Model, veri
setini bu sayıda kez geçerek öğrenme sürecini tamamlar.

Adam optimizer algoritmasının hiper parametreleri:

beta1: Momentum terimi için “exponential decay” faktörüdür. Bu değer,
momentum teriminin güncellenmesinde kullanılır. Tipik olarak 0.9 gibi
bir değer seçilir.

beta2: İkinci moment terimi için “exponential decay” faktörüdür. Bu
değer, ikinci moment teriminin güncellenmesinde kullanılır. 

epsilon: Bölme işleminin sıfıra bölme hatası almaması için kullanılan
bir küçük değerdir. Epsilon, genellikle 1e-8 gibi küçük bir değer olarak
ayarlanır.

Ağırlık Değerleri:


m_weights_input_hidden: İlk ağırlık seti için momentum
terimi. Gradient değerlerinin hareketli ortalamasıdır.
v_weights_input_hidden: İlk ağırlık seti için ikinci moment
terimi. Karelerinin hareketli ortalamasıdır.
m_weights_hidden_output: İkinci ağırlık seti için momentum
terimi. Gradient değerlerinin hareketli ortalamasıdır.
v_weights_hidden_output: İkinci ağırlık seti için ikinci
moment terimi. Karelerinin hareketli ortalamasıdır.

İleri Yayılım (Forward Propagation):


Giriş verisi (train_data[:, :2]) ile gizli katman arasındaki ağırlıklar
(weights_input_hidden) kullanılarak gizli katmana giriş hesaplanır
(hidden_layer_input).
Sigmoid aktivasyon fonksiyonu (sigmoid()) kullanılarak gizli katmanın
çıkışı (hidden_layer_output) hesaplanır.
Gizli katmanın çıkışı, çıkış katmanındaki ağırlıklar
(weights_hidden_output) kullanılarak çıkış katmanına giriş hesaplanır
(output_layer_input).
Yine sigmoid aktivasyon fonksiyonu kullanılarak çıkış katmanının çıkışı
(predicted_output) hesaplanır.


Hata Hesaplama:
Tahmin edilen çıkış ile gerçek çıkış arasındaki hata (error) hesaplanır.
Geri Yayılım (Backpropagation):
Çıkış Katmanı Hatasının Hesaplanması: İlk olarak, çıkış katmanındaki
hata hesaplanır. Bu, tahmin edilen çıktı ile gerçek çıktı arasındaki farktır.
Hata, error değişkenine atanır.

Çıkış Katmanı Hatasının Geriye Yayılması: Çıkış katmanındaki hata, çıkış
katmanı ile gizli katman arasındaki ağırlıkların transpozunu kullanarak
geriye doğru yayılır. Bu adımda, çıkış katmanındaki hata, gizli katmanın
hatalarını hesaplamak için kullanılır. output_error adlı değişkende
tutulur.

Gizli Katman Hatasının Hesaplanması: Daha sonra, gizli katmanın
hataları hesaplanır. Bu adımda, çıkış katmanı ile gizli katman arasındaki
hata kullanılarak gizli katmanın hataları hesaplanır.
hidden_layer_error adlı değişkende tutulur.

Ağırlıkların Güncellenmesi (Hidden-Output): Hesaplanan gizli katman
hataları ve çıkış katmanı ile gizli katman arasındaki ağırlıkların
güncellenmesi gerçekleştirilir. Bu adımda, Adam optimizer algoritması
kullanılarak ağırlıklar güncellenir.

Ağırlıkların Güncellenmesi (Input-Hidden): Gizli katmanın hataları
hesaplandıktan sonra, giriş katmanı ile gizli katman arasındaki ağırlıklar
güncellenir

Test:

test_inputs: Test veri setindeki giriş değerlerini temsil eder.
test_data[:, :2] ifadesi ile test veri setindeki her örnek için ilk iki
sütun alınarak giriş değerleri oluşturulur.
hidden_layer_test: Gizli katmandaki çıkışları temsil eder.
np.dot(test_inputs, weights_input_hidden) ifadesi ile test
girişleri ve eğitilen girişten gizli katmana giden ağırlıkların iç çarpımı
hesaplanır. Bu çıkışlar, daha sonra sigmoid aktivasyon fonksiyonu
(sigmoid()) ile geçirilir.
output_layer_test: Çıkış katmanındaki çıkışları temsil eder.
np.dot(hidden_layer_test, weights_hidden_output) ifadesi ile
gizli katmandan çıkışa giden ağırlıkların iç çarpımı hesaplanır. Bu çıkışlar
da sigmoid aktivasyon fonksiyonundan geçirilir.
Sonuç olarak, hidden_layer_test ve output_layer_test ifadeleri, test
veri seti üzerindeki girişlerin sinir ağından geçirilerek elde edilen gizli
katman ve çıkış katman çıkışlarını temsil eder



