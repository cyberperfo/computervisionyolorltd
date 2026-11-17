# 5. Test Süreçleri, Veri Sızıntısının Önlenmesi ve Etik Değerlendirme

## 5.1 Test süreçleri

Deney hattının doğruluğu, sonuçların geçerliliğinin ön koşuludur. Bu nedenle hat, her biri belirli bir hata sınıfını hedefleyen dört katmanlı bir test süreciyle doğrulanmıştır. Testlerin büyük kısmı notebook içinde `assert` ifadeleri olarak uygulanmıştır. Bu sayede bir test başarısız olduğunda deney başlamaz; testler sonradan yapılan bir kontrol olarak kalmaz, hattın parçası olarak çalışır.

| Katman | Test | Başarı ölçütü | Başarısızlıkta davranış |
|---|---|---|---|
| Veri bütünlüğü | Her görüntü için etiket dosyası var mı; sınıf indeksleri $[0,C)$ aralığında mı | Etiketsiz görüntü arka plan olarak açıkça işaretlenir | Eksik etiket boş dosyayla temsil edilir ve sayılır |
| Veri bütünlüğü | Bölmelere göre sınıf dağılımı tablosu | Her sınıf her bölmede temsil edilir | Görsel inceleme; gerekirse yeni `split_seed` |
| Sızıntı denetimi | Bölmeler arası ortak dosya yolu, MD5, grup kimliği | Kesişim kümeleri boş | `AssertionError`, deney durur |
| Sızıntı denetimi | Bölmeler arası en küçük dHash Hamming mesafesi | $\min d_H>\tau=5$ | `AssertionError`, deney durur |
| Türetilmiş küme tutarlılığı | Her bozuk test kümesinin dosya adları temiz test kümesiyle aynı mı | Küme eşitliği | `AssertionError` |
| Türetilmiş küme tutarlılığı | Bozulmaların görsel sağlaması | Her koşul aynı görüntü üzerinde gösterilir | Görsel inceleme |
| Tekrarlanabilirlik | Tohum sabitleme, deterministik cuDNN, dondurulmuş manifest | Aynı tohumla aynı bozulma örneklemi; aynı ayrım | Manifest varsa ayrım yeniden üretilmez |
| Tekrarlanabilirlik | Ortam kaydı (`environment.json`) | Sürümler ve donanım kayıtlı | — |
| Ölçüm geçerliliği | FPS için ısınma, GPU senkronizasyonu, medyan | Aynı mimaride senaryolar arası FPS farkı anlamsız | Tablo 4.2'de doğrulanmıştır ($p_{Holm}=0{,}909$) |
| Ölçüm geçerliliği | Değerlendirme yalnızca `split="test"` ile, yalnızca `best.pt` üzerinden | Eğitim YAML'ında test yolu bulunmaz | Yapısal olarak engellenir |

*Tablo 5.1. Deney hattının test süreci.*

İki test özellikle dikkate değerdir. Birincisi, **eğitim yapılandırma dosyasının test kümesinin yolunu hiç içermemesidir**. Bu tasarım, eğitim sırasında test verisine erişimi bir kural olarak değil, yapısal bir imkânsızlık olarak garanti eder. İkincisi, **FPS ölçümünün kendi kendini doğrulamasıdır**. Bozulma artırması mimariyi değiştirmediği için, aynı mimarinin iki senaryosu arasındaki FPS farkının sıfırdan ayırt edilememesi gerekir. Tablo 4.2'de bu beklenti doğrulanmıştır. Bu sonuç, ölçüm protokolünün sistematik bir yanlılık içermediğini ve gözlenen 1–2 FPS'lik değişimin ortam gürültüsü düzeyinde olduğunu gösterir.

Bu test sürecinin kapsamadığı iki alan vardır. Birincisi, mAP hesaplaması kütüphanenin kendi uygulamasına bırakılmış ve bağımsız bir uygulamayla (örneğin `pycocotools`) çapraz doğrulanmamıştır. İkincisi, bozulma fonksiyonları için piksel düzeyinde birim testleri yazılmamış, doğrulama görsel incelemeyle yapılmıştır.

## 5.2 Veri sızıntısının önlenmesi

Kapoor ve Narayanan'ın [11] sınıflandırması temel alınarak, bu çalışma için geçerli altı sızıntı türü belirlenmiştir. Tablo 5.2, her tür için alınan önlemi ve önlemin **gerçekte ne ölçüde doğrulandığını** göstermektedir.

| Sızıntı türü | Risk | Alınan önlem | Durum |
|---|---|---|---|
| **S1.** Bölmeler arası birebir kopya | Aynı görüntünün farklı adla hem eğitim hem testte bulunması | MD5 özeti ile gruplama | Doğrulandı (denetim: 0 ortak MD5) |
| **S2.** Bölmeler arası yakın kopya | Yeniden boyutlandırılmış, sıkıştırılmış sürümler [13] | dHash + Hamming ≤ 5, Union-Find gruplama | Doğrulandı (denetim: $\min d_H>5$) |
| **S3.** Ön işleme / artırma sızıntısı | Bozulma ve artırmaların bölmeden önce uygulanması | Tüm bozulmalar ayrımdan sonra, yalnızca ilgili bölmeden türetildi | Yapısal olarak garanti edildi |
| **S4.** Test kümesinin model seçiminde kullanılması | Epoch ve hiperparametre seçiminin test performansına göre yapılması | `best.pt` yalnızca val uygunluğuyla seçildi; hiperparametreler kütüphane varsayılanları | Yapısal olarak garanti edildi |
| **S5.** Sahne / çekim dizisi sızıntısı | Aynı sahneden ardışık karelerin farklı bölmelere düşmesi [2] | İsteğe bağlı `group_regex` ile sahne gruplama | **Kısmen.** Bu veri kümesinde sahne bilgisi bulunmadığından kullanılmadı |
| **S6.** Ön-eğitim sızıntısı | Test görüntülerinin COCO ön-eğitim verisinde bulunması | COCO'dan türetilmemiş bir veri kümesi seçildi | **Kısmen.** COCO ile görüntü düzeyinde örtüşme kontrol edilmedi |

*Tablo 5.2. Sızıntı türleri, önlemler ve doğrulama durumu.*

Tabloda "kısmen" olarak işaretlenen iki risk ayrıca açıklanmalıdır.

**S5 — Sahne sızıntısı.** dHash, yalnızca neredeyse aynı görüntüleri yakalar. Aynı sahnenin kısa aralıklarla çekilmiş farklı kareleri, örneğin bir hayvan sürüsünün ardışık fotoğrafları, 5 bitten fazla farklılık gösterebilir ve farklı bölmelere düşebilir. Kamera tuzağı verilerinde bu risk özellikle yüksektir; Beery ve arkadaşları [2], kamera konumuna göre ayrım yapılmadığında genelleme performansının ciddi biçimde abartıldığını göstermiştir. Kullanılan veri kümesi kamera ya da konum bilgisi içermediği için bu çalışmada konum tabanlı ayrım yapılamamıştır. Ayrıca dHash yatay çevirmeye karşı değişmez değildir; bir görüntünün aynalanmış kopyası tespit edilemez. Bu risk, çevrilmiş görüntünün özetinin de hesaplanmasıyla kolayca giderilebilir.

**S6 — Ön-eğitim sızıntısı.** İki model de COCO [12] üzerinde ön-eğitilmiş ağırlıklarla başlatılmıştır. COCO'nun 80 sınıfı arasında *fil* ve *zebra* da bulunmaktadır. Kullanılan veri kümesi COCO'dan türetilmemiştir, ancak web kaynaklı görüntülerden oluştuğu için bazı test görüntülerinin COCO eğitim kümesinde de yer alma olasılığı sıfır değildir. Bu olasılık, test görüntülerinin dHash özetlerinin COCO train2017 içindeki fil ve zebra görüntüleriyle karşılaştırılmasıyla test edilebilir. Bu kontrol bu çalışmada yapılmamıştır ve bir sınırlılık olarak kabul edilmektedir.

**Test kümesinin "tüketilmesi".** Bu rapordaki test sonuçları görüldükten sonra senaryo, hiperparametre ya da model seçiminde yapılacak her değişiklik, test kümesini dolaylı olarak model seçim sürecine katar. Bu nedenle tam bütçeli deneyde yalnızca önceden tanımlanmış yapılandırma (`QUICK_MODE = False`) değiştirilmeli, başka bir ayar test sonuçlarına bakılarak yapılmamalıdır. Bu kurala uyulamazsa yeni ve hiç kullanılmamış bir test ayrımı oluşturulmalı ya da sonuçlar açıkça "keşif amaçlı" olarak raporlanmalıdır.

## 5.3 Etik ve veri lisansı değerlendirmesi

### 5.3.1 Lisanslar ve kullanım koşulları

Tablo 5.3, çalışmada kullanılan yazılım, model ağırlığı ve verilerin lisans durumunu özetlemektedir.

| Bileşen | Lisans | Bu çalışma için sonuç |
|---|---|---|
| Ultralytics kütüphanesi (YOLOv8, RT-DETR uygulaması) [5] | AGPL-3.0 | Akademik kullanım serbesttir. Kütüphaneyi kullanan kod dağıtılır ya da ağ hizmeti olarak sunulursa kaynak kodunun AGPL-3.0 uyumlu bir lisansla paylaşılması gerekir. Ticari kullanım için Ultralytics'in kurumsal lisansı gerekir. |
| YOLOv8m ve RT-DETR-L ön-eğitimli ağırlıkları | Ultralytics dağıtımı, AGPL-3.0 | Yukarıdakiyle aynı. Özgün RT-DETR uygulaması ayrıca Apache-2.0 lisanslıdır. |
| COCO (ön-eğitim verisi) [12] | Etiketler CC BY 4.0; görüntüler Flickr kullanım koşullarına tabi | Ağırlıklar dolaylı olarak bu veriden türetilmiştir; atıf yapılmıştır. |
| African Wildlife veri kümesi | **Teslimden önce doğrulanmalıdır** | Ultralytics tarafından dağıtılmaktadır. Görüntülerin özgün kaynağı ve lisansı, veri kümesinin yayım sayfasından doğrulanmalı ve buraya yazılmalıdır. Web kaynaklı görüntüler ayrı telif haklarına tabi olabilir. Bu nedenle görüntüler raporda ve sunumda yalnızca akademik alıntı sınırları içinde kullanılmalı ve yeniden dağıtılmamalıdır. |
| PyTorch, NumPy, OpenCV, SciPy, Matplotlib | BSD / Apache-2.0 / benzeri izin verici lisanslar | Kısıtlama yoktur. |
| Bu çalışmanın kodu | Ultralytics'e bağımlı olduğu için AGPL-3.0 uyumlu olmalıdır | Kod paylaşılacaksa depoya bir `LICENSE` dosyası eklenmelidir. |

*Tablo 5.3. Kullanılan bileşenlerin lisans durumu.*

Veri kümesinin belgelenmesi için *Datasheets for Datasets* [20] çerçevesi önerilir. Bu çerçeve veri kümesinin kaynağını, toplama sürecini, onay durumunu ve bilinen yanlılıklarını tek bir belgede toplar. Bu raporun ekine kısa bir veri sayfası eklenmesi önerilmektedir.

### 5.3.2 Hassas bilgi: nesli tehlike altındaki türlerin konumu

Veri kümesindeki sınıflardan **gergedan**, kaçak avcılığın en yoğun hedef aldığı türlerdendir; kara gergedan kritik tehlike altındadır. Doğa fotoğrafları ve kamera tuzağı görüntüleri sıklıkla EXIF meta verisinde GPS koordinatı taşır. Bu tür verilerin yayımlanması, hayvanların konumunu kaçak avcılara açık edebilir. Deney hattı görüntüleri `shutil.copy2` ile kopyalamakta ve bu nedenle meta veriyi korumaktadır. Bu nedenle:

- Görüntüler paylaşılmadan, sunulmadan ya da yayımlanmadan önce EXIF meta verisi (özellikle GPS alanları) temizlenmelidir.
- Gerçek bir kamera tuzağı veri kümesiyle çalışılırsa, kamera konumları rapor ve kod deposunda yer almamalıdır.

### 5.3.3 Temsil ve yanlılık

Kullanılan görüntüler web kaynaklıdır. Bu tür fotoğraflar, nesnenin genellikle kare merkezinde, büyük ve gündüz ışığında çekildiği bir **fotoğrafçı yanlılığı** taşır. Gerçek kamera tuzağı görüntülerinde ise hayvanlar sıklıkla kısmen görünür, küçük, gece kızılötesi ışıkla ya da hareket hâlinde görüntülenir. Ayrıca bu çalışmadaki zorlu koşullar **sentetiktir** (Bölüm 3.7) ve gerçek düşük ışık ya da sis dağılımının yalnızca bir yaklaşığıdır. Bu nedenle:

- Raporlanan dayanıklılık sonuçları, gerçek saha koşullarına doğrudan genellenmemelidir.
- Sahada kullanım düşünülüyorsa, sonuçlar gerçek zorlu koşul verileriyle (ör. ExDark [9]) ve konum bazında ayrılmış kamera tuzağı verileriyle [2] doğrulanmalıdır.

### 5.3.4 Sorumlu kullanım ve çift kullanım

Bu çalışmadaki sınıflar yalnızca hayvan türlerinden oluşur ve insan tespiti içermez. Bununla birlikte, aynı teknik hat değiştirilmeden insan tespiti ve gözetim amacıyla kullanılabilir. Kaçak avcılıkla mücadele gibi meşru uygulamalar da insanların izlenmesini içerebilir. Bu kullanımlar mahremiyet, orantılılık ve yerel topluluklar üzerindeki etkiler açısından ayrıca değerlendirilmelidir.

Tespit hatalarının maliyeti de asimetriktir. Koruma uygulamasında kaçırılan bir tespit (yanlış negatif), gereksiz bir alarmdan (yanlış pozitif) genellikle daha pahalıdır. Bölüm 4.5'te gösterildiği gibi modeller düşük ışık ve hareket bulanıklığında önemli ölçüde başarısız olmaktadır. Bu nedenle bu modellere dayalı bir sistem, kararların **insan denetiminde** verildiği bir karar destek aracı olarak tasarlanmalıdır.

### 5.3.5 Çevresel maliyet

Derin öğrenme deneylerinin enerji tüketimi ve karbon ayak izi raporlanmalıdır [21], [22]. Tüketilen enerji yaklaşık olarak

$$
E\;[\mathrm{kWh}]=P_{GPU}\,[\mathrm{kW}]\times t\,[\mathrm{h}]\times \mathrm{PUE} \tag{5.1}
$$

ile tahmin edilebilir. Burada $P_{GPU}$ GPU'nun güç tüketimi (Tesla T4 için en fazla 70 W), $t$ toplam GPU süresi ve PUE veri merkezinin güç kullanım verimliliğidir. Örnek olarak, toplam 24 GPU-saatlik bir deney, $\mathrm{PUE}=1{,}1$ varsayımıyla en fazla yaklaşık $0{,}07\times24\times1{,}1\approx1{,}85$ kWh enerji tüketir. Gerçek değer, her çalıştırmanın süresi kaydedilerek hesaplanmalı ve nihai raporda verilmelidir. Bu çalışmada aynı ayrımın ve bozulmuş test kümelerinin tüm çalıştırmalar arasında yeniden kullanılması ve tamamlanmış çalıştırmaların atlanması, gereksiz hesaplamayı azaltan tasarım tercihleridir.

### 5.3.6 Bilimsel dürüstlük

- Çalıştırılan **tüm** tohumlar ve senaryolar raporlanmıştır. En iyi tohum seçilmemiş, olumsuz sonuçlar (bozulma artırmasının başarısızlığı) gizlenmemiştir.
- Çalıştırılmayan ablasyonlar tablolarda boş bırakılmış, tahmini değerle doldurulmamıştır.
- Ön deney niteliğindeki sonuçlar özet, giriş ve sonuç bölümlerinde açıkça bu şekilde nitelendirilmiştir.
- Desteklenmeyen iddialar (ör. küçük nesne performansı) Bölüm 4.5.6'da açıkça listelenmiştir.
- **[Kurumunuzun kuralına göre doldurun:]** Bu raporun hazırlanmasında yapay zekâ destekli araçlar kod ve metin taslağı için kullanılmıştır; tüm sonuçlar, sayılar ve iddialar yazar(lar) tarafından doğrulanmıştır.

# 6. Sonuç

Bu çalışmada YOLOv8m ve RT-DETR-L, sızıntısız bir veri ayrımı, beş zorlu görüntüleme koşulu ve üç tohumlu istatistiksel bir protokolle hız–doğruluk ödünleşimi açısından karşılaştırılmıştır. Kısa eğitim bütçeli ön deneylerde YOLOv8m yaklaşık 2,2 kat daha hızlı olmuştur. RT-DETR-L ise tüm tohumlarda ve tüm koşullarda daha doğru ve göreli olarak daha dayanıklı bulunmuştur. İki model de hız–doğruluk Pareto sınırı üzerinde yer almıştır; dolayısıyla mimari seçimi uygulamanın gecikme bütçesine bağlıdır. Hareket bulanıklığı iki mimari için de en zor koşul olmuş, başarısızlık ağırlıklı olarak kutu sınırlarının yanlış yerleştirilmesi biçiminde ortaya çıkmıştır. Sentetik bozulma artırması bu bütçede dayanıklılığı iyileştirmemiş, eğitimde görülmeyen bozulmalarda ise olumsuz yönde bir eğilim göstermiştir.

Doğruluk farkının büyüklüğünün önemli ölçüde yetersiz eğitimden kaynaklandığı değerlendirildiğinden, bu bulgular kesin bir mimari üstünlük iddiası olarak değil, az veri ve az hesaplama rejimindeki davranışın bir ölçümü olarak yorumlanmalıdır. Gelecek çalışmalar şunları içermelidir:

- Tam bütçeli eğitim ve en az beş tohum.
- Çözünürlük ve mozaik ablasyonlarının tamamlanması.
- Boyuta ve sınıfa göre ayrıştırılmış hata analizi [19].
- Gerçek zorlu koşul veri kümeleriyle [9] ve konum bazında ayrılmış kamera tuzağı verileriyle [2] doğrulama.
- TensorRT ile optimize edilmiş dağıtım senaryosunda hız ölçümü.

# Kaynakça

[1] M. S. Norouzzadeh, A. Nguyen, M. Kosmala, A. Swanson, M. S. Palmer, C. Packer, and J. Clune, "Automatically identifying, counting, and describing wild animals in camera-trap images with deep learning," *Proc. Nat. Acad. Sci. USA*, vol. 115, no. 25, pp. E5716–E5725, Jun. 2018.

[2] S. Beery, G. Van Horn, and P. Perona, "Recognition in terra incognita," in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, Munich, Germany, 2018, pp. 456–473.

[3] C. Michaelis, B. Mitzkus, R. Geirhos, E. Rusak, O. Bringmann, A. S. Ecker, M. Bethge, and W. Brendel, "Benchmarking robustness in object detection: Autonomous driving when winter is coming," 2019, *arXiv:1907.07484*.

[4] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, "You only look once: Unified, real-time object detection," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Las Vegas, NV, USA, 2016, pp. 779–788.

[5] G. Jocher, A. Chaurasia, and J. Qiu, "Ultralytics YOLOv8," version 8.0.0, 2023. [Online]. Available: https://github.com/ultralytics/ultralytics

[6] N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, and S. Zagoruyko, "End-to-end object detection with transformers," in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, 2020, pp. 213–229.

[7] Y. Zhao, W. Lv, S. Xu, J. Wei, G. Wang, Q. Dang, Y. Liu, and J. Chen, "DETRs beat YOLOs on real-time object detection," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Seattle, WA, USA, 2024, pp. 16965–16974.

[8] D. Hendrycks and T. Dietterich, "Benchmarking neural network robustness to common corruptions and perturbations," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, New Orleans, LA, USA, 2019.

[9] Y. P. Loh and C. S. Chan, "Getting to know low-light images with the Exclusively Dark dataset," *Comput. Vis. Image Understand.*, vol. 178, pp. 30–42, Jan. 2019.

[10] S. Kaufman, S. Rosset, C. Perlich, and O. Stitelman, "Leakage in data mining: Formulation, detection, and avoidance," *ACM Trans. Knowl. Discov. Data*, vol. 6, no. 4, Dec. 2012, Art. no. 15.

[11] S. Kapoor and A. Narayanan, "Leakage and the reproducibility crisis in machine-learning-based science," *Patterns*, vol. 4, no. 9, Sep. 2023, Art. no. 100804.

[12] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, and C. L. Zitnick, "Microsoft COCO: Common objects in context," in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, Zurich, Switzerland, 2014, pp. 740–755.

[13] B. Barz and J. Denzler, "Do we train on test data? Purging CIFAR of near-duplicates," *J. Imaging*, vol. 6, no. 6, Jun. 2020, Art. no. 41.

[14] H. Rezatofighi, N. Tsoi, J. Gwak, A. Sadeghian, I. Reid, and S. Savarese, "Generalized intersection over union: A metric and a loss for bounding box regression," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Long Beach, CA, USA, 2019, pp. 658–666.

[15] Z. Zheng, P. Wang, W. Liu, J. Li, R. Ye, and D. Ren, "Distance-IoU loss: Faster and better learning for bounding box regression," in *Proc. AAAI Conf. Artif. Intell.*, vol. 34, no. 7, 2020, pp. 12993–13000.

[16] X. Li, W. Wang, L. Wu, S. Chen, X. Hu, J. Li, J. Tang, and J. Yang, "Generalized focal loss: Learning qualified and distributed bounding boxes for dense object detection," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 33, 2020, pp. 21002–21012.

[17] H. Zhang, Y. Wang, F. Dayoub, and N. Sünderhauf, "VarifocalNet: An IoU-aware dense object detector," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2021, pp. 8514–8523.

[18] I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, New Orleans, LA, USA, 2019.

[19] D. Bolya, S. Foley, J. Hays, and J. Hoffman, "TIDE: A general toolbox for identifying object detection errors," in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, 2020, pp. 558–573.

[20] T. Gebru, J. Morgenstern, B. Vecchione, J. W. Vaughan, H. Wallach, H. Daumé III, and K. Crawford, "Datasheets for datasets," *Commun. ACM*, vol. 64, no. 12, pp. 86–92, Dec. 2021.

[21] E. Strubell, A. Ganesh, and A. McCallum, "Energy and policy considerations for deep learning in NLP," in *Proc. 57th Annu. Meeting Assoc. Comput. Linguistics (ACL)*, Florence, Italy, 2019, pp. 3645–3650.

[22] A. Lacoste, A. Luccioni, V. Schmidt, and T. Dandres, "Quantifying the carbon emissions of machine learning," 2019, *arXiv:1910.09700*.
