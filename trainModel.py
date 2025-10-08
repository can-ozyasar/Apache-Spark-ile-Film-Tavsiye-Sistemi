import os 
from pyspark.sql import SparkSession  #spark'a bağlanmak için  kullanacağız 
from pyspark.sql.functions import col # veri stun işlemleri için kullacağız
from pyspark.ml.recommendation import ALS #als modeli için 
from pyspark.sql.types import IntegerType ,FloatType # veri tiplerini değiştirmek için kullanacağız


def main():# tüm kodlarımızı bu fonksiyonun içine yazacağız 
    #spark session başlatma
    spark = SparkSession.builder \
        .appName("ModelEgitimi") \
        .getOrCreate()
    
    #veri setini okuma
    data_path = "ml-latest-small/"

    #spark dataframe olarak veri setini okuma movies ve ratings i 

    ratings_df=spark.read.csv(data_path +"ratings.csv",header=True,inferSchema=True)
    # veri setinde  ilk satırda veri isimleri var ise header=True ile belirleriz
    # inferSchema=True ile spark otomatik olarak veri tiplerini belirler sayısalmı text mi anlar 
    
    movies_df=spark.read.csv(data_path +"movies.csv",header=True,inferSchema=True)

    ratings_df=ratings_df.withColumn("userId",col("userId").cast(IntegerType()))    
    ratings_df=ratings_df.withColumn("movieId",col("movieId").cast(IntegerType()))
    ratings_df=ratings_df.withColumn("rating",col("rating").cast(FloatType()))


    # Model seçimi İÇİN En güçlü ve en popüler algoritmalardan biri olan ALS (Alternating Least Squares) algoritmasını kullanacağız.
    # ALS, özellikle büyük veri setlerinde ve seyrek matrislerde iyi performans göster
    
    
    ##?#  temel mantık sena benzer zevkte insanların beğendiği filmleri sana da önerecek     
    

    als=ALS(
        maxIter=10,# max kaç iterasyon çalışacağını belirler öğreneme aşamasında veri üzerinden en fazla 10 defa geçsin demektir
        regParam=0.1, # regularization parametresi overfitting i önlemek için kullanılır aşırı öğrenirse sadece o veri seti ile çalışacak seviyeye gelir ve yeni verilerde başarısız olur
        userCol="userId", # kullanıcı id sini belirler
        itemCol="movieId", # film id sini belirler  
        ratingCol="rating", # kullanıcıların filmlere verdiği puanları belirler
        coldStartStrategy="drop" #soğuk başlangıç, modelin eğitim sırasında görmediği kullanıcılar veya filmler için tahmin yaparken hata vermemesi için kullanılır
        # eğer ilk defa karşılaşırsa filmle yada bir kullanıcıya o kaydı drop yap diyoruz

    )

    #MODELİ EĞİTME 
    model =als.fit(ratings_df)
    model_path="als_model"    # modeli eğitmek zaman alan bir iş olduğu için her tavsiye için tekrar modeli eğitmek yerine bir defa eğitip kaydetmek daha mantıklı
    model.write().overwrite().save(model_path) # modeli kaydettik



    movies_parquet_path="movies_parquet" # movies verisetini parquet formatında kaydedeceğiz daha hızlı okuma ve yazma işlemi için
    movies_df.write.mode("overwrite").parquet(movies_parquet_path) # overwrite var ise üzerine yaz yoksa oluştur demek  

    # ekrana model başarıllı şekilde eğitildi yazdırma
    print(f"Model başarıyla eğitildi ve {model_path} , klasörüne kaydedildi.")
    print(f"filmler verisi{movies_parquet_path} klasörüne  kaydedildi")

    spark.stop() # spark session ı kapatıyoruz





# main fonksiyonu çalıştırma
if __name__ == "__main__":
    main()






