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
    ratings_df=ratings_df.withColumn("rating",col("rating").cast(IntegerType()))    

