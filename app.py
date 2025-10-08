#web ile model arasındaki bağlantı için flask modelini kullanacağız


import os 
import json
from flask import Flask, render_template #flask web sunucumuzun temelini oluşturacak , render_template ise python verilerini html dosyalarına gömmek için kullanacağız 
from pyspark.sql import SparkSession #spark'a bağlanmak için  kullanacağız
from  pyspark.ml.recommendation import ALSModel # eğitilmiş als modelini yüklemek için kullanacağız


app=Flask(__name__) #flask uygulamasını başlatıyoruz
spark= SparkSession.builder.appName("TavsiyeSunucusu").getOrCreate() #spark session başlatma

model_path="als_model" # eğitilmiş modelin yolu

movies_parquert_path="movies_parquet" # filmler verisetinin yolu

model= ALSModel.load(model_path) # modeli yükleme

movies_df =spark.read.parquet(movies_parquert_path) # filmler verisetini yükleme  model bize 500 511 gibi film id leri verecek bizde buların karşılı olan film isimlerini kulanabilmek için filmler verisetini yüklüyoruz

@app.route("/") #anasayfa için route belirleme kullanıcı anasayfaya geldiğinde index fonksiyonu çalışacak

def index():
    return "Film tavsiyesi almak için Adres çubuğuna /recommend/<kullanıcı_id> yazınız" # kullanıcıya anasayfada ne yapması gerektiğini söylüyoruz

@app.route("/recommend/<int:user_id>") # kullanıcı id sine göre tavsiye almak için route belirleme alınan user_id yi alttaki fonksiyona aktarır
def get_recommendations (user_id):

    # kullanıcı id sine göre tavsiye alma fonksiyonu
    user_df=spark.createDataFrame([(user_id)],["userId"]) # kullanıcı id sini spark dataframe e çevirme

    recommendations=model.recommendForUserSubset(user_df,10) # modelden kullanıcı id sine göre 10 tane film tavsiyesi alma

    recommendations_df=recommendations.select("recomamendations.movieId","recomamendations.rating").first().asDict() # göstermek istediğimiz tavsiyeleri dataframe den alıp listeye çevirme
    
    
    recs_list=[]
    if(recommendations_df and recommendations_df.get('movieId')): # eğer model bir sonuç döndürdüyse ve id si varsa
       
       recommended_movies_df=spark.CreateDataFrame(zip(recommendations_df["movieId"],recommendations_df["rating"]),["movieId","rating"]) # tavsiye edilen film id leri ve puanlarını dataframe e çevirme birleştirme işlemi 
       
       final_recommendations=recommended_movies_df.join(movies_df,"movieId") # tavsiye edilen film id leri ile filmler verisetini birleştirip film isimlerini alma
    
       recs_list=final_recommendations.toJson.map(lambda j:json.loads(j)).collect() # dataframe i json formatına çevirme ve listeye çevirme
       
    return render_template("recommendations.html",user_id=user_id,recommendations=recs_list) # tavsiyeleri html dosyasına gömme ve kullanıcı id sini de gönderme
    

if __name__=="__main__":
    app.run(debug=True,port=5001) # uygulamayı debug modunda 5001 portunda çlıştırıyoruz . çalıştırma hata ayıklama için











       