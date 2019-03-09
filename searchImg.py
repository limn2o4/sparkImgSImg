import findspark

findspark.init()

from pyspark import SparkConf,SparkContext,sql
from phash import get_pHash
import csv
from io import StringIO
import cv2



if __name__ == '__main__':

    conf = SparkConf().setMaster('local[*]').setAppName("imgSearch")
    sc = SparkContext(conf=conf)
    
    sqlContext = sql.SQLContext(sc)
    img_data_df = sqlContext.read.csv('/home/limn2o4/Documents/Code/SparkImgSImg/img_data.csv')
    img_data = img_data_df.rdd.map(lambda p : (p._c0,p._c1))

    target_img = cv2.imread('/home/limn2o4/Documents/jpg/100503.jpg')
    target_hash = get_pHash(target_img)
    #print(target_hash)
    
    search_hash = sc.broadcast(target_hash)

    def hamming_distance(str1):
        assert(len(str1)==len(search_hash.value))
        return sum([ch1 != ch2 for ch1,ch2 in zip(str1,search_hash.value)])
    
    #print(img_data.take(3))
    dist_rdd = img_data.mapValues(hamming_distance)
    sort_rdd = dist_rdd.sortBy(lambda x : x[1],ascending=True)
    #print(sort_rdd.take(10))
    sort_rdd.saveAsTextFile('result')

    
    
