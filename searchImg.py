#import findspark

#findspark.init()

from pyspark import SparkConf,SparkContext,sql
from pyspark.mllib.clustering import KMeans,KMeansModel
from pyspark.ml.linalg import Vectors
from pyspark import rdd
from phash import get_pHash,get_phash_int
import csv
from io import StringIO
import numpy as np
import cv2


def Hdistance(str1,str2):
        assert(len(str1)==len(str2.value))
        return sum([ch1 != ch2 for ch1,ch2 in zip(str1,str2.value)])/64.0

def mapPoint(data):
    zero_point_str = '0'*64
    return data.map(lambda x : (x[0],Hdistance(x[1],zero_point_str)))
    
if __name__ == '__main__':

    conf = SparkConf().setMaster('local[*]').setAppName("imgSearch")
    sc = SparkContext(conf=conf)
    
    sqlContext = sql.SQLContext(sc)
    img_data_df = sqlContext.read.csv('file:///home/limn2o4/Documents/sparkImgSImg/img_data.csv')
    img_data = img_data_df.rdd.map(lambda p : (p._c0,p._c1))


    points = mapPoint(img_data).cache()

    feat_points =points.map(lambda x : np.array(x[1]))

    model = KMeans.train(feat_points)

    #or load model from file

    target_img = cv2.imread('./3096.jpg')
    target_hash = get_pHash(target_img)
    #print(target_hash)

    target_cluster = model.predict([Hdistance(target_hash,'0'*64)])

    same_cluster_data = points.map(lambda x : (x[0],model.predict([x[1]]))).filter(lambda x : x[1] == target_map_cluster)

    search_hash = sc.broadcast(target_hash)

    def hamming_distance(str1):
        assert(len(str1)==len(search_hash.value))
        return sum([ch1 != ch2 for ch1,ch2 in zip(str1,search_hash.value)])/64.0
    
    #print(img_data.take(3))
    dist_rdd = same_cluster_data.mapValues(hamming_distance)
    sort_rdd = dist_rdd.sortBy(lambda x : x[1],ascending=True)
    print(sort_rdd.take(10))
    #sort_rdd.saveAsTextFile('result')




    
    
