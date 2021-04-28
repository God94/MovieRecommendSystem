package com.wildgoose.dataloader

import com.mongodb.casbah.commons.MongoDBObject
import com.mongodb.casbah.{MongoClient, MongoClientURI}
import org.apache.http.HttpHost
import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.elasticsearch.action.admin.indices.delete.DeleteIndexRequest
import org.elasticsearch.client.indices.{CreateIndexRequest, GetIndexRequest}
import org.elasticsearch.client.{RequestOptions, RestClient, RestHighLevelClient}


case class Movie(val mid: Int, val name: String, val descri: String, val timelong: String, val issue: String, val shoot: String, val language: String, val genres: String, val actors: String, val directors: String)

/*
* 151^Rob Roy (1995)^In the highlands of Scotland in the 1700s, Rob Roy tries to lead his small town to a better future, by borrowing money from the local nobility to buy cattle to herd to market. When the money is stolen, Rob is forced into a Robin Hood lifestyle to defend his family and honour.^139 minutes^August 26, 1997^1995^English ^Action|Drama|Romance|War ^Liam Neeson|Jessica Lange|John Hurt|Tim Roth|Eric Stoltz|Brian Cox|Jason Flemyng|Andrew Keir|Shirley Henderson|Brian McCardie|Gilbert Martin|Vicki Masson|Gilly Gilchrist|Ewan Stewart|David Hayman|David Brooks Palmer|Myra McFadyen|John Murtagh|Karen Matheson|Bill Gardiner|Brian McArthur|Valentine Nwanze|Richard Bonehill|Allan Sutherland|Liam Neeson|Jessica Lange|John Hurt|Tim Roth|Eric Stoltz ^Michael Caton-Jones
* Movies数据集，数据集字段通过^分割
* 151                             电影ID
* Rob Roy(1995)                   电影名称
* In the highLands                电影描述
* 139 minutes                     电影时长
* August 26, 1997                 电影的发行日期
* 1995                            电影的拍摄日期
* English                         电影的语言
* Action/Drama/Romance            电影的类型
* Liam Neeson                     电影的演员
* Michael Caton                   电影的导演
* */


case class Rating(val uid: Int, val mid: Int, val score: Double, val timestamp: Int)

/*
* Rating数据集，用户对于电影的评分数据集，用，分割
* 1                               用户的ID
* 31                              电影的ID
* 2.5                             用户对电影的评分
* 1193435061                      评价时间
* */


case class Tag(val uid: Int, val mid: Int, val tag: String, val timestamp: Int)

/* Tag数据集，用于对于电影的标签数据集，用，分割
* 15                              用户的ID
* 1995                            电影的ID
* dentist                         标签的具体内容
* 1193435061                      打标签的时间
* */


/**
 *
 * @param uri MongoDB的连接
 * @param db  MongoDB要操作的数据库
 */
case class MongoConfig(val uri: String, val db: String)


/**
 *
 * @param httpHosts      Http的主机列表，以，分割
 * @param index          需要操作的索引
 * @param clustername        ES集群的名称
 */
case class ESConfig(val httpHosts: String, val index: String, val clustername: String)


object DataLoader {

  val MOVIE_DATA_PATH = "/Users/petezhang/zhangxiang/idea_projects/MovieRecommendSystem/recommender/dataloader/src/main/resources/movies.csv"
  val RATING_DATA_PATH = "/Users/petezhang/zhangxiang/idea_projects/MovieRecommendSystem/recommender/dataloader/src/main/resources/ratings.csv"
  val TAG_DATA_PATH = "/Users/petezhang/zhangxiang/idea_projects/MovieRecommendSystem/recommender/dataloader/src/main/resources/tags.csv"

  val MONGODB_MOVIE_COLLECTION = "Movie"
  val MONGODB_RATING_COLLECTION = "Rating"
  val MONGODB_TAG_COLLECTION = "Tag"

  val ES_MOVIE_INDEX = ""

  def main(args: Array[String]): Unit = {
    val config = Map(
      "spark.cores" -> "local[*]",
      "mongo.uri" -> "mongodb://linux:27017/recommender",
      "mongo.db" -> "recommender",
      "es.httpHosts" -> "linux:9200",
      "es.index" -> "recommender",
      "es.cluster.name" -> "es-cluster"
    )

    // 创建SparkConf配置
    val sparkConf = new SparkConf().setAppName("Dataloader").setMaster(config.get("spark.cores").get)

    // 创建SparkSession
    val spark = SparkSession.builder().config(sparkConf).getOrCreate()

    import spark.implicits._

    // 将Movie Rating，Tag数据集加载进来
    val movieRDD = spark.sparkContext.textFile(MOVIE_DATA_PATH)

    // 将movieRDD转换为DataFrame
    val movieDF = movieRDD.map(item => {
      val attr = item.split("\\^")
      Movie(attr(0).toInt, attr(1).trim, attr(2).trim, attr(3).trim, attr(4).trim, attr(5).trim, attr(6).trim, attr(7).trim, attr(8).trim, attr(9).trim)
    }).toDF()

    val ratingRDD = spark.sparkContext.textFile(RATING_DATA_PATH)

    // 将ratingRDD转换为DataFrame
    val ratingDF = ratingRDD.map(item => {
      val attr = item.split(",")
      Rating(attr(0).toInt, attr(1).toInt, attr(2).toDouble, attr(3).toInt)
    }).toDF()

    val tagRDD = spark.sparkContext.textFile(TAG_DATA_PATH)

    // 将tagRDD转换为DataFrame
    val tagDF = tagRDD.map(item => {
      val attr = item.split(",")
      Tag(attr(0).toInt, attr(1).toInt, attr(2).trim, attr(3).toInt)
    }).toDF()

    implicit val mongoConfig = MongoConfig(config.get("mongo.uri").get, config.get("mongo.db").get)

    // 将数据保存到MongoDB中
    storeDataInMongoDB(movieDF, ratingDF, tagDF)

    // 首先需要将Tag数据集进行处理，处理后的形式为MID，tag1|tag2|tag3

    val newTag = tagDF.groupBy($"mid").agg(concat_ws("|", collect_set($"tag")).as("tags")).select("mid", "tags")

    // 需要将处理后的Tag数据，和Movie数据融合，产生新的Movie数据
    val movieWithTagsDF = movieDF.join(newTag, Seq("mid", "mid"), "left").select(movieDF("*"), newTag("tags"))

    implicit val esConfig = ESConfig(config.get("es.httpHosts").get, config.get("es.index").get, config.get("es.cluster.name").get)

    // 将数据保存到ES
    storeDataInES(movieWithTagsDF)

    // 关闭spark
    spark.stop()
  }

  // 将数据保存到MongoDB中的方法，柯里化
  def storeDataInMongoDB(movieDF: DataFrame, ratingDF: DataFrame, tagDF: DataFrame)(implicit mongoConfig: MongoConfig): Unit = {

    // 新建一个MongoDB的连接
    val mongoClient = MongoClient(MongoClientURI(mongoConfig.uri))

    // 如果MongoDB中有对应的数据库，那么应该删除
    mongoClient(mongoConfig.db)(MONGODB_MOVIE_COLLECTION).dropCollection()
    mongoClient(mongoConfig.db)(MONGODB_RATING_COLLECTION).dropCollection()
    mongoClient(mongoConfig.db)(MONGODB_TAG_COLLECTION).dropCollection()

    // 将当前数据写入到MongoDB
    movieDF
      .write
      .option("uri", mongoConfig.uri)
      .option("collection", MONGODB_MOVIE_COLLECTION)
      .mode("overwrite")
      .format("com.mongodb.spark.sql")
      .save()

    ratingDF
      .write
      .option("uri", mongoConfig.uri)
      .option("collection", MONGODB_RATING_COLLECTION)
      .mode("overwrite")
      .format("com.mongodb.spark.sql")
      .save()

    tagDF
      .write
      .option("uri", mongoConfig.uri)
      .option("collection", MONGODB_TAG_COLLECTION)
      .mode("overwrite")
      .format("com.mongodb.spark.sql")
      .save()

    // 对数据表建立索引
    mongoClient(mongoConfig.db)(MONGODB_MOVIE_COLLECTION).createIndex(MongoDBObject("mid" -> 1))
    mongoClient(mongoConfig.db)(MONGODB_RATING_COLLECTION).createIndex(MongoDBObject("uid" -> 1))
    mongoClient(mongoConfig.db)(MONGODB_RATING_COLLECTION).createIndex(MongoDBObject("mid" -> 1))
    mongoClient(mongoConfig.db)(MONGODB_TAG_COLLECTION).createIndex(MongoDBObject("uid" -> 1))
    mongoClient(mongoConfig.db)(MONGODB_TAG_COLLECTION).createIndex(MongoDBObject("mid" -> 1))

    // 关闭MongoDB的连接
    mongoClient.close()
  }

  // 将数据保存到ES中的方法
  def storeDataInES(movieDF: DataFrame)(implicit esConfig: ESConfig): Unit = {

    // 新建一个ES的客户端
    val esClient = new RestHighLevelClient(
      RestClient.builder(
        new HttpHost(esConfig.httpHosts.split(":")(0), esConfig.httpHosts.split(":")(1).toInt, "http")
      )
    )

    // 需要清除掉ES中遗留的数据
    if (esClient.indices().exists(new GetIndexRequest(esConfig.index), RequestOptions.DEFAULT)){
      esClient.indices().delete(new DeleteIndexRequest(esConfig.index), RequestOptions.DEFAULT)
    }
    esClient.indices().create(new CreateIndexRequest(esConfig.index), RequestOptions.DEFAULT)

    // 将数据写入到ES中
    movieDF
      .write
      .option("es.nodes", esConfig.httpHosts.split(":")(0))
      .option("es.mapping.id", "mid")
      .mode("overwrite")
      .option("es.port", esConfig.httpHosts.split(":")(1).toInt)
      .format("org.elasticsearch.spark.sql")
      .save(esConfig.index+"/"+ES_MOVIE_INDEX)

    esClient.close()
  }
}
