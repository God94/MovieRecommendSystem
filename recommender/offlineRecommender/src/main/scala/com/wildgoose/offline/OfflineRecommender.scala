package com.wildgoose.offline

import org.apache.spark.SparkConf
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.sql.SparkSession
import org.jblas.DoubleMatrix


case class MongoConfig(uri: String, db: String)

case class Movie(val mid: Int, val name: String, val descri: String, val timelong: String, val issue: String, val shoot: String, val language: String, val genres: String, val actors: String, val directors: String)

case class MovieRating(val uid: Int, val mid: Int, val score: Double, val timestamp: Int)

// 推荐
case class Recommendation(rid: Int, r: Double)

// 用户的推荐
case class UserRecs(uid: Int, recs: Seq[Recommendation])

// 电影的相似度
case class MovieRecs(mid: Int, recs: Seq[Recommendation])


object OfflineRecommender {

  val MONGODB_RATING_COLLECTION = "Rating"
  val MONGODB_MOVIE_COLLECTION = "Movie"
  val USER_MAX_RECOMMENDDATION = 20
  val USER_RECS = "UserRecs"
  val MOVIE_RECS = "MovieRecs"

  def main(args: Array[String]): Unit = {

    val config = Map(
      "spark.cores" -> "local[*]",
      "mongo.uri" -> "mongodb://linux:27017/recommender",
      "mongo.db" -> "recommender"
    )

    // 创建一个SparkConf配置
    val sparkConf = new SparkConf().setAppName("OfflineRecommender").setMaster(config("spark.cores"))
      .set("spark.executor.memory", "6G").set("spark.driver.memory", "3G")

    // 基于SparkConf创建一个SparkSession
    val spark = SparkSession.builder().config(sparkConf).getOrCreate()

    // 读取MongoDB中的业务数据
    val mongoConfig = MongoConfig(config("mongo.uri"), config("mongo.db"))

    import spark.implicits._

    // 训练ALS模型
    val ratingRDD = spark
      .read
      .option("uri", mongoConfig.uri)
      .option("collection", MONGODB_RATING_COLLECTION)
      .format("com.mongodb.spark.sql")
      .load()
      .as[MovieRating]
      .rdd
      .map(rating => (rating.uid, rating.mid, rating.score))

    // 用户数据集 RDD[Int]
    val userRDD = ratingRDD.map(_._1).distinct()

    // 电影数据集 RDD[Int]
    val movieRDD = spark
      .read
      .option("uri", mongoConfig.uri)
      .option("collection", MONGODB_MOVIE_COLLECTION)
      .format("com.mongodb.spark.sql")
      .load()
      .as[Movie]
      .rdd
      .map(_.mid)

    // 创建训练数据集
    val trainData = ratingRDD.map(x => Rating(x._1, x._2, x._3))

    val (rank, iterations, lambda) = (50, 10, 0.01)
    // 训练ALS模型
    val model = ALS.train(trainData, rank, iterations, lambda)

    // 计算用户推荐矩阵
    // 需要构造一个usersProducts RDD[(Int, Int)]
    val userMovies = userRDD.cartesian(movieRDD)

    val preRatings = model.predict(userMovies)

    val userRecs = preRatings.map(rating => (rating.user, (rating.product, rating.rating)))
      .groupByKey()
      .map {
        case (uid, recs) => UserRecs(uid, recs.toList.sortWith(_._2 > _._2).take(USER_MAX_RECOMMENDDATION).map(x => Recommendation(x._1, x._2)))
      }.toDF()

    userRecs
      .write
      .option("collection", USER_RECS)
      .option("uri", mongoConfig.uri)
      .mode("overwrite")
      .format("com.mongodb.spark.sql")
      .save()

    // 计算电影相似度矩阵
    val movieFeatures = model.productFeatures.map {
      case (mid, features) =>
        (mid, new DoubleMatrix(features))
    }

    val movieRecs = movieFeatures.cartesian(movieFeatures)
      .filter { case (a, b) => a._1 != a._2 }
      .map {
        case (a, b) =>
          val simScore = consinSim(a._2, b._2)
          (a._1, (b._1, simScore))
      }
      .filter(_._2._2 > 0.6)
      .groupByKey()
      .map {
        case (mid, items) =>
          MovieRecs(mid, items.toList.map(x => Recommendation(x._1, x._2)))
      }.toDF()

    movieRecs
        .write
        .option("uri", mongoConfig.uri)
        .option("collection", MOVIE_RECS)
        .format("com.mongodb.spark.sql")
        .save()

    // 关闭spark
    spark.stop

  }

  def consinSim(movie1: DoubleMatrix, movie2: DoubleMatrix): Double = {
    movie1.dot(movie2) / (movie1.norm2() * movie2.norm2())
  }

}
