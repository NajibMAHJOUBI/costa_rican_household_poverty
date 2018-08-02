package fr.poverty.spark.classification.stackingMethod

import fr.poverty.spark.classification.task.DecisionTreeTask
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
import org.apache.spark.sql.{DataFrame, SparkSession}


class StackingMethodDecisionTreeTask(override val classificationMethod: String,
                                     override val pathPrediction: List[String], override val formatPrediction: String,
                                     override val pathTrain: String, override val formatTrain: String,
                                     override val pathSave: String,
                                     override val validationMethod: String,
                                     override val ratio: Double,
                                     override  val idColumn: String,
                                     override val labelColumn: String,
                                     override val predictionColumn: String)
  extends StackingMethodTask(classificationMethod, pathPrediction, formatPrediction, pathTrain, formatTrain, pathSave, validationMethod, ratio, idColumn,
    labelColumn, predictionColumn)
    with StackingMethodFactory {

  val featureColumn: String = "features"
  var model: DecisionTreeClassifier = _
  var modelFit: DecisionTreeClassificationModel = _

  override def run(spark: SparkSession): StackingMethodDecisionTreeTask = {
    val labelFeatures = new StackingMethodTask(classificationMethod, pathPrediction, formatPrediction, pathTrain,
      formatTrain, pathSave, validationMethod, ratio, idColumn, labelColumn, predictionColumn).createLabelFeatures(spark)
    defineModel()
    fit(labelFeatures)
    this
  }

  override def defineModel(): StackingMethodDecisionTreeTask = {
    model = new DecisionTreeTask(labelColumn=labelColumn,
                                     featureColumn=featureColumn,
                                     predictionColumn=predictionColumn).defineModel.getModel
    this
  }

  override def fit(data: DataFrame): StackingMethodDecisionTreeTask = {
    modelFit = model.fit(data)
    this
  }

  override def transform(data: DataFrame): StackingMethodDecisionTreeTask = {
    prediction = modelFit.transform(data)
    this
  }

  override def saveModel(path: String): StackingMethodDecisionTreeTask = {
    model.write.overwrite().save(path)
    this
  }

  def loadModel(path: String): StackingMethodDecisionTreeTask = {
    modelFit = DecisionTreeClassificationModel.load(path)
    this
  }
}
