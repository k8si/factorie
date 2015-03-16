package cc.factorie.app.nlp.ner

/**
 * @author Kate Silverstein 
 *         created on 3/16/15
 */

import cc.factorie.app.strings._
import cc.factorie.variable.CategoricalVectorVar
import cc.factorie.app.nlp.{Document, Token}
import org.scalatest.FlatSpec
import cc.factorie.app.nlp.load.LoadConll2003

//ConllChainNer but with fewer features (just for testing purposes)
class ConllChainNer2(url: java.net.URL=null) extends ChainNer[BilouConllNerTag](BilouConllNerDomain, (t, s) => new BilouConllNerTag(t, s), l => l.token, url) {
  override def process(document:Document): Document = {
    if (document.tokenCount > 0) {
      val doc = super.process(document)
      doc.attr.+=(new ConllNerSpanBuffer ++= document.sections.flatMap(section => BilouConllNerDomain.spanList(section)))
      doc
    } else document
  }
  override def addFeatures(document: Document, vf: Token => CategoricalVectorVar[String]): Unit = {
    for (token <- document.tokens) {
      val features = vf(token)
      val rawWord = token.string
      val word = simplifyDigits(rawWord).toLowerCase
      features += s"W=$word"
      features += s"SHAPE=${cc.factorie.app.strings.stringShape(rawWord, 2)}"
      if (token.isPunctuation) features += "PUNCTUATION"
      if (word.length < 5){
        features += "P="+cc.factorie.app.strings.prefix(word, 4)
        features += "S="+cc.factorie.app.strings.suffix(word, 4)
      }
    }
  }
}

class TestNerTaggers extends FlatSpec {
  val trainFilename = this.getClass.getResource("/ner-train-input").getPath
  val testFilename = this.getClass.getResource("/ner-test-input").getPath
  "ConllChainNer" should "train and tag properly" in {
    implicit val random = new scala.util.Random(0)
    val trainDocs = LoadConll2003(BILOU=true).fromFilename(trainFilename)
    val testDocs = LoadConll2003(BILOU=true).fromFilename(testFilename)
    assert(trainDocs.length > 0 && testDocs.length > 0, "no train and/or test docs loaded")
    (trainDocs ++ testDocs).foreach{ doc => doc.sections.flatMap(_.tokens).foreach{ token => assert(token.attr.contains(classOf[LabeledBilouConllNerTag]), "token with no LabeledBilouConllNerTag") }}
    val ner = new ConllChainNer2
    val result = ner.train(trainDocs, testDocs)
    testDocs.foreach(ner.process)
    testDocs.foreach(doc => doc.sections.flatMap(_.tokens).foreach{ token => assert(token.attr.contains(classOf[BilouConllNerTag]), "processed token with no BilouConllNerTag") })
  }
}
