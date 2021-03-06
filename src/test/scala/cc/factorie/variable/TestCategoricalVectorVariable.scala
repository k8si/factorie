/* Copyright (C) 2008-2016 University of Massachusetts Amherst.
   This file is part of "FACTORIE" (Factor graphs, Imperative, Extensible)
   http://factorie.cs.umass.edu, http://github.com/factorie
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */
package cc.factorie.variable

import cc.factorie.la.GrowableSparseBinaryTensor1
import org.junit.Test
import org.scalatest.junit._


class TestCategoricalVectorVariable extends JUnitSuite with cc.factorie.util.FastLogging {

  @Test
  def testCategoricalVectorVariable(): Unit = {
    object DocumentDomain extends CategoricalVectorDomain[String]
    class Document extends CategoricalVectorVariable[String] {

      // the value is not set in CategoricalVectorVariable
      set(new GrowableSparseBinaryTensor1(domain.dimensionDomain))(null)

      override def domain: CategoricalVectorDomain[String] = DocumentDomain
    }

    val document = new Document
    document += "hello"
    document += "world"
    document ++= Seq("a", "b", "c")

    println(document.activeCategories.contains("hello"))
  }

}
