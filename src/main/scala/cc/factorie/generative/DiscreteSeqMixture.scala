/* Copyright (C) 2008-2010 University of Massachusetts Amherst,
   Department of Computer Science.
   This file is part of "FACTORIE" (Factor graphs, Imperative, Extensible)
   http://factorie.cs.umass.edu, http://code.google.com/p/factorie/
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

package cc.factorie.generative
import cc.factorie._
import scala.reflect.Manifest
import scala.collection.mutable.{HashSet,HashMap}
import scala.util.Random

trait PlatedDiscreteMixtureGeneratingFamily extends PlatedDiscreteGeneratingFamily /*with MixtureFamily*/ {
  type FamilyType <: DiscreteGeneratingFamily //with MixtureFamily
}

object PlatedDiscreteMixture extends PlatedDiscreteMixtureGeneratingFamily with GenerativeFamilyWithStatistics3[PlatedGeneratedDiscreteVar,Mixture[Proportions],PlatedGate] {
  def gate(f:Factor) = throw new Error("Not yet implemented. Need to make PlatedGate be a Gate?") // f._3
  def pr(s:Stat): Double = pr(s._1, s._2, s._3)
  def pr(ds:Seq[DiscreteValue], mixture:Seq[IndexedSeq[Double]], gates:Seq[DiscreteValue]): Double = ds.zip(gates).map(tuple => mixture(tuple._2.intValue).apply(tuple._1.intValue)).product
  override def logpr(s:Stat): Double = logpr(s._1, s._2, s._3)
  def logpr(ds:Seq[DiscreteValue], mixture:Seq[IndexedSeq[Double]], gates:Seq[DiscreteValue]): Double = ds.zip(gates).map(tuple => math.log(mixture(tuple._2.intValue).apply(tuple._1.intValue))).sum  
  def sampledValue(s:Stat): Seq[DiscreteValue] = sampledValue(s._1.first.domain, s._2, s._3)
  def sampledValue(d:DiscreteDomain, mixture:Seq[ProportionsValue], gates:Seq[DiscreteValue]): Seq[DiscreteValue] = 
    for (i <- 0 until gates.length) yield d.getValue(mixture(gates(i).intValue).sampleInt) 
  def prChoosing(s:StatisticsType, mixtureIndex:Int): Double = throw new Error
  def sampledValueChoosing(s:StatisticsType, mixtureIndex:Int): ChildType#Value = throw new Error
  def prValue(s:StatisticsType, value:Int, index:Int): Double = throw new Error
}