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

package cc.factorie

import scala.collection.mutable.ArrayBuffer
import scala.math
import java.util.Arrays

// Variables for dealing with sequences

/** A trait for setting the member type ElementType in SeqVar classes.
    @author Andrew McCallum */
trait ElementType[+ET] {
  type ElementType = ET
}

/** A trait with many of the same methods as Seq, but not actually a Seq itself  
    because Seq defines "equals" based on same contents, 
    but all variables must have equals based on identity.
    @author Andrew McCallum */
trait SeqSimilar[+E] extends Iterable[E] with ElementType[E] {
  def value: Seq[E]
  // Some of the methods of Seq, for convenience
  def length: Int = value.length
  def apply(index:Int): E = value.apply(index)
  def iterator: Iterator[E] = value.iterator
  def map[B](f:E=>B): Seq[B] = value.map(f)
  def contains(elem: Any): Boolean = value.contains(elem)
  def indexWhere(p:E => Boolean, from: Int): Int = value.indexWhere(p, from)
  def indexWhere(p:E => Boolean): Int = value.indexWhere(p)
  def indexOf[B>:E](elem:B): Int = value.indexOf(elem)
  def indexOf[B>:E](elem:B, from:Int): Int = value.indexOf(elem, from)
  // Methods overridden from Iterable
  override def toSeq: Seq[E] = value // TODO Should we also have "asSeq"? -akm
  override def foreach[U](f:(E)=>U): Unit = value.foreach(f)
  override def head: E = value.head
  override def last: E = value.last
  override def exists(f:E=>Boolean): Boolean = value.exists(f)
}

/** A trait with many of the same methods as IndexedSeq, but not actually a IndexedSeq itself  
    because Seq defines "equals" based on same contents, 
    but all variables must have equals based on identity.
    @author Andrew McCallum */
trait IndexedSeqSimilar[+E] extends SeqSimilar[E] {
  override def value: IndexedSeq[E]
  override def map[B](f:E=>B): IndexedSeq[B] = value.map(f)
  override def toSeq: IndexedSeq[E] = value
}

/** An abstract variable whose value is a Seq[E].  
    Note that this trait itself does not actually inherit from Seq[E] 
    because Seq defines "equals" based on same contents, 
    but all variables must have equals based on identity.
    @author Andrew McCallum */
trait SeqVar[+E] extends Var with ValueBound[Seq[E]] with SeqSimilar[E]


/** An abstract variable whose value is an IndexedSeq[E].  
    Note that this trait itself does not actually inherit from IndexedSeq[E] 
    because Seq defines "equals" based on same contents, 
    but all variables must have equals based on identity.
    @author Andrew McCallum */
trait IndexedSeqVar[+E] extends SeqVar[E] with ValueBound[IndexedSeq[E]] with IndexedSeqSimilar[E]


/** An abstract variable containing a mutable sequence of other variables.  
    This variable stores the sequence itself, and tracks changes to the contents and order of the sequence. 
    @author Andrew McCallum */
trait MutableSeqVar[E] extends IndexedSeqVar[E] with MutableVar[IndexedSeq[E]] {
  type Element = E
  protected val _seq = new ArrayBuffer[Element] // TODO Consider using an Array[] instead so that SeqVar[Int] is more efficient.
  @inline final def value: IndexedSeq[Element] = _seq // Note that for efficiency we don't return a copy, but this means that this value could change out from under a saved "value" if this variable value is changed. 
  def set(newValue:Value)(implicit d:DiffList): Unit = { _seq.clear; _seq ++= newValue }
  def update(seqIndex:Int, x:Element)(implicit d:DiffList): Unit = UpdateDiff(seqIndex, x)
  def append(x:Element)(implicit d:DiffList) = AppendDiff(x)
  def prepend(x:Element)(implicit d:DiffList) = PrependDiff(x)
  def trimStart(n:Int)(implicit d:DiffList) = TrimStartDiff(n)
  def trimEnd(n: Int)(implicit d:DiffList) = TrimEndDiff(n)
  def remove(n:Int)(implicit d:DiffList) = Remove1Diff(n)
  def swap(i:Int,j:Int)(implicit d:DiffList) = Swap1Diff(i,j)
  def swapLength(pivot:Int,length:Int)(implicit d:DiffList) = for (i <- pivot-length until pivot) Swap1Diff(i,i+length)
  abstract class SeqVariableDiff(implicit d:DiffList) extends AutoDiff {override def variable = MutableSeqVar.this}
  case class UpdateDiff(i:Int, x:Element)(implicit d:DiffList) extends SeqVariableDiff {val xo = _seq(i); def undo = _seq(i) = xo; def redo = _seq(i) = x}
  case class AppendDiff(x:Element)(implicit d:DiffList) extends SeqVariableDiff {def undo = _seq.trimEnd(1); def redo = _seq.append(x)}
  case class PrependDiff(x:Element)(implicit d:DiffList) extends SeqVariableDiff {def undo = _seq.trimStart(1); def redo = _seq.prepend(x)}
  case class TrimStartDiff(n:Int)(implicit d:DiffList) extends SeqVariableDiff {val s = _seq.take(n); def undo = _seq prependAll (s); def redo = _seq.trimStart(n)}
  case class TrimEndDiff(n:Int)(implicit d:DiffList) extends SeqVariableDiff {val s = _seq.drop(_seq.length - n); def undo = _seq appendAll (s); def redo = _seq.trimEnd(n)}
  case class Remove1Diff(n:Int)(implicit d:DiffList) extends SeqVariableDiff {val e = _seq(n); def undo = _seq.insert(n,e); def redo = _seq.remove(n)}
  case class Swap1Diff(i:Int,j:Int)(implicit d:DiffList) extends SeqVariableDiff { def undo = {val e = _seq(i); _seq(i) = _seq(j); _seq(j) = e}; def redo = undo }
  // Override some methods for a slight gain in efficiency
  override def length = _seq.length
  override def iterator = _seq.iterator
  override def apply(index: Int) = _seq(index)
  // for changes without Diff tracking
  def +=(x:Element) = _seq += x
  def ++=(xs:Iterable[Element]) = _seq ++= xs
  //def update(index:Int, x:Element): Unit = _seq(index) = x // What should this be named, since we already have an update method above? -akm
}

class SeqDomain[X] extends Domain[Seq[X]]
object SeqDomain extends SeqDomain[Var]

/** A variable containing a mutable sequence of other variables. */
class SeqVariable[X] extends MutableSeqVar[X] {
  def this(initialValue: Seq[X]) = { this(); _seq ++= initialValue }
  def domain = SeqDomain.asInstanceOf[SeqDomain[X]]
}

