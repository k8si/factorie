/* Copyright (C) 2008-2014 University of Massachusetts Amherst.
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
package cc.factorie.app.nlp.segment

import java.io.StringReader

import cc.factorie.app.nlp.{TokenString, Document, DocumentAnnotator, Token}

import scala.reflect.ClassTag

/** Split a String into a sequence of Tokens.  Aims to adhere to tokenization rules used in Ontonotes and Penn Treebank.
    Note that CoNLL tokenization would use tokenizeAllDashedWords=true.
    Punctuation that ends a sentence should be placed alone in its own Token, hence this segmentation implicitly defines sentence segmentation also.
    (Although our the DeterministicSentenceSegmenter does make a few adjustments beyond this tokenizer.)
    @author Andrew McCallum
  */
class DeterministicLexerTokenizer(
    tokenizeSgml:Boolean = false, // Keep sgml/html tags as tokens
    tokenizeNewline:Boolean = false, // Keep newlines as tokens
    tokenizeWhitespace:Boolean = false, // Keep all whitespace, including newlines, as tokens
    tokenizeAllDashedWords:Boolean = false, // Separate dashed words into separate tokens, such as in CoNLL
    abbrevPrecedesLowercase:Boolean = false, // Assume a period followed by a lower case word is an abbrev and not end of sentence (see below)
    normalize: Boolean = true, // Whether to normalize token strings
    normalizeQuote:Boolean = true, // Convert all double quotes to "
    normalizeApostrophe:Boolean = true, // Convert all apostrophes to ', even within token strings
    normalizeCurrency:Boolean = true, // Convert all currency symbols to "$", except cents symbol to "cents"
    normalizeAmpersand:Boolean = true, // Convert all ampersand symbols (including "&amp;" to "&"
    normalizeFractions:Boolean = true, // Convert unicode fraction characters to their spelled out analogues, like "3/4"
    normalizeEllipsis:Boolean = true, // Convert unicode ellipsis character to spelled out analogue, "..."
    undoPennParens:Boolean = true, // Change -LRB- etc to "(" etc.
    unescapeSlash:Boolean = true, // Change \/ to /
    unescapeAsterisk:Boolean = true, // Change \* to *
    normalizeMDash:Boolean = true, // Convert all em-dashes to double dash --
    normalizeDash:Boolean = true, // Convert all other dashes to single dash -
    normalizeHtmlSymbol:Boolean = true, // Convert &lt; to <, etc
    normalizeHtmlAccent:Boolean = true // Convert Beyonc&eacute; to Beyonce
  ) extends DocumentAnnotator {

  /** How the annotation of this DocumentAnnotator should be printed in one-word-per-line (OWPL) format.
      If there is no per-token annotation, return null.  Used in Document.owplString. */
  def tokenAnnotationString(token: Token) = token.stringStart.toString+'\t'+token.stringEnd.toString

  def process(document: Document): Document = {
    for (section <- document.sections) {
      /* Add this newline to avoid JFlex issue where we can't match EOF with lookahead */
      val reader = new StringReader(section.string + "\n")
      val lexer =
        // here we make sure that if normalize = false, we really don't normalize anything
        if(normalize)
          new EnglishLexer(reader, tokenizeSgml, tokenizeNewline, tokenizeWhitespace, tokenizeAllDashedWords, abbrevPrecedesLowercase,
          normalizeQuote, normalizeApostrophe, normalizeCurrency, normalizeAmpersand, normalizeFractions, normalizeEllipsis,
          undoPennParens, unescapeSlash, unescapeAsterisk, normalizeMDash, normalizeDash, normalizeHtmlSymbol, normalizeHtmlAccent)
        else
          new EnglishLexer(reader, tokenizeSgml, tokenizeNewline, tokenizeWhitespace, tokenizeAllDashedWords, abbrevPrecedesLowercase,
            false, false, false, false, false, false, false, false, false, false, false, false, false)

      var currentToken = lexer.yylex().asInstanceOf[(String, Int, Int)]
      while (currentToken != null){
        if (abbrevPrecedesLowercase && section.length > 1 && section.tokens.last.string == "." && java.lang.Character.isLowerCase(currentToken._1(0)) && section.tokens(section.length-2).stringEnd == section.tokens(section.length-1).stringStart) {
          // If we have a pattern like "Abbrev. has" (where "has" is any lowercase word) with token strings "Abbrev", ".", "is" (currently looking at "is")
          // then assume that the previous-previous word is actually an abbreviation; patch it up to become "Abbrev.", "has".
          val lastTwoTokens = section.takeRight(2).toIndexedSeq
          section.remove(section.length - 1); section.remove(section.length - 1)
          new Token(section, lastTwoTokens(0).stringStart, lastTwoTokens(1).stringEnd)
        }
        val tok = new Token(section, currentToken._2, currentToken._2 + currentToken._3)
        if(normalize && tok.string != currentToken._1) tok.attr += new PlainNormalizedTokenString(tok, currentToken._1)
        currentToken = lexer.yylex().asInstanceOf[(String, Int, Int)]
      }
      /* If tokenizing newlines, remove the trailing newline we added */
      if(tokenizeNewline) section.remove(section.tokens.length - 1)
    }
    if (!document.annotators.contains(classOf[Token]))
      document.annotators(classOf[Token]) = this.getClass
    document
  }

  def prereqAttrs: Iterable[Class[_]] = Nil
  def postAttrs: Iterable[Class[_]] = List(classOf[Token])

  /** Convenience function to run the tokenizer on an arbitrary String.  The implementation builds a Document internally, then maps to token strings. */
  def apply(s:String): Seq[String] = process(new Document(s)).tokens.toSeq.map(_.string)
}

/* This version does not perform normalization, only tokenization */
object DeterministicLexerTokenizer extends DeterministicLexerTokenizer(
  tokenizeSgml = false,
  tokenizeNewline = false,
  tokenizeWhitespace = false,
  tokenizeAllDashedWords = false,
  abbrevPrecedesLowercase = false,
  normalize = false,
  normalizeQuote = false,
  normalizeApostrophe = false,
  normalizeCurrency = false,
  normalizeAmpersand = false,
  normalizeFractions = false,
  normalizeEllipsis = false,
  undoPennParens = false,
  unescapeSlash = false,
  unescapeAsterisk = false,
  normalizeMDash = false,
  normalizeDash = false,
  normalizeHtmlSymbol = false,
  normalizeHtmlAccent = false
)

/* This token performs normalization while it tokenizes, you probably want to use this one */
object DeterministicNormalizingTokenizer extends DeterministicLexerTokenizer(
  tokenizeSgml = false,
  tokenizeNewline = false,
  tokenizeWhitespace = false,
  tokenizeAllDashedWords = false,
  abbrevPrecedesLowercase = false,
  normalize = true,
  normalizeQuote = true,
  normalizeApostrophe = true,
  normalizeCurrency = true,
  normalizeAmpersand = true,
  normalizeFractions = true,
  normalizeEllipsis = true,
  undoPennParens = true,
  unescapeSlash = true,
  unescapeAsterisk = true,
  normalizeMDash = true,
  normalizeDash = true,
  normalizeHtmlSymbol = true,
  normalizeHtmlAccent = true
){
/* For testing purposes: Takes a filename as input and tokenizes it */
def main(args: Array[String]): Unit = {
    val fname = "/iesl/canvas/strubell/data/tackbp/source/2013/LDC2013E45_TAC_2013_KBP_Source_Corpus_disc_2/data/English/discussion_forums/bolt-eng-DF-200"
//    val fname = "/iesl/canvas/strubell/weird_character.txt"
//    val fname = "/Users/strubell/Documents/research/tunisia.txt"
    println(s"Loading $fname")
    val string = io.Source.fromFile(fname, "utf-8").mkString

//    val string = "A.  A.A.A.I.  and U.S. in U.S.. etc., but not A... or A..B iPhone 3G in Washington D.C.\n"
//    val string = "Washington D.C.... A..B!!C??D.!?E.!?.!?F..!!?? U.S.." // want: [A, .., B, !!, C, ??, D, .!?, E, .!?.!?, F, ..!!??]
//    val string = "AT&T but don't grab LAT&Eacute; and be sure not to grab PE&gym AT&T"
//    val string = "2012-04-05"
//    val string = "ethno-centric art-o-torium sure. thing"
//    val string = "prof. ph.d. a. a.b. a.b a.b.c. men.cd ab.cd"
//    val string = "he'll go to hell we're"
//    val string = "I paid $50 USD"
//    val string =  "$1 E2 L3 USD1 2KPW ||$1 USD1.." // want: "[$, 1, E2, L3, USD, 1, 2, KPW, |, |, $, 1, USD, 1, ..]"
//    val string = " 1. Buy a new Chevrolet (37%-owned in the U.S..) . 15%"
//    val string = "blah blah Abbrev. has blah"
//  val string = "''Going to the store to grab . . . some coffee for \u20ac50. He`s right\u2026 &ndash; &amp; \u2154 \\*\\* -- &trade; &mdash; \u2015 \u0096 -- --- .."


    println("Tokenizing...")
    val doc = new Document(string)
    val t0 = System.currentTimeMillis()
    DeterministicLexerTokenizer.process(doc)
    val time = System.currentTimeMillis()-t0
    println(s"Processed ${doc.tokenCount} tokens in ${time}ms (${doc.tokenCount.toDouble/time*1000} tokens/second)")
//    println(string)
//    println(doc.tokens.map(_.string).mkString("\n"))
  }
}
