# Seed phrase generation using dice; a printable visual flowchart

Generating a seed phrase using this visual guide requires a total of 253 dice rolls. One word requires 11 rolls and you will generate 23 words. The final 24th word is (mostly) a checksum word and it is okay to let the (offline) wallet app give you this word.

Note: If you suspect I gave you a tampered word list where many leaf nodes end up at the same word to limit the key space, don’t worry - you can use the BIP39 word index in parenthesis instead. The word indices are sorted in ascending order, per page, so they are very easy to visually verify as non-repeated.

For each of the 23 words, start at this **Starting sheet** and move along the flowchart, following pages, until you reach a leaf node. Then you come back here again to determine your next word.


![](temporary.png)
