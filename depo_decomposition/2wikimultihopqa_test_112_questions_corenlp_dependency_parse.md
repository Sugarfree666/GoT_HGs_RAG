# 2WikiMultiHopQA test_112 Questions and CoreNLP Dependency Parses

## 1. When did Lothair Ii's mother die?

Question:
When did Lothair Ii's mother die?

CoreNLP Dependency Parse:
- die[6] --advmod--> When[1]
- die[6] --aux--> did[2]
- mother[5] --nmod:poss--> PersonA[3]
- PersonA[3] --case--> 's[4]
- die[6] --nsubj--> mother[5]
- die[6] --punct--> ?[7]

## 2. Which film was released first, Aas Ka Panchhi or Phoolwari?

Question:
Which film was released first, Aas Ka Panchhi or Phoolwari?

CoreNLP Dependency Parse:
- film[2] --det--> Which[1]
- released[4] --nsubj:pass--> film[2]
- released[4] --aux:pass--> was[3]
- released[4] --advmod--> first[5]
- released[4] --punct--> ,[6]
- released[4] --obj--> FilmA[7]
- FilmB[9] --cc--> or[8]
- released[4] --obj--> FilmB[9]
- FilmA[7] --conj:or--> FilmB[9]
- released[4] --punct--> ?[10]

## 3. What is the place of birth of the performer of song Changed It?

Question:
What is the place of birth of the performer of song Changed It?

CoreNLP Dependency Parse:
- What[1] --cop--> is[2]
- place[4] --det--> the[3]
- What[1] --nsubj--> place[4]
- birth[6] --case--> of[5]
- place[4] --nmod:of--> birth[6]
- performer[9] --case--> of[7]
- performer[9] --det--> the[8]
- birth[6] --nmod:of--> performer[9]
- SongA[13] --case--> of[10]
- SongA[13] --det--> the[11]
- SongA[13] --compound--> song[12]
- performer[9] --nmod:of--> SongA[13]
- What[1] --punct--> ?[14]

## 4. Are Marufabad and Nasamkhrali both located in the same country?

Question:
Are Marufabad and Nasamkhrali both located in the same country?

CoreNLP Dependency Parse:
- located[6] --cop--> Are[1]
- located[6] --nsubj--> LocationA[2]
- LocationB[4] --cc--> and[3]
- LocationA[2] --conj:and--> LocationB[4]
- located[6] --nsubj--> LocationB[4]
- located[6] --cc:preconj--> both[5]
- country[10] --case--> in[7]
- country[10] --det--> the[8]
- country[10] --amod--> same[9]
- located[6] --obl:in--> country[10]
- located[6] --punct--> ?[11]

## 5. Which film has the director who is older, God'S Gift To Women or Aldri Annet Enn Br氓k?

Question:
Which film has the director who is older, God'S Gift To Women or Aldri Annet Enn Br氓k?

CoreNLP Dependency Parse:
- film[2] --det--> Which[1]
- has[3] --nsubj--> film[2]
- director[5] --det--> the[4]
- has[3] --obj--> director[5]
- older[8] --nsubj--> director[5]
- director[5] --ref--> who[6]
- older[8] --cop--> is[7]
- director[5] --acl:relcl--> older[8]
- FilmA[10] --case--> than[9]
- older[8] --obl:than--> FilmA[10]
- PersonA[12] --cc--> or[11]
- older[8] --obl:than--> PersonA[12]
- FilmA[10] --conj:or--> PersonA[12]
- has[3] --punct--> ?[13]

## 6. Which film whose director was born first, El Tonto or The Heart Of Doreon?

Question:
Which film whose director was born first, El Tonto or The Heart Of Doreon?

CoreNLP Dependency Parse:
- film[2] --det--> Which[1]
- film[2] --punct--> ,[3]
- director[5] --nmod:poss--> whose[4]
- born[7] --nsubj:pass--> director[5]
- born[7] --aux:pass--> was[6]
- film[2] --dep--> born[7]
- born[7] --advmod--> first[8]
- born[7] --punct--> ,[9]
- born[7] --obj--> FilmA[10]
- FilmB[12] --cc--> or[11]
- born[7] --obj--> FilmB[12]
- FilmA[10] --conj:or--> FilmB[12]
- film[2] --punct--> ?[13]

## 7. Who was born first out of Aivar Kuusmaa and Andy Summers?

Question:
Who was born first out of Aivar Kuusmaa and Andy Summers?

CoreNLP Dependency Parse:
- born[3] --nsubj:pass--> Who[1]
- born[3] --aux:pass--> was[2]
- born[3] --advmod--> first[4]
- PersonA[7] --case--> out[5]
- out[5] --fixed--> of[6]
- born[3] --obl:out_of--> PersonA[7]
- PersonB[9] --cc--> and[8]
- born[3] --obl:out_of--> PersonB[9]
- PersonA[7] --conj:and--> PersonB[9]
- born[3] --punct--> ?[10]

## 8. Who is Raghnall Mac Ruaidhr铆's paternal grandfather?

Question:
Who is Raghnall Mac Ruaidhr铆's paternal grandfather?

CoreNLP Dependency Parse:
- Who[1] --cop--> is[2]
- grandfather[5] --det--> the[3]
- grandfather[5] --amod--> paternal[4]
- Who[1] --nsubj--> grandfather[5]
- PersonA[7] --case--> of[6]
- grandfather[5] --nmod:of--> PersonA[7]
- Who[1] --punct--> ?[8]

## 9. Do both films Interview With A Hitman and The Last Coupon have the directors from the same country?

Question:
Do both films Interview With A Hitman and The Last Coupon have the directors from the same country?

CoreNLP Dependency Parse:
- have[7] --aux--> Do[1]
- FilmA[4] --cc:preconj--> both[2]
- FilmA[4] --compound--> films[3]
- have[7] --nsubj--> FilmA[4]
- FilmB[6] --cc--> and[5]
- FilmA[4] --conj:and--> FilmB[6]
- have[7] --nsubj--> FilmB[6]
- directors[9] --det--> the[8]
- have[7] --obj--> directors[9]
- country[13] --case--> from[10]
- country[13] --det--> the[11]
- country[13] --amod--> same[12]
- directors[9] --nmod:from--> country[13]
- have[7] --punct--> ?[14]

## 10. What nationality is the director of film Blood Street?

Question:
What nationality is the director of film Blood Street?

CoreNLP Dependency Parse:
- nationality[2] --det--> What[1]
- is[3] --obj--> nationality[2]
- director[5] --det--> the[4]
- is[3] --nsubj--> director[5]
- FilmA[9] --case--> of[6]
- FilmA[9] --det--> the[7]
- FilmA[9] --compound--> film[8]
- director[5] --nmod:of--> FilmA[9]
- is[3] --punct--> ?[10]

## 11. What is the place of birth of the director of film Gaby: A True Story?

Question:
What is the place of birth of the director of film Gaby: A True Story?

CoreNLP Dependency Parse:
- What[1] --cop--> is[2]
- place[4] --det--> the[3]
- What[1] --nsubj--> place[4]
- birth[6] --case--> of[5]
- place[4] --nmod:of--> birth[6]
- director[9] --case--> of[7]
- director[9] --det--> the[8]
- birth[6] --nmod:of--> director[9]
- FilmA[13] --case--> of[10]
- FilmA[13] --det--> the[11]
- FilmA[13] --compound--> film[12]
- director[9] --nmod:of--> FilmA[13]
- What[1] --punct--> ?[14]

## 12. Are Vasilyevsky Island and Preobrazheniya Island located in the same country?

Question:
Are Vasilyevsky Island and Preobrazheniya Island located in the same country?

CoreNLP Dependency Parse:
- located[5] --cop--> Are[1]
- located[5] --nsubj--> LocationA[2]
- LocationB[4] --cc--> and[3]
- LocationA[2] --conj:and--> LocationB[4]
- located[5] --nsubj--> LocationB[4]
- country[9] --case--> in[6]
- country[9] --det--> the[7]
- country[9] --amod--> same[8]
- located[5] --obl:in--> country[9]
- located[5] --punct--> ?[10]

## 13. What nationality is the performer of song When The Stars Go Blue?

Question:
What nationality is the performer of song When The Stars Go Blue?

CoreNLP Dependency Parse:
- nationality[2] --det--> What[1]
- is[3] --obj--> nationality[2]
- performer[5] --det--> the[4]
- is[3] --nsubj--> performer[5]
- song[8] --case--> of[6]
- song[8] --det--> the[7]
- performer[5] --nmod:of--> song[8]
- SongA[10] --advmod--> When[9]
- is[3] --dep--> SongA[10]
- is[3] --punct--> ?[11]

## 14. Who is the child of the performer of song Me And Bobby Mcgee?

Question:
Who is the child of the performer of song Me And Bobby Mcgee?

CoreNLP Dependency Parse:
- Who[1] --cop--> is[2]
- child[4] --det--> the[3]
- Who[1] --nsubj--> child[4]
- performer[7] --case--> of[5]
- performer[7] --det--> the[6]
- child[4] --nmod:of--> performer[7]
- SongA[11] --case--> of[8]
- SongA[11] --det--> the[9]
- SongA[11] --compound--> song[10]
- performer[7] --nmod:of--> SongA[11]
- Who[1] --punct--> ?[12]

## 15. Where was the place of death of Maurice, Prince Of Orange's father?

Question:
Where was the place of death of Maurice, Prince Of Orange's father?

CoreNLP Dependency Parse:
- was[2] --advmod--> Where[1]
- father[12] --dep--> was[2]
- place[4] --det--> the[3]
- was[2] --nsubj--> place[4]
- death[6] --case--> of[5]
- place[4] --nmod:of--> death[6]
- PersonA[8] --case--> of[7]
- death[6] --nmod:of--> PersonA[8]
- father[12] --punct--> ,[9]
- father[12] --nmod:poss--> PersonB[10]
- PersonB[10] --case--> 's[11]
- father[12] --punct--> ?[13]

## 16. Which country Aleksander Koniecpolski (1620鈥?659)'s father is from?

Question:
Which country Aleksander Koniecpolski (1620鈥?659)'s father is from?

CoreNLP Dependency Parse:
- country[2] --det--> Which[1]
- father[6] --nsubj--> country[2]
- father[6] --cop--> is[3]
- father[6] --nmod:poss--> PersonA[4]
- PersonA[4] --case--> 's[5]
- father[6] --dep--> from[7]
- father[6] --punct--> ?[8]

## 17. What is the date of death of the director of film Madame La Presidente?

Question:
What is the date of death of the director of film Madame La Presidente?

CoreNLP Dependency Parse:
- What[1] --cop--> is[2]
- date[4] --det--> the[3]
- What[1] --nsubj--> date[4]
- death[6] --case--> of[5]
- date[4] --nmod:of--> death[6]
- director[9] --case--> of[7]
- director[9] --det--> the[8]
- death[6] --nmod:of--> director[9]
- FilmA[13] --case--> of[10]
- FilmA[13] --det--> the[11]
- FilmA[13] --compound--> film[12]
- director[9] --nmod:of--> FilmA[13]
- What[1] --punct--> ?[14]

## 18. Do both directors of films Wrong Turn 5: Bloodlines and Dark River (2017 Film) have the same nationality?

Question:
Do both directors of films Wrong Turn 5: Bloodlines and Dark River (2017 Film) have the same nationality?

CoreNLP Dependency Parse:
- directors[3] --det--> both[2]
- Do[1] --obj--> directors[3]
- FilmA[7] --case--> of[4]
- FilmA[7] --det--> the[5]
- FilmA[7] --compound--> films[6]
- directors[3] --nmod:of--> FilmA[7]
- FilmA[7] --nummod--> 5[8]
- directors[3] --punct--> :[9]
- have[13] --nsubj--> Bloodlines[10]
- FilmB[12] --cc--> and[11]
- Bloodlines[10] --conj:and--> FilmB[12]
- have[13] --nsubj--> FilmB[12]
- directors[3] --dep--> have[13]
- nationality[16] --det--> the[14]
- nationality[16] --amod--> same[15]
- have[13] --obj--> nationality[16]
- Do[1] --punct--> ?[17]

## 19. Which film has the director who died first, The Goose Woman or You Can No Longer Remain Silent?

Question:
Which film has the director who died first, The Goose Woman or You Can No Longer Remain Silent?

CoreNLP Dependency Parse:
- film[2] --det--> Which[1]
- has[3] --nsubj--> film[2]
- director[5] --det--> the[4]
- has[3] --obj--> director[5]
- died[7] --nsubj--> director[5]
- director[5] --ref--> who[6]
- director[5] --acl:relcl--> died[7]
- died[7] --advmod--> first[8]
- died[7] --punct--> ,[9]
- died[7] --obj--> FilmA[10]
- FilmB[12] --cc--> or[11]
- died[7] --obj--> FilmB[12]
- FilmA[10] --conj:or--> FilmB[12]
- has[3] --punct--> ?[13]

## 20. Where was the director of film The Private Life Of Cinema born?

Question:
Where was the director of film The Private Life Of Cinema born?

CoreNLP Dependency Parse:
- was[2] --advmod--> Where[1]
- born[9] --aux:pass--> was[2]
- director[4] --det--> the[3]
- born[9] --nsubj:pass--> director[4]
- FilmA[8] --case--> of[5]
- FilmA[8] --det--> the[6]
- FilmA[8] --compound--> film[7]
- director[4] --nmod:of--> FilmA[8]
- born[9] --punct--> ?[10]

