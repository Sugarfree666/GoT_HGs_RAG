# DEPO Decomposition #17

- Dataset: `hotpotqa`
- Question: In what year was the Golden State NBA player, who was part of the Cavaliers-Warriors rivalry, named NBA Finals Most Valuable Player?
- Gold answer: 2015

## 1. Semantic-Normalized Question
In what year was the Golden State NBA player who was part of the Cavaliers-Warriors rivalry named NBA Finals Most Valuable Player?

## 2. Mask Spans
- Golden State NBA (entity, Person)
- NBA Finals Most Valuable Player (entity, Event)

## 3. Selective Masked Question
In what year was the PersonA player who was part of the Cavaliers-Warriors rivalry named SomeEntityA?

## 4. CoreNLP Dependency Parse
- year[3] --case--> In[1]
- year[3] --det--> what[2]
- player[7] --obl:in--> year[3]
- player[7] --cop--> was[4]
- player[7] --det--> the[5]
- player[7] --compound--> PersonA[6]
- named[17] --obj--> player[7]
- player[7] --ref--> who[8]
- who[8] --cop--> was[9]
- named[17] --nsubj--> part[10]
- rivalry[16] --case--> of[11]
- rivalry[16] --det--> the[12]
- Warriors[15] --compound--> Cavaliers[13]
- Warriors[15] --punct--> -[14]
- rivalry[16] --compound--> Warriors[15]
- part[10] --nmod:of--> rivalry[16]
- player[7] --acl:relcl--> named[17]
- named[17] --obj--> SomeEntityA[18]
- player[7] --punct--> ?[19]

## 5. Undirected Dependency Graph
- In[1] --case-- year[3]
- what[2] --det-- year[3]
- year[3] --obl:in-- player[7]
- was[4] --cop-- player[7]
- the[5] --det-- player[7]
- Golden State NBA[6] --compound-- player[7]
- player[7] --obj/acl:relcl-- named[17]
- player[7] --ref-- who[8]
- player[7] --punct-- ?[19]
- who[8] --cop-- was[9]
- part[10] --nsubj-- named[17]
- part[10] --nmod:of-- rivalry[16]
- of[11] --case-- rivalry[16]
- the[12] --det-- rivalry[16]
- Cavaliers[13] --compound-- Warriors[15]
- -[14] --punct-- Warriors[15]
- Warriors[15] --compound-- rivalry[16]
- named[17] --obj-- NBA Finals Most Valuable Player[18]

## 6. Entity Start Nodes
- e1: Golden State NBA graph_node_ids=['6']
- e2: NBA Finals Most Valuable Player graph_node_ids=['18']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Golden State NBA -- player -- named -- part -- rivalry -- Warriors -- Cavaliers
- e1_p2 (e1): Golden State NBA -- player -- named -- part -- rivalry -- Warriors
- e1_p3 (e1): Golden State NBA -- player -- named -- part -- rivalry -- Warriors -- -
- e1_p4 (e1): Golden State NBA -- player -- named -- part -- rivalry
- e1_p5 (e1): Golden State NBA -- player -- named -- part -- rivalry -- of
- e1_p6 (e1): Golden State NBA -- player -- named -- part -- rivalry -- the
- e1_p7 (e1): Golden State NBA -- player -- named -- part
- e1_p8 (e1): Golden State NBA -- player -- year -- what
- e1_p9 (e1): Golden State NBA -- player -- year
- e1_p10 (e1): Golden State NBA -- player -- named
- e1_p11 (e1): Golden State NBA -- player -- year -- In
- e1_p12 (e1): Golden State NBA -- player -- who
- e1_p13 (e1): Golden State NBA -- player -- who -- was
- e1_p14 (e1): Golden State NBA -- player
- e1_p15 (e1): Golden State NBA -- player -- was
- e1_p16 (e1): Golden State NBA -- player -- the
- e1_p17 (e1): Golden State NBA -- player -- ?
- e1_p18 (e1): Golden State NBA -- player -- named -- NBA Finals Most Valuable Player
- e2_p1 (e2): NBA Finals Most Valuable Player -- named -- part -- rivalry -- Warriors -- Cavaliers
- e2_p2 (e2): NBA Finals Most Valuable Player -- named -- part -- rivalry -- Warriors
- e2_p3 (e2): NBA Finals Most Valuable Player -- named -- part -- rivalry -- Warriors -- -
- e2_p4 (e2): NBA Finals Most Valuable Player -- named -- player -- year -- what
- e2_p5 (e2): NBA Finals Most Valuable Player -- named -- player -- year
- e2_p6 (e2): NBA Finals Most Valuable Player -- named -- part -- rivalry
- e2_p7 (e2): NBA Finals Most Valuable Player -- named -- player -- year -- In
- e2_p8 (e2): NBA Finals Most Valuable Player -- named -- part -- rivalry -- of
- e2_p9 (e2): NBA Finals Most Valuable Player -- named -- part -- rivalry -- the
- e2_p10 (e2): NBA Finals Most Valuable Player -- named -- player -- who
- e2_p11 (e2): NBA Finals Most Valuable Player -- named -- player -- who -- was
- e2_p12 (e2): NBA Finals Most Valuable Player -- named -- player
- e2_p13 (e2): NBA Finals Most Valuable Player -- named -- part
- e2_p14 (e2): NBA Finals Most Valuable Player -- named -- player -- was
- e2_p15 (e2): NBA Finals Most Valuable Player -- named -- player -- the
- e2_p16 (e2): NBA Finals Most Valuable Player -- named -- player -- ?
- e2_p17 (e2): NBA Finals Most Valuable Player -- named
- e2_p18 (e2): NBA Finals Most Valuable Player -- named -- player -- Golden State NBA

## 8. LLM Selected Entity Paths
- e1: e1_p18 Golden State NBA -- player -- named -- NBA Finals Most Valuable Player
  Reason: This path connects the Golden State NBA player directly to the NBA Finals Most Valuable Player, providing a clear reasoning chain to the answer.
- e2: e2_p5 NBA Finals Most Valuable Player -- named -- player -- year
  Reason: This path connects the NBA Finals Most Valuable Player to the player and year, which is essential for answering the question about the year they were named MVP.

## 9. Selected Path Semantic Transduction
Nodes:
- golden_state_nba_player: Golden State NBA player (type_variable)
- nba_finals_mvp: NBA Finals Most Valuable Player (entity)
- year: year (value_slot)

Edges:
- golden_state_nba_player -> nba_finals_mvp (named)
- nba_finals_mvp -> year (year of NBA Finals Most Valuable Player)

## 10. Atomic Subquestion DAG
- None: Who was the NBA Finals Most Valuable Player among Golden State NBA players?
- None: In what year was the NBA Finals Most Valuable Player of the Golden State NBA player awarded?
