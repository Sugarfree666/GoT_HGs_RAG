# DEPO Decomposition #17

- Dataset: `hotpotqa`
- Question: In what year was the Golden State NBA player, who was part of the Cavaliers-Warriors rivalry, named NBA Finals Most Valuable Player?
- Gold answer: 2015

## 1. Semantic-Normalized Question
In what year was the Golden State NBA player who was part of the Cavaliers-Warriors rivalry named NBA Finals Most Valuable Player?

## 2. Explicit Entities
- Golden State (Organization) span=(21, 33)
- Cavaliers-Warriors (Event) span=(65, 83)
- NBA Finals Most Valuable Player (Person) span=(98, 129)

## 3. Entity Masking
- OrganizationA -> Golden State
- EventA -> Cavaliers-Warriors
- PersonA -> NBA Finals Most Valuable Player

In what year was the OrganizationA NBA player who was part of the EventA rivalry named PersonA?

## 4. CoreNLP Dependency Parse
- year[3] --case--> In[1]
- year[3] --det--> what[2]
- player[8] --obl:in--> year[3]
- player[8] --cop--> was[4]
- player[8] --det--> the[5]
- player[8] --compound--> OrganizationA[6]
- player[8] --compound--> NBA[7]
- named[16] --obj--> player[8]
- player[8] --ref--> who[9]
- who[9] --cop--> was[10]
- named[16] --nsubj--> part[11]
- rivalry[15] --case--> of[12]
- rivalry[15] --det--> the[13]
- rivalry[15] --compound--> EventA[14]
- part[11] --nmod:of--> rivalry[15]
- player[8] --acl:relcl--> named[16]
- named[16] --obj--> PersonA[17]
- player[8] --punct--> ?[18]

## 5. Undirected Dependency Graph
- In[1] --case-- year[3]
- what[2] --det-- year[3]
- year[3] --obl:in-- player[8]
- was[4] --cop-- player[8]
- the[5] --det-- player[8]
- Golden State[6] --compound-- player[8]
- NBA[7] --compound-- player[8]
- player[8] --obj/acl:relcl-- named[16]
- player[8] --ref-- who[9]
- player[8] --punct-- ?[18]
- who[9] --cop-- was[10]
- part[11] --nsubj-- named[16]
- part[11] --nmod:of-- rivalry[15]
- of[12] --case-- rivalry[15]
- the[13] --det-- rivalry[15]
- Cavaliers-Warriors[14] --compound-- rivalry[15]
- named[16] --obj-- NBA Finals Most Valuable Player[17]

## 6. Entity Start Nodes from Explicit Entities
- e1: Golden State graph_node_ids=['6']
- e2: Cavaliers-Warriors graph_node_ids=['14']
- e3: NBA Finals Most Valuable Player graph_node_ids=['17']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Golden State -- player -- named -- part -- rivalry
- e1_p2 (e1): Golden State -- player -- named -- part -- rivalry -- of
- e1_p3 (e1): Golden State -- player -- named -- part -- rivalry -- the
- e1_p4 (e1): Golden State -- player -- named -- part
- e1_p5 (e1): Golden State -- player -- year -- what
- e1_p6 (e1): Golden State -- player -- year
- e1_p7 (e1): Golden State -- player -- NBA
- e1_p8 (e1): Golden State -- player -- named
- e1_p9 (e1): Golden State -- player -- year -- In
- e1_p10 (e1): Golden State -- player -- who
- e1_p11 (e1): Golden State -- player -- who -- was
- e1_p12 (e1): Golden State -- player
- e1_p13 (e1): Golden State -- player -- was
- e1_p14 (e1): Golden State -- player -- the
- e1_p15 (e1): Golden State -- player -- ?
- e1_p16 (e1): Golden State -- player -- named -- part -- rivalry -- Cavaliers-Warriors
- e1_p17 (e1): Golden State -- player -- named -- NBA Finals Most Valuable Player
- e2_p1 (e2): Cavaliers-Warriors -- rivalry -- part -- named -- player -- year -- what
- e2_p2 (e2): Cavaliers-Warriors -- rivalry -- part -- named -- player -- year
- e2_p3 (e2): Cavaliers-Warriors -- rivalry -- part -- named -- player -- NBA
- e2_p4 (e2): Cavaliers-Warriors -- rivalry -- part -- named -- player -- year -- In
- e2_p5 (e2): Cavaliers-Warriors -- rivalry -- part -- named -- player -- who
- e2_p6 (e2): Cavaliers-Warriors -- rivalry -- part -- named -- player -- who -- was
- e2_p7 (e2): Cavaliers-Warriors -- rivalry -- part -- named -- player
- e2_p8 (e2): Cavaliers-Warriors -- rivalry -- part -- named -- player -- was
- e2_p9 (e2): Cavaliers-Warriors -- rivalry -- part -- named -- player -- the
- e2_p10 (e2): Cavaliers-Warriors -- rivalry -- part -- named -- player -- ?
- e2_p11 (e2): Cavaliers-Warriors -- rivalry -- part -- named
- e2_p12 (e2): Cavaliers-Warriors -- rivalry -- part
- e2_p13 (e2): Cavaliers-Warriors -- rivalry
- e2_p14 (e2): Cavaliers-Warriors -- rivalry -- of
- e2_p15 (e2): Cavaliers-Warriors -- rivalry -- the
- e2_p16 (e2): Cavaliers-Warriors -- rivalry -- part -- named -- player -- Golden State
- e2_p17 (e2): Cavaliers-Warriors -- rivalry -- part -- named -- NBA Finals Most Valuable Player
- e3_p1 (e3): NBA Finals Most Valuable Player -- named -- player -- year -- what
- e3_p2 (e3): NBA Finals Most Valuable Player -- named -- player -- year
- e3_p3 (e3): NBA Finals Most Valuable Player -- named -- player -- NBA
- e3_p4 (e3): NBA Finals Most Valuable Player -- named -- part -- rivalry
- e3_p5 (e3): NBA Finals Most Valuable Player -- named -- player -- year -- In
- e3_p6 (e3): NBA Finals Most Valuable Player -- named -- part -- rivalry -- of
- e3_p7 (e3): NBA Finals Most Valuable Player -- named -- part -- rivalry -- the
- e3_p8 (e3): NBA Finals Most Valuable Player -- named -- player -- who
- e3_p9 (e3): NBA Finals Most Valuable Player -- named -- player -- who -- was
- e3_p10 (e3): NBA Finals Most Valuable Player -- named -- player
- e3_p11 (e3): NBA Finals Most Valuable Player -- named -- part
- e3_p12 (e3): NBA Finals Most Valuable Player -- named -- player -- was
- e3_p13 (e3): NBA Finals Most Valuable Player -- named -- player -- the
- e3_p14 (e3): NBA Finals Most Valuable Player -- named -- player -- ?
- e3_p15 (e3): NBA Finals Most Valuable Player -- named
- e3_p16 (e3): NBA Finals Most Valuable Player -- named -- part -- rivalry -- Cavaliers-Warriors
- e3_p17 (e3): NBA Finals Most Valuable Player -- named -- player -- Golden State

## 8. LLM Path Scores
- e1: e1_p1 score=85.0 valid=True terminal=NBA Finals Most Valuable Player
  Reason: The path starts from Golden State, reaches player, and includes the named predicate, but it does not reach the year.
- e1: e1_p2 score=90.0 valid=True terminal=NBA Finals Most Valuable Player
  Reason: The path starts from Golden State, reaches player, includes the named predicate, and connects to year, covering all necessary cues.
- e1: e1_p3 score=80.0 valid=True terminal=NBA Finals Most Valuable Player
  Reason: The path starts from Golden State, reaches player, and includes the named predicate, but it does not reach the year.
- e1: e1_p4 score=70.0 valid=True terminal=NBA Finals Most Valuable Player
  Reason: The path starts from Golden State and reaches player and named, but it does not connect to year.
- e1: e1_p5 score=75.0 valid=True terminal=NBA Finals Most Valuable Player
  Reason: The path starts from Golden State, reaches player, and connects to year, but lacks the named predicate.
- e1: e1_p6 score=65.0 valid=True terminal=NBA Finals Most Valuable Player
  Reason: The path starts from Golden State and reaches player, but it does not connect to year or include the named predicate.
- e1: e1_p7 score=50.0 valid=True terminal=NBA Finals Most Valuable Player
  Reason: The path starts from Golden State and reaches player, but it does not connect to year or include the named predicate.
- e1: e1_p8 score=60.0 valid=True terminal=NBA Finals Most Valuable Player
  Reason: The path starts from Golden State and reaches player and named, but it does not connect to year.
- e1: e1_p9 score=55.0 valid=True terminal=NBA Finals Most Valuable Player
  Reason: The path starts from Golden State and reaches player, but it does not connect to year or include the named predicate.
- e1: e1_p10 score=40.0 valid=True terminal=NBA Finals Most Valuable Player
  Reason: The path starts from Golden State and reaches player, but it does not connect to year or include the named predicate.
- e1: e1_p11 score=45.0 valid=True terminal=NBA Finals Most Valuable Player
  Reason: The path starts from Golden State and reaches player, but it does not connect to year or include the named predicate.
- e1: e1_p12 score=30.0 valid=True terminal=NBA Finals Most Valuable Player
  Reason: The path starts from Golden State and reaches player, but it does not connect to year or include the named predicate.
- e1: e1_p13 score=20.0 valid=True terminal=NBA Finals Most Valuable Player
  Reason: The path starts from Golden State and reaches player, but it does not connect to year or include the named predicate.
- e1: e1_p14 score=25.0 valid=True terminal=NBA Finals Most Valuable Player
  Reason: The path starts from Golden State and reaches player, but it does not connect to year or include the named predicate.
- e1: e1_p15 score=10.0 valid=True terminal=NBA Finals Most Valuable Player
  Reason: The path starts from Golden State and reaches player, but it does not connect to year or include the named predicate.
- e1: e1_p16 score=90.0 valid=True terminal=NBA Finals Most Valuable Player
  Reason: The path starts from Golden State, reaches player, includes the named predicate, and connects to Cavaliers-Warriors, covering all necessary cues.
- e1: e1_p17 score=95.0 valid=True terminal=NBA Finals Most Valuable Player
  Reason: The path starts from Golden State, reaches player, includes the named predicate, and connects to NBA Finals Most Valuable Player, covering all necessary cues.
- e2: e2_p1 score=90.0 valid=True terminal=NBA Finals Most Valuable Player
  Reason: The path starts from Cavaliers-Warriors, reaches rivalry, includes the named predicate, and connects to player, covering all necessary cues.
- e2: e2_p2 score=85.0 valid=True terminal=NBA Finals Most Valuable Player
  Reason: The path starts from Cavaliers-Warriors, reaches rivalry, includes the named predicate, and connects to player, covering all necessary cues.
- e2: e2_p3 score=80.0 valid=True terminal=NBA Finals Most Valuable Player
  Reason: The path starts from Cavaliers-Warriors, reaches rivalry, includes the named predicate, and connects to player, covering all necessary cues.
- e2: e2_p4 score=90.0 valid=True terminal=NBA Finals Most Valuable Player
  Reason: The path starts from Cavaliers-Warriors, reaches rivalry, includes the named predicate, and connects to player, covering all necessary cues.
- e2: e2_p5 score=75.0 valid=True terminal=NBA Finals Most Valuable Player
  Reason: The path starts from Cavaliers-Warriors, reaches rivalry, includes the named predicate, and connects to player, covering all necessary cues.
- e2: e2_p6 score=70.0 valid=True terminal=NBA Finals Most Valuable Player
  Reason: The path starts from Cavaliers-Warriors, reaches rivalry, includes the named predicate, and connects to player, covering all necessary cues.
- e2: e2_p7 score=60.0 valid=True terminal=NBA Finals Most Valuable Player
  Reason: The path starts from Cavaliers-Warriors, reaches rivalry, includes the named predicate, and connects to player, covering all necessary cues.
- e2: e2_p8 score=55.0 valid=True terminal=NBA Finals Most Valuable Player
  Reason: The path starts from Cavaliers-Warriors, reaches rivalry, includes the named predicate, and connects to player, covering all necessary cues.
- e2: e2_p9 score=50.0 valid=True terminal=NBA Finals Most Valuable Player
  Reason: The path starts from Cavaliers-Warriors and reaches rivalry, but it does not connect to player or include the named predicate.
- e2: e2_p10 score=45.0 valid=True terminal=NBA Finals Most Valuable Player
  Reason: The path starts from Cavaliers-Warriors and reaches rivalry, but it does not connect to player or include the named predicate.
- e2: e2_p11 score=40.0 valid=True terminal=NBA Finals Most Valuable Player
  Reason: The path starts from Cavaliers-Warriors and reaches rivalry, but it does not connect to player or include the named predicate.
- e2: e2_p12 score=30.0 valid=True terminal=NBA Finals Most Valuable Player
  Reason: The path starts from Cavaliers-Warriors and reaches rivalry, but it does not connect to player or include the named predicate.
- e2: e2_p13 score=20.0 valid=True terminal=NBA Finals Most Valuable Player
  Reason: The path starts from Cavaliers-Warriors and reaches rivalry, but it does not connect to player or include the named predicate.
- e2: e2_p14 score=25.0 valid=True terminal=NBA Finals Most Valuable Player
  Reason: The path starts from Cavaliers-Warriors and reaches rivalry, but it does not connect to player or include the named predicate.
- e2: e2_p15 score=10.0 valid=True terminal=NBA Finals Most Valuable Player
  Reason: The path starts from Cavaliers-Warriors and reaches rivalry, but it does not connect to player or include the named predicate.
- e2: e2_p16 score=90.0 valid=True terminal=NBA Finals Most Valuable Player
  Reason: The path starts from Cavaliers-Warriors, reaches rivalry, includes the named predicate, and connects to player, covering all necessary cues.
- e2: e2_p17 score=95.0 valid=True terminal=NBA Finals Most Valuable Player
  Reason: The path starts from Cavaliers-Warriors, reaches rivalry, includes the named predicate, and connects to NBA Finals Most Valuable Player, covering all necessary cues.
- e3: e3_p1 score=90.0 valid=True terminal=NBA Finals Most Valuable Player
  Reason: The path starts from NBA Finals Most Valuable Player, reaches named, and connects to player, covering all necessary cues.
- e3: e3_p2 score=85.0 valid=True terminal=NBA Finals Most Valuable Player
  Reason: The path starts from NBA Finals Most Valuable Player, reaches named, and connects to player, covering all necessary cues.
- e3: e3_p3 score=80.0 valid=True terminal=NBA Finals Most Valuable Player
  Reason: The path starts from NBA Finals Most Valuable Player, reaches named, and connects to player, covering all necessary cues.
- e3: e3_p4 score=75.0 valid=True terminal=NBA Finals Most Valuable Player
  Reason: The path starts from NBA Finals Most Valuable Player, reaches named, and connects to player, covering all necessary cues.
- e3: e3_p5 score=70.0 valid=True terminal=NBA Finals Most Valuable Player
  Reason: The path starts from NBA Finals Most Valuable Player, reaches named, and connects to player, covering all necessary cues.
- e3: e3_p6 score=65.0 valid=True terminal=NBA Finals Most Valuable Player
  Reason: The path starts from NBA Finals Most Valuable Player, reaches named, but does not connect to year.
- e3: e3_p7 score=60.0 valid=True terminal=NBA Finals Most Valuable Player
  Reason: The path starts from NBA Finals Most Valuable Player, reaches named, but does not connect to year.
- e3: e3_p8 score=55.0 valid=True terminal=NBA Finals Most Valuable Player
  Reason: The path starts from NBA Finals Most Valuable Player, reaches named, but does not connect to year.
- e3: e3_p9 score=50.0 valid=True terminal=NBA Finals Most Valuable Player
  Reason: The path starts from NBA Finals Most Valuable Player, reaches named, but does not connect to year.
- e3: e3_p10 score=40.0 valid=True terminal=NBA Finals Most Valuable Player
  Reason: The path starts from NBA Finals Most Valuable Player, reaches named, but does not connect to year.
- e3: e3_p11 score=30.0 valid=True terminal=NBA Finals Most Valuable Player
  Reason: The path starts from NBA Finals Most Valuable Player, reaches named, but does not connect to year.
- e3: e3_p12 score=25.0 valid=True terminal=NBA Finals Most Valuable Player
  Reason: The path starts from NBA Finals Most Valuable Player, reaches named, but does not connect to year.
- e3: e3_p13 score=20.0 valid=True terminal=NBA Finals Most Valuable Player
  Reason: The path starts from NBA Finals Most Valuable Player, reaches named, but does not connect to year.
- e3: e3_p14 score=15.0 valid=True terminal=NBA Finals Most Valuable Player
  Reason: The path starts from NBA Finals Most Valuable Player, reaches named, but does not connect to year.
- e3: e3_p15 score=10.0 valid=True terminal=NBA Finals Most Valuable Player
  Reason: The path starts from NBA Finals Most Valuable Player, reaches named, but does not connect to year.
- e3: e3_p16 score=90.0 valid=True terminal=NBA Finals Most Valuable Player
  Reason: The path starts from NBA Finals Most Valuable Player, reaches named, and connects to Cavaliers-Warriors, covering all necessary cues.
- e3: e3_p17 score=95.0 valid=True terminal=NBA Finals Most Valuable Player
  Reason: The path starts from NBA Finals Most Valuable Player, reaches named, and connects to player, covering all necessary cues.

## 8.1 Top-2 Paths per Entity
- e1: e1_p17, e1_p16
- e2: e2_p17, e2_p1
- e3: e3_p17, e3_p1

## 8.2 Candidate Path Sets
- ps1: {'e1': 'e1_p17', 'e2': 'e2_p17', 'e3': 'e3_p17'} mean_path_score=95.0
- ps2: {'e1': 'e1_p17', 'e2': 'e2_p17', 'e3': 'e3_p1'} mean_path_score=93.33333333333333
- ps3: {'e1': 'e1_p17', 'e2': 'e2_p1', 'e3': 'e3_p17'} mean_path_score=93.33333333333333
- ps4: {'e1': 'e1_p17', 'e2': 'e2_p1', 'e3': 'e3_p1'} mean_path_score=91.66666666666667
- ps5: {'e1': 'e1_p16', 'e2': 'e2_p17', 'e3': 'e3_p17'} mean_path_score=93.33333333333333
- ps6: {'e1': 'e1_p16', 'e2': 'e2_p17', 'e3': 'e3_p1'} mean_path_score=91.66666666666667
- ps7: {'e1': 'e1_p16', 'e2': 'e2_p1', 'e3': 'e3_p17'} mean_path_score=91.66666666666667
- ps8: {'e1': 'e1_p16', 'e2': 'e2_p1', 'e3': 'e3_p1'} mean_path_score=90.0

## 9. Grounded Atomic DAG Generation
Inputs:
- Original question: In what year was the Golden State NBA player, who was part of the Cavaliers-Warriors rivalry, named NBA Finals Most Valuable Player?
- ps1
  - e1_p17: Golden State -> player -> named -> NBA Finals Most Valuable Player
  - e2_p17: Cavaliers-Warriors -> rivalry -> part -> named -> NBA Finals Most Valuable Player
  - e3_p17: NBA Finals Most Valuable Player -> named -> player -> Golden State
- ps2
  - e1_p17: Golden State -> player -> named -> NBA Finals Most Valuable Player
  - e2_p17: Cavaliers-Warriors -> rivalry -> part -> named -> NBA Finals Most Valuable Player
  - e3_p1: NBA Finals Most Valuable Player -> named -> player -> year -> what
- ps3
  - e1_p17: Golden State -> player -> named -> NBA Finals Most Valuable Player
  - e2_p1: Cavaliers-Warriors -> rivalry -> part -> named -> player -> year -> what
  - e3_p17: NBA Finals Most Valuable Player -> named -> player -> Golden State
- ps4
  - e1_p17: Golden State -> player -> named -> NBA Finals Most Valuable Player
  - e2_p1: Cavaliers-Warriors -> rivalry -> part -> named -> player -> year -> what
  - e3_p1: NBA Finals Most Valuable Player -> named -> player -> year -> what

Output:
- selected_path_set_ids: ['ps1', 'ps2']
- reason: The DAG decomposes the question into one-hop lookup questions grounded by selected dependency path evidence.
- q1: Who is the NBA Finals Most Valuable Player from Golden State? depends_on=[] support=['e1_p17']
- q2: In what year was q1's answer named NBA Finals Most Valuable Player? depends_on=['q1'] support=['e3_p1']

## 10. Atomic Subquestion DAG
- None: Who is the NBA Finals Most Valuable Player from Golden State?
- None: In what year was q1's answer named NBA Finals Most Valuable Player?
