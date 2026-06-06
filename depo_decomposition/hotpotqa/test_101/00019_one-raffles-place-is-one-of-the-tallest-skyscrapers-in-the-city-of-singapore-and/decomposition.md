# DEPO Decomposition #19

- Dataset: `hotpotqa`
- Question: One Raffles Place is one of the tallest skyscrapers in the city of Singapore and tallest in the wolrd outside North America until it was succeeded by a Building in city?
- Gold answer: Hong Kong

## 1. Semantic-Normalized Question
One Raffles Place is one of the tallest skyscrapers in the city of Singapore and the tallest in the world outside North America until it was succeeded by a building in the city?

## 2. Mask Spans
- One Raffles Place (entity, City)
- North America (entity, Region)

## 3. Selective Masked Question
CityA is one of the tallest skyscrapers in the city of Singapore and the tallest in the world outside RegionA until it was succeeded by a building in the city?

## 4. CoreNLP Dependency Parse
- skyscrapers[7] --nsubj--> CityA[1]
- tallest[15] --nsubj--> CityA[1]
- skyscrapers[7] --cop--> is[2]
- skyscrapers[7] --det:qmod--> one[3]
- one[3] --fixed--> of[4]
- skyscrapers[7] --det--> the[5]
- skyscrapers[7] --amod--> tallest[6]
- city[10] --case--> in[8]
- city[10] --det--> the[9]
- skyscrapers[7] --nmod:in--> city[10]
- Singapore[12] --case--> of[11]
- city[10] --nmod:of--> Singapore[12]
- tallest[15] --cc--> and[13]
- tallest[15] --det--> the[14]
- skyscrapers[7] --conj:and--> tallest[15]
- world[18] --case--> in[16]
- world[18] --det--> the[17]
- tallest[15] --nmod:in--> world[18]
- RegionA[20] --case--> outside[19]
- tallest[15] --nmod:outside--> RegionA[20]
- succeeded[24] --mark--> until[21]
- succeeded[24] --nsubj:pass--> it[22]
- succeeded[24] --aux:pass--> was[23]
- tallest[15] --advcl:until--> succeeded[24]
- building[27] --case--> by[25]
- building[27] --det--> a[26]
- succeeded[24] --obl:agent--> building[27]
- city[30] --case--> in[28]
- city[30] --det--> the[29]
- building[27] --nmod:in--> city[30]
- skyscrapers[7] --punct--> ?[31]

## 5. Undirected Dependency Graph
- One Raffles Place[1] --nsubj-- skyscrapers[7]
- One Raffles Place[1] --nsubj-- tallest[15]
- is[2] --cop-- skyscrapers[7]
- one[3] --det:qmod-- skyscrapers[7]
- one[3] --fixed-- of[4]
- the[5] --det-- skyscrapers[7]
- tallest[6] --amod-- skyscrapers[7]
- skyscrapers[7] --nmod:in-- city[10]
- skyscrapers[7] --conj:and-- tallest[15]
- skyscrapers[7] --punct-- ?[31]
- in[8] --case-- city[10]
- the[9] --det-- city[10]
- city[10] --nmod:of-- Singapore[12]
- of[11] --case-- Singapore[12]
- and[13] --cc-- tallest[15]
- the[14] --det-- tallest[15]
- tallest[15] --nmod:in-- world[18]
- tallest[15] --nmod:outside-- North America[20]
- tallest[15] --advcl:until-- succeeded[24]
- in[16] --case-- world[18]
- the[17] --det-- world[18]
- outside[19] --case-- North America[20]
- until[21] --mark-- succeeded[24]
- it[22] --nsubj:pass-- succeeded[24]
- was[23] --aux:pass-- succeeded[24]
- succeeded[24] --obl:agent-- building[27]
- by[25] --case-- building[27]
- a[26] --det-- building[27]
- building[27] --nmod:in-- city[30]
- in[28] --case-- city[30]
- the[29] --det-- city[30]

## 6. Entity Start Nodes
- e1: One Raffles Place graph_node_ids=['1']
- e2: North America graph_node_ids=['20']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): One Raffles Place -- skyscrapers -- tallest -- succeeded -- building -- city
- e1_p2 (e1): One Raffles Place -- skyscrapers -- tallest -- succeeded -- building -- city -- in
- e1_p3 (e1): One Raffles Place -- skyscrapers -- tallest -- succeeded -- building -- city -- the
- e1_p4 (e1): One Raffles Place -- tallest -- skyscrapers -- city -- Singapore
- e1_p5 (e1): One Raffles Place -- tallest -- succeeded -- building -- city
- e1_p6 (e1): One Raffles Place -- tallest -- skyscrapers -- city -- Singapore -- of
- e1_p7 (e1): One Raffles Place -- tallest -- succeeded -- building -- city -- in
- e1_p8 (e1): One Raffles Place -- tallest -- succeeded -- building -- city -- the
- e1_p9 (e1): One Raffles Place -- skyscrapers -- tallest -- succeeded -- until
- e1_p10 (e1): One Raffles Place -- skyscrapers -- tallest -- succeeded -- it
- e1_p11 (e1): One Raffles Place -- skyscrapers -- tallest -- succeeded -- building
- e1_p12 (e1): One Raffles Place -- skyscrapers -- tallest -- succeeded -- building -- by
- e1_p13 (e1): One Raffles Place -- skyscrapers -- tallest -- succeeded -- building -- a
- e1_p14 (e1): One Raffles Place -- skyscrapers -- city -- Singapore
- e1_p15 (e1): One Raffles Place -- tallest -- skyscrapers -- city
- e1_p16 (e1): One Raffles Place -- skyscrapers -- city -- Singapore -- of
- e1_p17 (e1): One Raffles Place -- tallest -- skyscrapers -- city -- in
- e1_p18 (e1): One Raffles Place -- tallest -- skyscrapers -- city -- the
- e1_p19 (e1): One Raffles Place -- skyscrapers -- tallest -- world
- e1_p20 (e1): One Raffles Place -- skyscrapers -- tallest -- succeeded
- e1_p21 (e1): One Raffles Place -- tallest -- skyscrapers -- one
- e1_p22 (e1): One Raffles Place -- tallest -- skyscrapers -- tallest
- e1_p23 (e1): One Raffles Place -- tallest -- succeeded -- until
- e1_p24 (e1): One Raffles Place -- tallest -- succeeded -- it
- e1_p25 (e1): One Raffles Place -- tallest -- succeeded -- building
- e1_p26 (e1): One Raffles Place -- skyscrapers -- tallest -- world -- in
- e1_p27 (e1): One Raffles Place -- skyscrapers -- tallest -- world -- the
- e1_p28 (e1): One Raffles Place -- skyscrapers -- tallest -- succeeded -- was
- e1_p29 (e1): One Raffles Place -- tallest -- skyscrapers -- one -- of
- e1_p30 (e1): One Raffles Place -- tallest -- succeeded -- building -- by
- e1_p31 (e1): One Raffles Place -- tallest -- succeeded -- building -- a
- e1_p32 (e1): One Raffles Place -- skyscrapers -- city
- e1_p33 (e1): One Raffles Place -- skyscrapers -- city -- in
- e1_p34 (e1): One Raffles Place -- skyscrapers -- city -- the
- e1_p35 (e1): One Raffles Place -- skyscrapers -- one
- e1_p36 (e1): One Raffles Place -- skyscrapers -- tallest
- e1_p37 (e1): One Raffles Place -- skyscrapers -- tallest
- e1_p38 (e1): One Raffles Place -- tallest -- skyscrapers
- e1_p39 (e1): One Raffles Place -- tallest -- world
- e1_p40 (e1): One Raffles Place -- tallest -- succeeded
- e1_p41 (e1): One Raffles Place -- skyscrapers -- one -- of
- e1_p42 (e1): One Raffles Place -- skyscrapers -- tallest -- and
- e1_p43 (e1): One Raffles Place -- skyscrapers -- tallest -- the
- e1_p44 (e1): One Raffles Place -- tallest -- skyscrapers -- is
- e1_p45 (e1): One Raffles Place -- tallest -- skyscrapers -- the
- e1_p46 (e1): One Raffles Place -- tallest -- skyscrapers -- ?
- e1_p47 (e1): One Raffles Place -- tallest -- world -- in
- e1_p48 (e1): One Raffles Place -- tallest -- world -- the
- e1_p49 (e1): One Raffles Place -- tallest -- succeeded -- was
- e1_p50 (e1): One Raffles Place -- skyscrapers
- e1_p51 (e1): One Raffles Place -- tallest
- e1_p52 (e1): One Raffles Place -- skyscrapers -- is
- e1_p53 (e1): One Raffles Place -- skyscrapers -- the
- e1_p54 (e1): One Raffles Place -- skyscrapers -- ?
- e1_p55 (e1): One Raffles Place -- tallest -- and
- e1_p56 (e1): One Raffles Place -- tallest -- the
- e1_p57 (e1): One Raffles Place -- skyscrapers -- tallest -- North America
- e1_p58 (e1): One Raffles Place -- tallest -- North America
- e1_p59 (e1): One Raffles Place -- skyscrapers -- tallest -- North America -- outside
- e1_p60 (e1): One Raffles Place -- tallest -- North America -- outside
- e2_p1 (e2): North America -- tallest -- skyscrapers -- city -- Singapore
- e2_p2 (e2): North America -- tallest -- succeeded -- building -- city
- e2_p3 (e2): North America -- tallest -- skyscrapers -- city -- Singapore -- of
- e2_p4 (e2): North America -- tallest -- succeeded -- building -- city -- in
- e2_p5 (e2): North America -- tallest -- succeeded -- building -- city -- the
- e2_p6 (e2): North America -- tallest -- skyscrapers -- city
- e2_p7 (e2): North America -- tallest -- skyscrapers -- city -- in
- e2_p8 (e2): North America -- tallest -- skyscrapers -- city -- the
- e2_p9 (e2): North America -- tallest -- skyscrapers -- one
- e2_p10 (e2): North America -- tallest -- skyscrapers -- tallest
- e2_p11 (e2): North America -- tallest -- succeeded -- until
- e2_p12 (e2): North America -- tallest -- succeeded -- it
- e2_p13 (e2): North America -- tallest -- succeeded -- building
- e2_p14 (e2): North America -- tallest -- skyscrapers -- one -- of
- e2_p15 (e2): North America -- tallest -- succeeded -- building -- by
- e2_p16 (e2): North America -- tallest -- succeeded -- building -- a
- e2_p17 (e2): North America -- tallest -- skyscrapers
- e2_p18 (e2): North America -- tallest -- world
- e2_p19 (e2): North America -- tallest -- succeeded
- e2_p20 (e2): North America -- tallest -- skyscrapers -- is
- e2_p21 (e2): North America -- tallest -- skyscrapers -- the
- e2_p22 (e2): North America -- tallest -- skyscrapers -- ?
- e2_p23 (e2): North America -- tallest -- world -- in
- e2_p24 (e2): North America -- tallest -- world -- the
- e2_p25 (e2): North America -- tallest -- succeeded -- was
- e2_p26 (e2): North America -- tallest
- e2_p27 (e2): North America -- outside
- e2_p28 (e2): North America -- tallest -- and
- e2_p29 (e2): North America -- tallest -- the
- e2_p30 (e2): North America -- tallest -- skyscrapers -- One Raffles Place
- e2_p31 (e2): North America -- tallest -- One Raffles Place
- e2_p32 (e2): North America -- tallest -- One Raffles Place -- skyscrapers -- city -- Singapore
- e2_p33 (e2): North America -- tallest -- One Raffles Place -- skyscrapers -- city -- Singapore -- of
- e2_p34 (e2): North America -- tallest -- One Raffles Place -- skyscrapers -- city
- e2_p35 (e2): North America -- tallest -- One Raffles Place -- skyscrapers -- city -- in
- e2_p36 (e2): North America -- tallest -- One Raffles Place -- skyscrapers -- city -- the
- e2_p37 (e2): North America -- tallest -- One Raffles Place -- skyscrapers -- one
- e2_p38 (e2): North America -- tallest -- One Raffles Place -- skyscrapers -- tallest
- e2_p39 (e2): North America -- tallest -- One Raffles Place -- skyscrapers -- one -- of
- e2_p40 (e2): North America -- tallest -- One Raffles Place -- skyscrapers
- e2_p41 (e2): North America -- tallest -- One Raffles Place -- skyscrapers -- is
- e2_p42 (e2): North America -- tallest -- One Raffles Place -- skyscrapers -- the
- e2_p43 (e2): North America -- tallest -- One Raffles Place -- skyscrapers -- ?

## 8. LLM Selected Entity Paths
- e1: e1_p57 One Raffles Place -- skyscrapers -- tallest -- North America
  Reason: This path connects 'One Raffles Place' to 'tallest' and 'North America', which is crucial for understanding the comparison in the question.
- e2: e2_p1 North America -- tallest -- skyscrapers -- city -- Singapore
  Reason: This path connects 'North America' to 'tallest' and 'Singapore', which is essential for reasoning about the original question regarding the tallest skyscraper.

## 9. Selected Path Semantic Transduction
Nodes:
- one_raffles_place: One Raffles Place (entity)
- tallest: tallest (type_variable)
- north_america: North America (entity)
- city: city (type_variable)
- singapore: Singapore (entity)
- building: building (type_variable)

Edges:
- one_raffles_place -> tallest (tallest skyscraper)
- tallest -> north_america (outside)
- tallest -> city (in)
- city -> singapore (of)
- tallest -> building (succeeded by)
- building -> city (in)

## 10. Atomic Subquestion DAG
- None: What is the tallest skyscraper in Singapore?
- None: What is the tallest building outside North America?
- None: In which city is the tallest of One Raffles Place located?
- None: What is the city of Singapore?
- None: What building succeeded One Raffles Place as the tallest in the world outside North America?
- None: In which city is the building of the tallest of One Raffles Place located?
