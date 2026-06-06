# DEPO Decomposition #13

- Dataset: `hotpotqa`
- Question: How do the Peter Laufer books Forbidden Creatures and No Animals Were Harmed differ in their focus on animals?
- Gold answer: his own opinions changed

## 1. Semantic-Normalized Question
How do the books of Peter Laufer, Forbidden Creatures and No Animals Were Harmed, differ in their focus on animals?

## 2. Mask Spans
- Peter Laufer (entity, Person)
- Forbidden Creatures (entity, Book)
- No Animals Were Harmed (entity, Book)

## 3. Selective Masked Question
How do the books of PersonA, BookA and BookB, differ in their focus on animals?

## 4. CoreNLP Dependency Parse
- differ[12] --advmod--> How[1]
- differ[12] --aux--> do[2]
- books[4] --det--> the[3]
- differ[12] --nsubj--> books[4]
- PersonA[6] --case--> of[5]
- books[4] --nmod:of--> PersonA[6]
- PersonA[6] --punct--> ,[7]
- books[4] --nmod:of--> BookA[8]
- PersonA[6] --conj:and--> BookA[8]
- BookB[10] --cc--> and[9]
- books[4] --nmod:of--> BookB[10]
- PersonA[6] --conj:and--> BookB[10]
- differ[12] --punct--> ,[11]
- focus[15] --case--> in[13]
- focus[15] --nmod:poss--> their[14]
- differ[12] --obl:in--> focus[15]
- animals[17] --case--> on[16]
- focus[15] --nmod:on--> animals[17]
- differ[12] --punct--> ?[18]

## 5. Undirected Dependency Graph
- How[1] --advmod-- differ[12]
- do[2] --aux-- differ[12]
- the[3] --det-- books[4]
- books[4] --nsubj-- differ[12]
- books[4] --nmod:of-- Peter Laufer[6]
- books[4] --nmod:of-- Forbidden Creatures[8]
- books[4] --nmod:of-- No Animals Were Harmed[10]
- of[5] --case-- Peter Laufer[6]
- Peter Laufer[6] --punct-- ,[7]
- Peter Laufer[6] --conj:and-- Forbidden Creatures[8]
- Peter Laufer[6] --conj:and-- No Animals Were Harmed[10]
- and[9] --cc-- No Animals Were Harmed[10]
- ,[11] --punct-- differ[12]
- differ[12] --obl:in-- focus[15]
- differ[12] --punct-- ?[18]
- in[13] --case-- focus[15]
- their[14] --nmod:poss-- focus[15]
- focus[15] --nmod:on-- animals[17]
- on[16] --case-- animals[17]

## 6. Entity Start Nodes
- e1: Peter Laufer graph_node_ids=['6']
- e2: Forbidden Creatures graph_node_ids=['8']
- e3: No Animals Were Harmed graph_node_ids=['10']

## 7. Entity-Origin Dependency Paths
- e1_p1 (e1): Peter Laufer -- books -- differ -- focus -- animals -- on
- e1_p2 (e1): Peter Laufer -- books -- differ -- focus -- their
- e1_p3 (e1): Peter Laufer -- books -- differ -- focus -- animals
- e1_p4 (e1): Peter Laufer -- books -- differ -- How
- e1_p5 (e1): Peter Laufer -- books -- differ -- focus
- e1_p6 (e1): Peter Laufer -- books -- differ -- focus -- in
- e1_p7 (e1): Peter Laufer -- books -- differ
- e1_p8 (e1): Peter Laufer -- books -- differ -- do
- e1_p9 (e1): Peter Laufer -- books -- differ -- ,
- e1_p10 (e1): Peter Laufer -- books -- differ -- ?
- e1_p11 (e1): Peter Laufer -- books
- e1_p12 (e1): Peter Laufer -- books -- the
- e1_p13 (e1): Peter Laufer -- of
- e1_p14 (e1): Peter Laufer -- ,
- e1_p15 (e1): Peter Laufer -- books -- Forbidden Creatures
- e1_p16 (e1): Peter Laufer -- books -- No Animals Were Harmed
- e1_p17 (e1): Peter Laufer -- Forbidden Creatures
- e1_p18 (e1): Peter Laufer -- No Animals Were Harmed
- e1_p19 (e1): Peter Laufer -- Forbidden Creatures -- books -- differ -- focus -- animals -- on
- e1_p20 (e1): Peter Laufer -- No Animals Were Harmed -- books -- differ -- focus -- animals -- on
- e1_p21 (e1): Peter Laufer -- Forbidden Creatures -- books -- differ -- focus -- their
- e1_p22 (e1): Peter Laufer -- Forbidden Creatures -- books -- differ -- focus -- animals
- e1_p23 (e1): Peter Laufer -- No Animals Were Harmed -- books -- differ -- focus -- their
- e1_p24 (e1): Peter Laufer -- No Animals Were Harmed -- books -- differ -- focus -- animals
- e1_p25 (e1): Peter Laufer -- Forbidden Creatures -- books -- differ -- How
- e1_p26 (e1): Peter Laufer -- Forbidden Creatures -- books -- differ -- focus
- e1_p27 (e1): Peter Laufer -- No Animals Were Harmed -- books -- differ -- How
- e1_p28 (e1): Peter Laufer -- No Animals Were Harmed -- books -- differ -- focus
- e1_p29 (e1): Peter Laufer -- Forbidden Creatures -- books -- differ -- focus -- in
- e1_p30 (e1): Peter Laufer -- No Animals Were Harmed -- books -- differ -- focus -- in
- e1_p31 (e1): Peter Laufer -- Forbidden Creatures -- books -- differ
- e1_p32 (e1): Peter Laufer -- No Animals Were Harmed -- books -- differ
- e1_p33 (e1): Peter Laufer -- Forbidden Creatures -- books -- differ -- do
- e1_p34 (e1): Peter Laufer -- Forbidden Creatures -- books -- differ -- ,
- e1_p35 (e1): Peter Laufer -- Forbidden Creatures -- books -- differ -- ?
- e1_p36 (e1): Peter Laufer -- No Animals Were Harmed -- books -- differ -- do
- e1_p37 (e1): Peter Laufer -- No Animals Were Harmed -- books -- differ -- ,
- e1_p38 (e1): Peter Laufer -- No Animals Were Harmed -- books -- differ -- ?
- e1_p39 (e1): Peter Laufer -- Forbidden Creatures -- books
- e1_p40 (e1): Peter Laufer -- No Animals Were Harmed -- books
- e1_p41 (e1): Peter Laufer -- books -- No Animals Were Harmed -- and
- e1_p42 (e1): Peter Laufer -- Forbidden Creatures -- books -- the
- e1_p43 (e1): Peter Laufer -- No Animals Were Harmed -- books -- the
- e1_p44 (e1): Peter Laufer -- No Animals Were Harmed -- and
- e1_p45 (e1): Peter Laufer -- Forbidden Creatures -- books -- No Animals Were Harmed
- e1_p46 (e1): Peter Laufer -- No Animals Were Harmed -- books -- Forbidden Creatures
- e1_p47 (e1): Peter Laufer -- Forbidden Creatures -- books -- No Animals Were Harmed -- and
- e2_p1 (e2): Forbidden Creatures -- books -- differ -- focus -- animals -- on
- e2_p2 (e2): Forbidden Creatures -- books -- differ -- focus -- their
- e2_p3 (e2): Forbidden Creatures -- books -- differ -- focus -- animals
- e2_p4 (e2): Forbidden Creatures -- books -- differ -- How
- e2_p5 (e2): Forbidden Creatures -- books -- differ -- focus
- e2_p6 (e2): Forbidden Creatures -- books -- differ -- focus -- in
- e2_p7 (e2): Forbidden Creatures -- books -- differ
- e2_p8 (e2): Forbidden Creatures -- books -- differ -- do
- e2_p9 (e2): Forbidden Creatures -- books -- differ -- ,
- e2_p10 (e2): Forbidden Creatures -- books -- differ -- ?
- e2_p11 (e2): Forbidden Creatures -- books
- e2_p12 (e2): Forbidden Creatures -- books -- the
- e2_p13 (e2): Forbidden Creatures -- books -- Peter Laufer
- e2_p14 (e2): Forbidden Creatures -- books -- No Animals Were Harmed
- e2_p15 (e2): Forbidden Creatures -- Peter Laufer
- e2_p16 (e2): Forbidden Creatures -- Peter Laufer -- books -- differ -- focus -- animals -- on
- e2_p17 (e2): Forbidden Creatures -- Peter Laufer -- books -- differ -- focus -- their
- e2_p18 (e2): Forbidden Creatures -- Peter Laufer -- books -- differ -- focus -- animals
- e2_p19 (e2): Forbidden Creatures -- Peter Laufer -- books -- differ -- How
- e2_p20 (e2): Forbidden Creatures -- Peter Laufer -- books -- differ -- focus
- e2_p21 (e2): Forbidden Creatures -- Peter Laufer -- books -- differ -- focus -- in
- e2_p22 (e2): Forbidden Creatures -- Peter Laufer -- books -- differ
- e2_p23 (e2): Forbidden Creatures -- Peter Laufer -- books -- differ -- do
- e2_p24 (e2): Forbidden Creatures -- Peter Laufer -- books -- differ -- ,
- e2_p25 (e2): Forbidden Creatures -- Peter Laufer -- books -- differ -- ?
- e2_p26 (e2): Forbidden Creatures -- Peter Laufer -- books
- e2_p27 (e2): Forbidden Creatures -- books -- Peter Laufer -- of
- e2_p28 (e2): Forbidden Creatures -- books -- Peter Laufer -- ,
- e2_p29 (e2): Forbidden Creatures -- books -- No Animals Were Harmed -- and
- e2_p30 (e2): Forbidden Creatures -- Peter Laufer -- books -- the
- e2_p31 (e2): Forbidden Creatures -- Peter Laufer -- of
- e2_p32 (e2): Forbidden Creatures -- Peter Laufer -- ,
- e2_p33 (e2): Forbidden Creatures -- books -- Peter Laufer -- No Animals Were Harmed
- e2_p34 (e2): Forbidden Creatures -- books -- No Animals Were Harmed -- Peter Laufer
- e2_p35 (e2): Forbidden Creatures -- Peter Laufer -- books -- No Animals Were Harmed
- e2_p36 (e2): Forbidden Creatures -- Peter Laufer -- No Animals Were Harmed
- e2_p37 (e2): Forbidden Creatures -- Peter Laufer -- No Animals Were Harmed -- books -- differ -- focus -- their
- e2_p38 (e2): Forbidden Creatures -- Peter Laufer -- No Animals Were Harmed -- books -- differ -- focus -- animals
- e2_p39 (e2): Forbidden Creatures -- Peter Laufer -- No Animals Were Harmed -- books -- differ -- How
- e2_p40 (e2): Forbidden Creatures -- Peter Laufer -- No Animals Were Harmed -- books -- differ -- focus
- e2_p41 (e2): Forbidden Creatures -- Peter Laufer -- No Animals Were Harmed -- books -- differ -- focus -- in
- e2_p42 (e2): Forbidden Creatures -- Peter Laufer -- No Animals Were Harmed -- books -- differ
- e2_p43 (e2): Forbidden Creatures -- Peter Laufer -- No Animals Were Harmed -- books -- differ -- do
- e2_p44 (e2): Forbidden Creatures -- Peter Laufer -- No Animals Were Harmed -- books -- differ -- ,
- e2_p45 (e2): Forbidden Creatures -- Peter Laufer -- No Animals Were Harmed -- books -- differ -- ?
- e2_p46 (e2): Forbidden Creatures -- Peter Laufer -- No Animals Were Harmed -- books
- e2_p47 (e2): Forbidden Creatures -- books -- Peter Laufer -- No Animals Were Harmed -- and
- e2_p48 (e2): Forbidden Creatures -- books -- No Animals Were Harmed -- Peter Laufer -- of
- e2_p49 (e2): Forbidden Creatures -- books -- No Animals Were Harmed -- Peter Laufer -- ,
- e2_p50 (e2): Forbidden Creatures -- Peter Laufer -- books -- No Animals Were Harmed -- and
- e2_p51 (e2): Forbidden Creatures -- Peter Laufer -- No Animals Were Harmed -- books -- the
- e2_p52 (e2): Forbidden Creatures -- Peter Laufer -- No Animals Were Harmed -- and
- e3_p1 (e3): No Animals Were Harmed -- books -- differ -- focus -- animals -- on
- e3_p2 (e3): No Animals Were Harmed -- books -- differ -- focus -- their
- e3_p3 (e3): No Animals Were Harmed -- books -- differ -- focus -- animals
- e3_p4 (e3): No Animals Were Harmed -- books -- differ -- How
- e3_p5 (e3): No Animals Were Harmed -- books -- differ -- focus
- e3_p6 (e3): No Animals Were Harmed -- books -- differ -- focus -- in
- e3_p7 (e3): No Animals Were Harmed -- books -- differ
- e3_p8 (e3): No Animals Were Harmed -- books -- differ -- do
- e3_p9 (e3): No Animals Were Harmed -- books -- differ -- ,
- e3_p10 (e3): No Animals Were Harmed -- books -- differ -- ?
- e3_p11 (e3): No Animals Were Harmed -- books
- e3_p12 (e3): No Animals Were Harmed -- books -- the
- e3_p13 (e3): No Animals Were Harmed -- and
- e3_p14 (e3): No Animals Were Harmed -- books -- Peter Laufer
- e3_p15 (e3): No Animals Were Harmed -- books -- Forbidden Creatures
- e3_p16 (e3): No Animals Were Harmed -- Peter Laufer
- e3_p17 (e3): No Animals Were Harmed -- Peter Laufer -- books -- differ -- focus -- animals -- on
- e3_p18 (e3): No Animals Were Harmed -- Peter Laufer -- books -- differ -- focus -- their
- e3_p19 (e3): No Animals Were Harmed -- Peter Laufer -- books -- differ -- focus -- animals
- e3_p20 (e3): No Animals Were Harmed -- Peter Laufer -- books -- differ -- How
- e3_p21 (e3): No Animals Were Harmed -- Peter Laufer -- books -- differ -- focus
- e3_p22 (e3): No Animals Were Harmed -- Peter Laufer -- books -- differ -- focus -- in
- e3_p23 (e3): No Animals Were Harmed -- Peter Laufer -- books -- differ
- e3_p24 (e3): No Animals Were Harmed -- Peter Laufer -- books -- differ -- do
- e3_p25 (e3): No Animals Were Harmed -- Peter Laufer -- books -- differ -- ,
- e3_p26 (e3): No Animals Were Harmed -- Peter Laufer -- books -- differ -- ?
- e3_p27 (e3): No Animals Were Harmed -- Peter Laufer -- books
- e3_p28 (e3): No Animals Were Harmed -- books -- Peter Laufer -- of
- e3_p29 (e3): No Animals Were Harmed -- books -- Peter Laufer -- ,
- e3_p30 (e3): No Animals Were Harmed -- Peter Laufer -- books -- the
- e3_p31 (e3): No Animals Were Harmed -- Peter Laufer -- of
- e3_p32 (e3): No Animals Were Harmed -- Peter Laufer -- ,
- e3_p33 (e3): No Animals Were Harmed -- books -- Peter Laufer -- Forbidden Creatures
- e3_p34 (e3): No Animals Were Harmed -- books -- Forbidden Creatures -- Peter Laufer
- e3_p35 (e3): No Animals Were Harmed -- Peter Laufer -- books -- Forbidden Creatures
- e3_p36 (e3): No Animals Were Harmed -- Peter Laufer -- Forbidden Creatures
- e3_p37 (e3): No Animals Were Harmed -- Peter Laufer -- Forbidden Creatures -- books -- differ -- focus -- their
- e3_p38 (e3): No Animals Were Harmed -- Peter Laufer -- Forbidden Creatures -- books -- differ -- focus -- animals
- e3_p39 (e3): No Animals Were Harmed -- Peter Laufer -- Forbidden Creatures -- books -- differ -- How
- e3_p40 (e3): No Animals Were Harmed -- Peter Laufer -- Forbidden Creatures -- books -- differ -- focus
- e3_p41 (e3): No Animals Were Harmed -- Peter Laufer -- Forbidden Creatures -- books -- differ -- focus -- in
- e3_p42 (e3): No Animals Were Harmed -- Peter Laufer -- Forbidden Creatures -- books -- differ
- e3_p43 (e3): No Animals Were Harmed -- Peter Laufer -- Forbidden Creatures -- books -- differ -- do
- e3_p44 (e3): No Animals Were Harmed -- Peter Laufer -- Forbidden Creatures -- books -- differ -- ,
- e3_p45 (e3): No Animals Were Harmed -- Peter Laufer -- Forbidden Creatures -- books -- differ -- ?
- e3_p46 (e3): No Animals Were Harmed -- Peter Laufer -- Forbidden Creatures -- books
- e3_p47 (e3): No Animals Were Harmed -- books -- Forbidden Creatures -- Peter Laufer -- of
- e3_p48 (e3): No Animals Were Harmed -- books -- Forbidden Creatures -- Peter Laufer -- ,
- e3_p49 (e3): No Animals Were Harmed -- Peter Laufer -- Forbidden Creatures -- books -- the

## 8. LLM Selected Entity Paths
- e1: e1_p3 Peter Laufer -- books -- differ -- focus -- animals
  Reason: This path effectively connects Peter Laufer to the focus on animals in his books, providing a clear reasoning chain for the comparison.
- e2: e2_p3 Forbidden Creatures -- books -- differ -- focus -- animals
  Reason: This path connects Forbidden Creatures to the focus on animals, allowing for a direct comparison with No Animals Were Harmed.
- e3: e3_p3 No Animals Were Harmed -- books -- differ -- focus -- animals
  Reason: This path connects No Animals Were Harmed to the focus on animals, facilitating a direct comparison with Forbidden Creatures.

## 9. Selected Path Semantic Transduction
Nodes:
- peter_laufer: Peter Laufer (entity)
- books_e1: books (type_variable)
- books_e2: books (type_variable)
- books_e3: books (type_variable)
- focus_e1: focus (type_variable)
- focus_e2: focus (type_variable)
- focus_e3: focus (type_variable)
- animals_e1: animals (value_slot)
- animals_e2: animals (value_slot)
- animals_e3: animals (value_slot)
- forbidden_creatures: Forbidden Creatures (entity)
- no_animals_were_harmed: No Animals Were Harmed (entity)

Edges:
- peter_laufer -> books_e1 (books of Peter Laufer)
- books_e1 -> focus_e1 (focus of books)
- books_e2 -> focus_e2 (focus of books)
- books_e3 -> focus_e3 (focus of books)
- focus_e1 -> animals_e1 (focus on animals)
- focus_e2 -> animals_e2 (focus on animals)
- focus_e3 -> animals_e3 (focus on animals)
- forbidden_creatures -> books_e2 (books of Forbidden Creatures)
- no_animals_were_harmed -> books_e3 (books of No Animals Were Harmed)

## 10. Atomic Subquestion DAG
- None: What are the books of Peter Laufer?
- None: What is the focus of the books of Peter Laufer?
- None: What is the focus on animals in the books of Peter Laufer?
- None: What are the books of Forbidden Creatures?
- None: What is the focus of the books of Forbidden Creatures?
- None: What is the focus on animals in the books of Forbidden Creatures?
- None: What are the books of No Animals Were Harmed?
- None: What is the focus of the books of No Animals Were Harmed?
- None: What is the focus on animals in the books of No Animals Were Harmed?
