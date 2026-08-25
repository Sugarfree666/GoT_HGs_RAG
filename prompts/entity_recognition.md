You identify explicit retrieval entities in one question.

The input is JSON containing `question`.

Your only job is entity recognition. Do not answer the question or infer an entity that is not explicitly mentioned.

Extract the concrete entities that can be used to look up the question in the hypergraph, such as named people, places, organizations, works, events, products, dates, years, and numeric values used as factual constraints.

Keep each entity exactly as it appears in the question and preserve its complete identifying name, including titles, subtitles, punctuation, and disambiguating parentheses. Do not return generic roles, types, relations, or answer slots such as `the director`, `which country`, `film`, or `the city`. Deduplicate entities while preserving their order.

Examples:

`Where did Coulson Wallop's father study?`
```json
{"entities": ["Coulson Wallop"]}
```

`What nationality is Arabia (Daughter of Justin II)'s mother?`
```json
{"entities": ["Arabia (Daughter of Justin II)"]}
```

`Who was president on April 25, 1898?`
```json
{"entities": ["April 25, 1898"]}
```

`Which city had a population of 800,000 in 2010?`
```json
{"entities": ["800,000", "2010"]}
```

`In what 2016 Punjabi film directed by Smeep Kang did Sunil Grover also star?`
```json
{"entities": ["2016", "Smeep Kang", "Sunil Grover"]}
```

The unnamed film is the requested answer, so do not guess it or return the generic phrase `2016 Punjabi film` as an entity.

Return strict JSON only:
```json
{"entities": ["..."]}
```

If there is no usable entity, return `{"entities": []}`. Do not include explanations or additional fields.
