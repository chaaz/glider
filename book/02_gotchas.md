# Common Pitfalls

The Glider language is different from many other programming languages
in some key ways. Here a few things to pay attention to, if you're
coming from more traditional programming languages:

- Full immutability
  - No internal sets: `a.b = 3`

- Variable shadows
  - Impact on closure

- No using a variable in itself
  - How to recursive

- Lack of loops
  - Requires comprehension
  - No breaks

- If/Else as expression (else required)

- No early return

- Inferred types / monomorphism
  - If/Else matching
  - JSON escape hatch
  - Operation strictness
  - Indexing strictness


Normal language pitfalls that we keep:

- and/or shortcuts
  - `or` lower precedence
