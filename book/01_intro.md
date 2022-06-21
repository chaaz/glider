# Syntax Intro

Let's get straight into it, shall we? This is an example of a script
written in the Glider language:

```
value = 0;
value
```

Every script is just a _block_ of code.

A **block** is just zero or more _assignments_ followed by a optional
return _expression_. Our example has a single assignment: `value = 0`,
followed by the return expression `value`.

An **expression** is something that you can calculate into a single
value. Expressions can be things like _literals_, _variables_, _arrays_,
_objects_, _math_ and _logic_ operations, _if/else_ conditions,
_indexes_, _functions_, and _function calls_.

Here's line 2 from our example, which has the script's return
expression:

```
value
```

Easy, right? This is a type of expression called a **variable**.
Variables are names made of letters, numbers, and the underscore
character, although they always start with a letter. `value` is a valid
variable; so is `the_great_bandini`, `hax0rA23`, or even just `i`. The
value of a variable is just whatever value was stored into that
particular name.

And how to you store a value into a variable? An **assignment** has the
form "_target_ = _expression_ ;". (There's always a trailing semicolon
to show where an assignment ends.) The target is usually a variable, but
can be a _destructure_ if you want to assign to more than one variable
at once; we'll talk more about those later.

Here's line 1 from our example, which has the single assignment:

```
value = 0;
```

The expression on the right side of the assignment is another type of
expression called a **literal**. Literal values are the simplest
expression: they evaluate to the thing that's actually typed in the
code, whether it's a bit of text, a boolean `true` or `false`, or--like
we have here--a literal number.

When the program runs, it first runs the assignment: it evaluates the
right-hand side expression (our literal `0`) and then saves that to the
variable named `value` on left-hand side. Using variables like this is
nice for a couple reasons:

- It's usually easier to understand a variable's purpose by reading its
  name than its value. If you see the number `17` everywhere, it might
  be tough to figure out what it is, but if you see `age` instead, it
  might make more sense.

- Variables are assign-once, use-many. If you ever want to rewrite the
  variable's value, you only have to go the variable assignment, rather
  than all the places where the variable is used.

After the assignment, the script exits with the return value `value`,
which contains the value `0` that we just assigned.
