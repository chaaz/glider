(or : just run types until we hit steady state)


# Detecting Recursive Call Compliance

## The Problem

When developing a strongly typed monomorphizing language such as Glider,
we run into the general problem of dealing with recursive function calls
that have derivative types. Consider the following example: (Note:
currently, direct recursion is not allowed by in Glider because
referencing a variable in its declaration is not allowed. However,
recursion can still be achieved via the y-combinator or similar. In
these examples, we pretend that recursion is allowed directly in order
to simplify the code.)

```
f = fn(a) { if dynamic_condition(a) { f(push(a, 1)) } else { a.1 } }
f([0, 1])
```

Under naive monomorphizing conditions, we'd want to create a new
implementation for each distinct argument type. We'd first compile for
`f([int, int])`, and then `f([int, int, int])`, and then `f([int, int,
int, int])`, _ad nauseum_ until the compiler ran of memory. We use
`dynamic_condition` here to show that the compiler can't always use a
static constexpr condition (e.g. `len(a)`) to solve this problem.

## Current Work

The Glider compiler solves this by differentiating between a "recursive
call" and "non-recursive" a.k.a. "base" call. A **recursive call** is
one where the function definition is already invoked somewhere in the
stack, i.e. the same function is a direct or indirect caller. This
definition is agnostic of a particular implementation: in the example
above, `f([int, int, int])` is a recursive call because it's called by
`f([int, int])`, even though they have different argument types. When
making a recursive call, Glider requires that the recursive caller
exactly match the argument types of the callee so that they have the
same implementation, and returns an error if they do not, such as the
example above. This is _very_ roughly analogous to the `let .. in ..`
binding mechanism of Standard ML, in which only specific monomorphisms
defined in `let` can be used for `in`.

## Improvement

In the example above, the only non-recursive use of the input argument
`a` is the dot-operator `a.1`. Thus, the implementation for `f([int,
int])` is compatible with `f([int, int, int])`. By **compatible** we
mean that the bytecode generated for the two-int version is identical to
three-int version, so those implementations could be safely unified and
re-used in a recursive context. (We ignore for now potential execution
changes that treat those stack variables differently for some reason,
but we'll revisit that case.)

While it's tempting to just look-ahead to the 3-int version, check that
it can be unified, and then cause the function implementation to re-call
itself as if it were re-using the same type; we must be more cautious.
Recursion calls go on _forever_, potentially, so we need to look ahead
to the 4-int case, 5-int case, etc. Needing to perform an infinite
number of unifications is no better than needing an infinite number of
implementations; there must be a better way.

## Solution

We're going to explore if we can effectively perform an infinite series
calculation in finite time. In much the same way that we can predict
that the sum of the geometric series `1/2 + 1/4 + 1/8 ... + 1/(n^2)`
converges to `1`, we may be able to programmatically predict that the
example function given above is safe to unify recursively for all arrays
whose length are least length 2.

To do this effectively, we use the concept of a **partial type**. The
type `[int, int]` is a "full" or "exact" type, but has more constraints
than necessary to satisfy the non-recursive uses in our example.
Instead, we consider the partial type `[_, int, ..]`, where `_` means
"any type", and `..` means "zero or more elements of any type". There is
really on a single **type constraint** here: that the second element of
the array has the type `int`. (We'll write that as `a.1==int`.) We
construct a partial type first by examining the actual argument passed
to the function call on line 2, and then building its constraints based
on usage. Glider compiles with full access to all source code, so it
will be able to trace through functions to construct a detailed
production of every use.

Partial types and type constraints only apply to complex types such as
arrays and objects. Primitive types like `int` can't get any simpler or
less restrictive. For array types, there is really only one kind of
constraint: that an element at some index has some type. Of course, that
type might itself be the partial type of an array, and so constraints
must be able to be recursively defined: something like `[_, [_, int,
..], ..]` should be possible.

Once we've constructed partial types for a function's arguments, we then
need to consider the operations which can be performed to convert them
to the recursive call. For arrays, there are four such operations:

1. Inserting a value into the array at a static point. 
2. Pushing a new value onto the end of the array.
3. Changing a value at a static point into another value.
4. Deleting a value (and shifting all later values over one)
5. Deleting the last element of the array.

For 1. and 2., the new value's type can either match the type of some
input (including itself), or it can be a fixed type.

Every operation to construct an array is thus composed of one or more of
these operations.

## Basic constraints

Let's examine how each operation has an effect on constraints.

### Inserting / Pushing a Value

If we have a constraint `a.2==int`, then inserting a new element _after_
index `2` has no effect on the constraint: the type of the second
element is unchanged. If we insert it _before_, however, then the
constraint shifts: if element `2` of the new array must be an int, that
means that the element `1` of the original array must also be an int. If
we insert at exactly element `2`, then the new value we are inserting
must be an `int`.

Pushing to the end of the array is just like inserting after a
constraint's index: it has no effect on the constraint.

### Setting a Value

This is a simple one: if the constraint is `a.2==int`, and we set the
value of `a.2`, this means that whatever we set the value to must be an
int. Additionally, if we set a value at exactly the constraint, it has
the effect of erasing that constraint. Setting a value at a different
index has no effect on the constraint.

### Deleting / Popping a Value

Like insertion, this has no effect on constraints with an index after
the deleted value. For constraints with indexes that come on or before
the deleted item, the constraint shifts in the opposite direction: a
constraint `a.2==int` will transform to `a.3==int` on the original
array. If the constraint refers to the last element of an array, this
will cause the constraint to fail.

Deleting the last element of the array is just like deleting a value on
or after the constraint's index: it causes the constraint to fail
if the constraint index is the last element; but has no effect
otherwise.

### Composing Operations

Each basic operation works as a transformation of a constraint; when we
consider composite operations, they simply apply that transformation in
the reverse order.

### Putting it all together

We call the complete set of operations that convert one set of types
into another a **composition**. When we discuss how a recursive call's
input is made from a caller input, that's a **call composition**; when
we talk about how a function's output is constructed from it's input,
that's a **return composition**. A function's composition, along with
any additional constraints it imposes on its input is known as the
function's **production**. The goal for each function is to analyze its
code to find the minimal correct constraints on its input, and to then
determine if its recursive call compositions can still satisfy those
constraints.

Of course, we've already discussed that simply looking one step into the
future isn't sufficient: we need to continue that indefinitely. However,
we note that every transform is linear in nature, and that some use
patterns show up:

- Pushing to the end has no effect on any constraints (except that it
  may prevent the array from shrinking).
- If there are more pops/deletes than inserts, the array will eventually
  shrink down to an untenable state: recursion will fail, even if there
  are no constraints.
- Location changes from inserts/deletes will repeat at regular intervals
  through the bounds of the insert/delete, and will eventually target
  item(s) that are inserted.

So, the "minimal correct constraints" has to be constraints that apply
to all future iterations of the argument. We term the constraints that
apply to the current function the **current constraints** (which are
part of the current production and current partial type), the
constraints that apply to the next iteration of a recursive functions
the **follow constraints** and the constraints that must apply to all
iterations the **recursive constraints**.

TODO: HOW?

### Generalized Constraints

Ideally, we want to consider constraints that extend beyond the bounds
of the original arguments, so that more generalized arguments also work.

For example, if we find a call site f([1, 2, 3, 4, 5]), and we determine
that the singular constraint is `a.2==int` and the recursive composition
is `(push (del 2) int)`, this leads to the recursive partial type `[_,
_, int, int, int]`. We'd like to generalize that to `[_, _, int, int..]`
so that if ever come across a call to e.g. `f([1, 2, 3, 4, 5, 6, 7])`, we
can be sure that the implementation can be used there as well.

TODO: HOW?

### Native functions


