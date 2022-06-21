# Glider

**Glider** is a dramatically simple programming language. It is
functional, immutable, and strongly typed via inference. It also has no
looping constructs, no early return; strict operators; and uses a JSON
type to express dynamic objects.

See _The Book_ for more information about Glider features and
principles.

## Quick Start

Versio is a binary written in the Rust programming language. If you have
[installed Rust](https://www.rust-lang.org/tools/install), you can do
this:

```
$ cargo install glider
```

And then you can run this little gem:

```
$ cat > my_script.gli << EOF

f = fn([f, a]) { f(a) };
a = fn(a) { a };
f([a, 3])
EOF

$ glider -s my_script.gli

Done: 3
```

See _The Book_ for more information about the language syntax.

## Background

Glider was developed to satisfy the need for a very simple, very
functional language scripts that could manage complex processes at a
high level--for example, managing an HTTP request/response
stream--without getting bogged down in details.

## Contributing

We would love your contributions to Glider! Feel free to branch or fork
this repository and submit a pull request.

`glider` is written in Rust, a powerful and safe language for writing
native executables. Visit the Rust lang
[homepage](https://www.rust-lang.org/en-US/index.html) to learn more
about writing and compiling Rust programs, and see the
[Contributing](docs/contributing.md) page for Versio specifically.

We also happily accept ideas, suggestions, documentation, tutorials, and
any and all feedback. Leave a message on the support pages of this repo,
or send messages directly to its owners.
