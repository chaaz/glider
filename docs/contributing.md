# Contributing to Glider

This page is for developers that want to contribute to the Glider
language. It assumes that you already have basic familiarity with
building Rust programs.

See ex. [the book](https://doc.rust-lang.org/book/index.html), or
elsewhere on the internet for help getting started with Rust.

Also, make sure you've read the project [README](../README.md) and
[Dependency page](./dependencies.md) so that you have an idea of how
Glider must be installed.

## Project structure

Here is the structure of "glider":

```
glider
├─ LICENSE.md
├─ README.md
├─ docs
│  ├─ contributing.md  . . . . . .  this document
│  └─ ...
├─ Cargo.toml  . . . . . . . . . .  project file
├─ Cargo.lock  . . . . . . . . . .  deps locking
├─ rustfmt.toml  . . . . . . . . .  format config
├─ src
│  └─ ...          . . . . . . . .  src and unit tests
└─ tests
   └─ ...          . . . . . . . .  integration tests
```

## Running

The `glider` app is very simple with minimal runtime dependencies; you
can run it locally just with `glider -s <script>`.

## Dev Guidelines

[dev guidelines]: #dev-guidelines

Here are the development guidelines for Versio. In order to practice
them, you may need to install some tools:

```
$ rustup toolchain install nightly
$ rustup component add rustfmt --toolchain nightly
$ cargo install cargo-audit
$ rustup component add clippy
```

### Warnings

Unless there's a very good reason, you should never commit code that
compiles with warnings. In fact, it is suggested that you set the
`RUSTFLAGS='-D warnings'` before building, which treats all warnings as
errors. Most rust warnings have reasonable work-arounds; use them.

For example, "unused variable" warnings can be suppressed by starting
the variable name with an underscore (`_thing`). Of course, it's always
better to re-factor the code so that the variable doesn't exist, where
possible.

### Style

We generally adhere to the principles of "Clean Code", as described in
the first half of [Clean
Code](https://www.amazon.com/Clean-Code-Handbook-Software-Craftsmanship-ebook-dp-B001GSTOAM/dp/B001GSTOAM/ref=mt_kindle?_encoding=UTF8&me=&qid=1541523061).
This means well-chosen names; small, concise functions; limited, well
written comments; and clear boundaries of abstraction.

We also follow best Rust and Cargo practices: using references,
iterators, functional techniques, and idiomatic use of `Option`,
`Result`, `?` operator, and the type system in general. Most of this is
shown clearly in the [the
book](https://www.amazon.com/Clean-Code-Handbook-Software-Craftsmanship-ebook-dp-B001GSTOAM/dp/B001GSTOAM/ref=mt_kindle?_encoding=UTF8&me=&qid=1541523061)

### Documentation

You should keep all technical documentation--including the top-level
README, code comments, and this document--up-to-date as you make changes
to tests and code.

### Coding Format

**Always format your code!** You should format your (compiling, tested)
code before submitting a pull request, or the PR will be rejected.

We use the nightly [rust
formatter](https://github.com/rust-lang-nursery/rustfmt), and have a
`rustfmt.toml` file already committed for its use. Run `cargo +nightly
fmt -- --check` to preview changes, and `cargo +nightly fmt` to apply
them.

### Linting

**Always lint your code!** You should lint your code before submitting a
pull request, or the PR will be rejected.

[Clippy](https://github.com/rust-lang/rust-clippy) is the standard cargo
linter. Run `cargo clippy` to run all lint checks.

### Security/Dependency Auditing

**Always audit your dependencies!** If you don't, your PR will be
rejected.

We use the default [cargo audit](https://github.com/RustSec/cargo-audit)
command. Run `cargo audit --deny-warnings` to perform a vulnerability
scan on all dependencies. The `--deny-warnings` flag treats warnings as
errors.

### Testing

**Always run tests** Obviously, if your code fails any unit or service
test, then your PR will be rejected.

Any new modules created should have their own set of unit tests.
Additions to modules should also expand that module's unit tests. New
functionality should expand the application's integration tests. We
don't currently enforce test coverage in Glider, but we may at a later
date.

Run `cargo test` to run all unit tests.

## Platform-specific help

[platform-specific help]: #platform-specific-help

### Linux

[linux]: #linux

When building on linux, you should make sure that the `strip =
"symbols"` is set on your Cargo.toml in the `profiles.release` section.
Otherwise, the linux binary can be enormous in size. If you don't want
to use that, you can set the environment variable:

```
$ export RUSTFLAGS='-D warnings -C link-args=-s'
```

the `strip` or `link-args=-s` options pass `--strip-debug` to the
linker, and ensures that the resulting executable is a reasonable size:
without that option, the binary easily expand to over 100M. If you
forget to include this option, you should manually run `strip` on the
resulting executable.

### Windows

[windows]: #windows

We compile using the MSVC toolchain (which is the default), so you'll
need to install either Visual Studio (Community Edition 2017 works), or
install the MSVC runtime components. Make sure you install the C/C++
base components during the installation dialog. If you try to install
Rust without these, it will provide intructions.

### MacOS

[macos]: #macos

No special build instructions for MacOS.
