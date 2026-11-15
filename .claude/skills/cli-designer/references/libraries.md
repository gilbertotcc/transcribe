# CLI libraries

Prefer the library the project already uses; otherwise pick the default (listed
first) for the language. Helpers must honor [NO_COLOR](https://no-color.org),
disable color when the output stream isn't a TTY, and prompt only when
stdin is a TTY.

## Python

- [Typer](https://github.com/fastapi/typer): default; typed
  options from type hints, subcommands, generated help and completion.
- [Click](https://github.com/pallets/click): mature, explicit
  decorators; choose for fine control or plugin-style command groups.
- [argparse](https://docs.python.org/3/library/argparse.html): stdlib; choose
  when no dependencies are allowed and the CLI is small.

Helpers: `sys.stdout.isatty()`; [Rich](https://github.com/Textualize/rich)
for color, tables and progress.

## Go

- [Cobra](https://github.com/spf13/cobra): default; nested
  subcommands, generated help, shell completion.
- [urfave/cli](https://github.com/urfave/cli): lighter; flags and
  subcommands with less ceremony.
- [Kong](https://github.com/alecthomas/kong): struct-tag
  declarations; typed options and nested commands.

Helpers: [golang.org/x/term](https://github.com/golang/term)
(`term.IsTerminal`); [fatih/color](https://github.com/fatih/color)
or [lipgloss](https://github.com/charmbracelet/lipgloss).

## Rust

- [clap](https://github.com/clap-rs/clap) (derive API): default;
  typed options, subcommands, generated help, completion via `clap_complete`.
- [argh](https://github.com/google/argh): small and fast to
  compile; choose for simple CLIs with few features.

Helpers: [`std::io::IsTerminal`](https://doc.rust-lang.org/stable/std/io/trait.IsTerminal.html);
[anstream](https://github.com/rust-cli/anstyle) (strips
color when piped) or [owo-colors](https://github.com/owo-colors/owo-colors).

## JavaScript/TypeScript (Node.js)

- [Commander](https://github.com/tj/commander.js): default;
  subcommands, generated help, typed options with TypeScript.
- [yargs](https://github.com/yargs/yargs): rich option parsing and
  completion generation.
- [oclif](https://github.com/oclif/oclif): framework for large,
  plugin-based multi-command CLIs.

Helpers: [`process.stdout.isTTY`](https://nodejs.org/api/tty.html);
[picocolors](https://github.com/alexeyraspopov/picocolors) (tiny)
or [chalk](https://github.com/chalk/chalk).

## Java/Kotlin

- [picocli](https://github.com/remkop/picocli): default; annotated
  options, subcommands, generated help and completion; Java and Kotlin.
- [Clikt](https://github.com/ajalt/clikt): idiomatic Kotlin DSL;
  typed options, subcommands, completion.

Helpers: `System.console()`; [JLine](https://github.com/jline/jline3)
or [Jansi](https://github.com/fusesource/jansi) for ANSI.

## C#/.NET

- [System.CommandLine](https://github.com/dotnet/command-line-api):
  default; first-party, typed options, subcommands, help, completion.
- [Spectre.Console.Cli](https://github.com/spectreconsole/spectre.console.cli):
  command classes with rich help output; pairs with Spectre.Console.

Helpers: `Console.IsOutputRedirected`;
[Spectre.Console](https://github.com/spectreconsole/spectre.console)
for color, tables and progress.

## Ruby

- [Thor](https://github.com/rails/thor): default; subcommands,
  generated help, widely used (Rails generators).
- [dry-cli](https://github.com/dry-rb/dry-cli): command objects;
  choose for a modular, testable command registry.

Helpers: `$stdout.tty?`; [pastel](https://github.com/piotrmurach/pastel)
for color and [tty-prompt](https://github.com/piotrmurach/tty-prompt)
for prompts.

## Shell (POSIX sh/Bash)

- [`getopts`](https://man7.org/linux/man-pages/man1/getopts.1p.html):
  default; built-in, portable short options only; handle `--help` and
  `--version` by hand. Move to another language when subcommands, typed
  options or completion are needed.

Helpers: `[ -t 1 ]` (stdout is a TTY); `tput` for portable color codes, unset
when `NO_COLOR` is set or `TERM=dumb`.
