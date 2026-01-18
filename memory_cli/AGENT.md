You are writing code in Rust 2018 edition. Follow these project layout rules exactly.

1. Crate structure

* This is a CLI deliverable with a library core.
* Always create:

  * src/lib.rs (library root)
  * src/bin/.rs (CLI entrypoint; thin)
* The CLI entrypoint must only:

  * parse CLI args
  * call into library functions
  * handle printing/logging/errors
* Do NOT put business logic, SQL, or large workflows in src/bin/*.

2. File-module style (NO mod.rs anywhere)

* Never create mod.rs files.
* For every module named X:

  * module root is src/X.rs
  * submodules live in folder src/X/
* Example:

  * src/contracts.rs declares: pub mod fresh_paper; pub mod get_best_run;
  * src/contracts/fresh_paper.rs and src/contracts/get_best_run.rs contain implementations.
* You may have src/X.rs even if X currently has only one submodule (keep the pattern consistent).

3. Module grouping convention
   Use the following top-level modules (create them when applicable):

* src/cli.rs + src/cli/…       (CLI parsing/types only)
* src/contracts.rs + src/contracts/… (external JSON I/O types; request/response)
* src/commands.rs + src/commands/…   (use-case handlers invoked by CLI)
* src/db.rs + src/db/…         (DB access: migrations, repositories, PRAGMAs, connections)
* Additional modules are allowed, but must follow the same file-module root + folder pattern.

4. Library root responsibilities (src/lib.rs)

* src/lib.rs must only define and re-export modules:

  * pub mod cli;
  * pub mod contracts;
  * pub mod commands;
  * pub mod db;
* Do not implement large logic directly in lib.rs.

5. Placeholder policy

* Do NOT create placeholder/stub .rs implementations for future features.
* All future work, notes, and planned commands must go into a root-level TODO.md.
* Only add code when the feature is intended to compile, run, and be used.

6. Documentation and supporting assets

* Keep docs in:

  * docs/ (design notes, contracts overview)
* Keep DB schemas in:

  * schemas/ (e.g., schema.sql)
* Keep example payloads in:

  * examples/ (e.g., JSON inputs)
* Keep project planning notes in:

  * TODO.md (required)

7. Consistency requirements

* The folder/module layout must remain consistent across the project.
* Do not mix styles (no mod.rs; no ad-hoc module trees).
* Any new module must follow the file-module root + folder submodules pattern.
