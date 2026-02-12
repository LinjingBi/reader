# Lab Directory

This lab directory is used to run any reader, memo integration tests, and data analysis.

## Directory Structure

### `data/`
Holds the original/raw data/dataset files used for testing and analysis.

**Current Items:**
- `hf_2025_monthly_papers.json` - HuggingFace monthly papers dataset from January 2025 to December 2025, containing paper metadata including IDs, authors, and related information grouped by month. Used for testing reader and memo integration with real-world paper data.

### `db/schema/`
Holds different memo database schemas for different test purposes. These schema files define the structure of test databases.

**Current Items:**
- *(empty - schemas will be added here as needed)*

### `db/snapshot/`
Holds different SQLite database files for different test purposes. These are pre-built SQLite databases that are ready to be consumed by the memo CLI.

**Current Items:**
- *(empty - snapshot databases will be added here as needed)*

### `scripts/`
Holds the logic for:
- Data processing
- Running schemas to build databases
- Filling databases with test data
- Creating snapshots
- Partial reader functions

**Current Items:**
- *(empty - scripts will be added here as needed)*

### `memo/`
Holds different memo binaries that will be used in different test scenarios.

**Current Items:**
- *(empty - memo binaries will be added here as needed)*

## Rules

**Important:** Every new item added to any subdirectory must:
1. Declare its usage or version in this README, OR
2. Be documented with a summary explaining its purpose

When adding new items, update the relevant section above with:
- File/directory name
- Brief description of its purpose
- Version information (if applicable)
- Usage instructions (if applicable)

