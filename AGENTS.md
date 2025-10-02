# Working Principles

- Ask the user if an imported symbol points to code that is missing from the repository before proceeding.
- Define any new variables or helper methods as close as possible to their point of use.
- Avoid introducing temporary variables when a value is only needed once.
- Add only the guards that the surrounding logic actually requires.
