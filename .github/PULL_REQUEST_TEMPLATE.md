<!-- Please read the following guidelines for new Pull Requests -- thank you! -->

<!--
Make sure that you submit this pull request as a separate topic or feature branch and not as master branch. The new feature branch of your fork will then be merged to the master branch of the original repository, following the "Fork-and-Branch Git Workflow:"

1. Fork the original GitHub project
2. Clone the fork to your local machine
3. Create a new topic branch
4. Make your code changes to this new topic branch
5. Commit the changes and push the commit to the topic branch to your fork upstream on GitHub
6. Create a new pull request from the upstream topic branch to the master branch of the original repo
-->

<!-- Provide a small summary describing the Pull Request below -->

### Description

Insert Description Here

### Related issues or pull requests

<!-- Please provide a link to the respective issue on the [Issue Tracker](https://github.com/rasbt/mlxtend/issues) if one exists. E.g.,

Fixes #<ISSUE_NUMBER> -->

Link related issues/pull requests here

<!-- Below is a general todo list for typical pull request -->

### Pull Request requirements

- [ ] Added appropriate unit test functions in the `./mlxtend/*/tests` directories
- [ ] Ran `nosetests ./mlxtend -sv` and make sure that all unit tests pass
- [ ] Checked the test coverage by running `nosetests ./mlxtend --with-coverage`
- [ ] Checked for style issues by running `flake8 ./mlxtend`
- [ ] Added a note about the modification or contribution to the `./docs/sources/CHANGELOG.md` file
- [ ] Modify documentation in the appropriate location under `mlxtend/docs/sources/` (optional)
- [ ] Checked that the Travis-CI build passed at https://travis-ci.org/rasbt/mlxtend




<!--
NOTE

Due to the improved GitHub UI, the squashing of commits is no longer necessary.
Please DO NOT SQUASH commits since they help with keeping track of the changes during the discussion).

For more information and instructions, please see http://rasbt.github.io/mlxtend/contributing/
-->
