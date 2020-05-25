============
Contributing
============

Contributions are welcome, and they are greatly appreciated!

Bug reports
===========

When `reporting a bug <https://github.com/garrettwrong/cuTWED/issues>`_ please include:

    * Your operating system name and version.
    * Your CUDA Toolkit version, driver, and card(s).
    * Any details about your local setup that might be helpful in troubleshooting.
    * Detailed steps to reproduce the bug.

Documentation improvements
==========================

cuTWED could always use more documentation, whether as part of the
official cuTWED docs, in docstrings, or even on the web in blog posts,
articles, and such.

Feature requests and feedback
=============================

The best way to send feedback is to file an issue at https://github.com/garrettwrong/cuTWED/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a voluntary project.  You are welcome to improve things and push the improvements back to the author for review. Consider discussing such work in the issue.

Development
===========

To set up `cuTWED` for local development:

1. Fork `cuTWED <https://github.com/garrettwrong/cuTWED>`_
   (look for the "Fork" button).
2. Clone your fork locally::

    git clone git@github.com:garrettwrong/cuTWED.git

3. Create a branch for local development::

    git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

4. When you're done making changes run all the checks and docs builder with `tox <https://tox.readthedocs.io/en/latest/install.html>`_ one command::

    tox

5. Commit your changes and push your branch to GitHub::

    git add .
    git commit -m "Your detailed description of your changes."
    git push origin name-of-your-bugfix-or-feature

6. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

If you need some code review or feedback while you're developing the code just make the pull request as a draft.

For merging, you should:

1. Include passing tests (run ``tox``) [1]_  [2]_.
2. Update documentation when there's new API, functionality etc.
3. Add a note to ``CHANGELOG.rst`` about the changes.
4. Add yourself to ``AUTHORS.rst``.

.. [1] If you don't have all the necessary python versions available locally you can try running against what you have with ```tox --skip-missing-interpreters```.
.. [2] You may find the basic Docker containers helpful.  They can be easily extended and still kept in isolation.  See `.gitlab-ci.yml` for a basis.

Tips
----

To run a subset of tests::

    tox -e envname -- pytest -k test_myfeature

To run all the test environments in *parallel* (you need to ``pip install detox``)::

    detox
