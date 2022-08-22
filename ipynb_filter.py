#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.

# `ipynb_filter.py`:
# This is a git filters that strips out the outputs and
# meta data of a Jupyer notebook using `nbconvert`.
# Execute the following line in order to activate this filter:
# python ipynb_filter.py
#
# The following line should be in `.gitattributes`:
# *.ipynb filter=ipynb_filter
#
# from github.com/basnijholt/ipynb_git_filters

from nbconvert.preprocessors import Preprocessor


class RemoveMetadata(Preprocessor):
    def preprocess(self, nb, resources):
        nb.metadata = {
            "language_info": {"name": "python", "pygments_lexer": "ipython3"}
        }
        return nb, resources


if __name__ == "__main__":
    # The filter is getting activated
    import os

    cmd = "jupyter nbconvert --to notebook --config ipynb_filter.py --stdin --stdout"
    git_cmd = f'git config filter.ipynb_filter.clean "{cmd}"'
    os.system(git_cmd)  # nosec
else:
    # This script is used as config
    c.Exporter.preprocessors = [RemoveMetadata]  # noqa: F821
    c.ClearOutputPreprocessor.enabled = True  # noqa: F821
    c.ClearOutputPreprocessor.remove_metadata_fields = [  # noqa: F821
        "deletable",
        "editable",
        "collapsed",
        "scrolled",
    ]
