# TODO: implement a bibtex normalizer script
"""
Example:
    Bring single line string:
@ARTICLE{6634187,  author={Biswas, Ayan and Dutta, Soumya and Shen, Han-Wei and Woodring, Jonathan},  journal={IEEE Transactions on Visualization and Computer Graphics},   title={An Information-Aware Framework for Exploring Multivariate Data Sets},   year={2013},  volume={19},  number={12},  pages={2683-2692},  doi={10.1109/TVCG.2013.133}}

    To appropriate format (and set informative key):
@ARTICLE{Biswas2013ExploringMultivariateDataSets,
    author      = {Biswas, Ayan and Dutta, Soumya and Shen, Han-Wei and Woodring, Jonathan},
    journal     = {IEEE Transactions on Visualization and Computer Graphics},
    title       = {An Information-Aware Framework for Exploring Multivariate Data Sets},
    year        = {2013},
    volume      = {19},
    number      = {12},
    pages       = {2683-2692},
    doi         = {10.1109/TVCG.2013.133}
}

"""
