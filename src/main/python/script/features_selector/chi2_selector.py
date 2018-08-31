
from sklearn import feature_selection

class Chi2SelectorTask:

    def __init__(self, X, y, alpha):
        self.__X__ = X
        self.__y__ = y
        self.__alpha__ = alpha

    def __str__(self):
        s = "Chi2 Selector Task"
        return s

    def chi2_pvalue(self):
       chi2, p_value = feature_selection.chi2(self.__X__, self.__y__)
       return p_value
