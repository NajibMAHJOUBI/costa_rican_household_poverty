
from imblearn.over_sampling import ADASYN, SMOTE


class OverSamplingTask:

    def __init__(self, features, label):
        self.__features__ = features
        self.__label__ = label

    def __str__(self):
        s = "Over Sampling Task"
        return s

    def adasyn(self):
        ada = ADASYN()
        return ada.fit_sample(self.__features__, self.__label__)

    def smote(self):
        smo = SMOTE(kind="svm")
        return smo.fit_sample(self.__features__, self.__label__)
