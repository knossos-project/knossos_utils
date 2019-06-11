import re


def nat_sorted(sequence):
    def convert(val): return int(val) if val.isdigit() else val
    return sorted(sequence, key=lambda elem: [convert(elem) for elem in re.split(r'(\d+)', elem)])    


def compare_version(v1, v2):
    """
    Return '>', '<' or '==', depending on whether the version
    represented by the iterable v1 is larger, smaller or equal
    to the version represented by the iterable v2.
    """

    v1 = v1.split('.')
    v2 = v2.split('.')

    if len(v1) > len(v2):
        v2.extend(["0"] * (len(v1) - len(v2)))
    elif len(v1) < len(v2):
        v1.extend(["0"] * (len(v2) - len(v1)))

    v1v2 = [(v1[x], v2[x]) for (x, y) in enumerate(v1)]

    for (cur_v1, cur_v2) in v1v2:
        if cur_v1 == cur_v2:
            pass
        elif nat_sorted((cur_v1, cur_v2))[0] == cur_v1:
            return '<'
        else:
            return '>'

    return '=='


class Version(list):
    def __lt__(self, other):
        assert isinstance(other, Version)
        
        return compare_version(self, other) == '<'

    def __le__(self, other):
        assert isinstance(other, Version)

        return compare_version(self, other) == '<' or \
               compare_version(self, other) == '=='

    def __gt__(self, other):
        assert isinstance(other, Version)

        return compare_version(self, other) == '>'
    def __ge__(self, other):
        assert isinstance(other, Version)

        return compare_version(self, other) == '>' or \
               compare_version(self, other) == '=='

    def __eq__(self, other):
        assert isinstance(other, Version)

        return compare_version(self, other) == '=='
    def __ne__(self, other):
        assert isinstance(other, Version)

        return not compare_version(self, other) == '=='
