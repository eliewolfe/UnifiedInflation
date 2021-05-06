import numpy as np


def target_distribution_gen(name, parameter1, parameter2):
    """ parameter1 is usually a parameter of distribution (not always relevant). parameter2 is usually noise."""

    if name == "Fritz-visibility":
        """ parameter2 is the visibility"""
        ids = np.zeros((4, 4, 4)).astype(str)
        p = np.zeros((4, 4, 4))
        for i, j, k, l, m, n in product('01', repeat=6):
            a = int(i + j, 2)
            b = int(k + l, 2)
            c = int(m + n, 2)
            temp0 = [str(a), str(b), str(c)]
            temp = [a, b, c]
            ids[temp[0], temp[1], temp[2]] = ''.join(temp0)
            cspi8 = 1 / (2 * np.sqrt(2))
            cos2pi8 = (2 + np.sqrt(2)) / 4
            sin2pi8 = (2 - np.sqrt(2)) / 4
            if m == j and n == l:
                if n == '0':
                    if i == k:
                        p[temp[0], temp[1], temp[2]] = (1 - parameter2 * (cos2pi8 - sin2pi8)) / 16
                    if i != k:
                        p[temp[0], temp[1], temp[2]] = (1 + parameter2 * (cos2pi8 - sin2pi8)) / 16
                if n == '1':
                    if m == '0':
                        if i == k:
                            p[temp[0], temp[1], temp[2]] = 1 / 16 - cspi8 * parameter2 / 8
                        if i != k:
                            p[temp[0], temp[1], temp[2]] = 1 / 16 + cspi8 * parameter2 / 8
                    if m == '1':
                        if i == k:
                            p[temp[0], temp[1], temp[2]] = 1 / 16 + cspi8 * parameter2 / 8
                        if i != k:
                            p[temp[0], temp[1], temp[2]] = 1 / 16 - cspi8 * parameter2 / 8
        p = p.flatten()
        ids = ids.flatten()

    if name == "Renou-visibility":
        """ Info: If param_c >~ 0.886 or <~0.464, there is no classical 3-local model."""
        """ In terms of c**2: above 0.785 or below 0.215 no classical 3-local model."""
        c = parameter1
        v = parameter2
        p = np.array([
            -(-1 + v) ** 3 / 64., -((-1 + v) * (1 + v) ** 2) / 64., ((-1 + v) ** 2 * (1 + v)) / 64.,
            ((-1 + v) ** 2 * (1 + v)) / 64., -((-1 + v) * (1 + v) ** 2) / 64., -((-1 + v) * (1 + v) ** 2) / 64.,
            ((1 + v) * (1 + (-2 + 4 * c ** 2) * v + v ** 2)) / 64.,
            ((1 + v) * (1 + (2 - 4 * c ** 2) * v + v ** 2)) / 64., ((-1 + v) ** 2 * (1 + v)) / 64.,
            ((1 + v) * (1 + (2 - 4 * c ** 2) * v + v ** 2)) / 64.,
            (1 + (1 - 2 * c ** 2) ** 2 * v - (1 - 2 * c ** 2) ** 2 * v ** 2 - v ** 3) / 64.,
            (1 + v - 4 * c ** 4 * v + (-1 + 4 * c ** 4) * v ** 2 - v ** 3) / 64., ((-1 + v) ** 2 * (1 + v)) / 64.,
            ((1 + v) * (1 + (-2 + 4 * c ** 2) * v + v ** 2)) / 64.,
            (1 + (-3 + 8 * c ** 2 - 4 * c ** 4) * v + (3 - 8 * c ** 2 + 4 * c ** 4) * v ** 2 - v ** 3) / 64.,
            (1 + (1 - 2 * c ** 2) ** 2 * v - (1 - 2 * c ** 2) ** 2 * v ** 2 - v ** 3) / 64.,
            -((-1 + v) * (1 + v) ** 2) / 64., -((-1 + v) * (1 + v) ** 2) / 64.,
            ((1 + v) * (1 + (2 - 4 * c ** 2) * v + v ** 2)) / 64.,
            ((1 + v) * (1 + (-2 + 4 * c ** 2) * v + v ** 2)) / 64.,
            -((-1 + v) * (1 + v) ** 2) / 64., -(-1 + v) ** 3 / 64., ((-1 + v) ** 2 * (1 + v)) / 64.,
            ((-1 + v) ** 2 * (1 + v)) / 64., ((1 + v) * (1 + (-2 + 4 * c ** 2) * v + v ** 2)) / 64.,
            ((-1 + v) ** 2 * (1 + v)) / 64.,
            (1 + (1 - 2 * c ** 2) ** 2 * v - (1 - 2 * c ** 2) ** 2 * v ** 2 - v ** 3) / 64.,
            (1 + (-3 + 8 * c ** 2 - 4 * c ** 4) * v + (3 - 8 * c ** 2 + 4 * c ** 4) * v ** 2 - v ** 3) / 64.,
            ((1 + v) * (1 + (2 - 4 * c ** 2) * v + v ** 2)) / 64., ((-1 + v) ** 2 * (1 + v)) / 64.,
            (1 + v - 4 * c ** 4 * v + (-1 + 4 * c ** 4) * v ** 2 - v ** 3) / 64.,
            (1 + (1 - 2 * c ** 2) ** 2 * v - (1 - 2 * c ** 2) ** 2 * v ** 2 - v ** 3) / 64.,
            ((-1 + v) ** 2 * (1 + v)) / 64., ((1 + v) * (1 + (-2 + 4 * c ** 2) * v + v ** 2)) / 64.,
            (1 + (1 - 2 * c ** 2) ** 2 * v - (1 - 2 * c ** 2) ** 2 * v ** 2 - v ** 3) / 64.,
            (1 + (-3 + 8 * c ** 2 - 4 * c ** 4) * v + (3 - 8 * c ** 2 + 4 * c ** 4) * v ** 2 - v ** 3) / 64.,
            ((1 + v) * (1 + (2 - 4 * c ** 2) * v + v ** 2)) / 64., ((-1 + v) ** 2 * (1 + v)) / 64.,
            (1 + (1 - 2 * c ** 2) ** 2 * v - (1 - 2 * c ** 2) ** 2 * v ** 2 - v ** 3) / 64.,
            (1 + v - 4 * c ** 4 * v + (-1 + 4 * c ** 4) * v ** 2 - v ** 3) / 64.,
            (1 + (1 - 2 * c ** 2) ** 2 * v - (1 - 2 * c ** 2) ** 2 * v ** 2 - v ** 3) / 64.,
            (1 + (1 - 2 * c ** 2) ** 2 * v - (1 - 2 * c ** 2) ** 2 * v ** 2 - v ** 3) / 64., (
                        1 + 3 * (1 - 2 * c ** 2) ** 2 * v + 3 * (1 - 2 * c ** 2) ** 2 * v ** 2 + (
                            1 + 16 * c ** 3 * np.sqrt(1 - c ** 2) - 16 * c ** 5 * np.sqrt(1 - c ** 2)) * v ** 3) / 64.,
            (1 - (1 - 2 * c ** 2) ** 2 * v - (1 - 2 * c ** 2) ** 2 * v ** 2 + (
                        1 - 16 * c ** 3 * np.sqrt(1 - c ** 2) + 16 * c ** 5 * np.sqrt(1 - c ** 2)) * v ** 3) / 64.,
            (1 + v - 4 * c ** 4 * v + (-1 + 4 * c ** 4) * v ** 2 - v ** 3) / 64.,
            (1 + (-3 + 8 * c ** 2 - 4 * c ** 4) * v + (3 - 8 * c ** 2 + 4 * c ** 4) * v ** 2 - v ** 3) / 64., (
                        1 - (1 - 2 * c ** 2) ** 2 * v - (1 - 2 * c ** 2) ** 2 * v ** 2 + (
                            1 - 16 * c ** 3 * np.sqrt(1 - c ** 2) + 16 * c ** 5 * np.sqrt(1 - c ** 2)) * v ** 3) / 64.,
            (1 - (1 - 2 * c ** 2) ** 2 * v - (1 - 2 * c ** 2) ** 2 * v ** 2 + (
                        1 + 16 * c ** 3 * np.sqrt(1 - c ** 2) - 16 * c ** 5 * np.sqrt(1 - c ** 2)) * v ** 3) / 64.,
            ((-1 + v) ** 2 * (1 + v)) / 64., ((1 + v) * (1 + (2 - 4 * c ** 2) * v + v ** 2)) / 64.,
            (1 + v - 4 * c ** 4 * v + (-1 + 4 * c ** 4) * v ** 2 - v ** 3) / 64.,
            (1 + (1 - 2 * c ** 2) ** 2 * v - (1 - 2 * c ** 2) ** 2 * v ** 2 - v ** 3) / 64.,
            ((1 + v) * (1 + (-2 + 4 * c ** 2) * v + v ** 2)) / 64., ((-1 + v) ** 2 * (1 + v)) / 64.,
            (1 + (-3 + 8 * c ** 2 - 4 * c ** 4) * v + (3 - 8 * c ** 2 + 4 * c ** 4) * v ** 2 - v ** 3) / 64.,
            (1 + (1 - 2 * c ** 2) ** 2 * v - (1 - 2 * c ** 2) ** 2 * v ** 2 - v ** 3) / 64.,
            (1 + (-3 + 8 * c ** 2 - 4 * c ** 4) * v + (3 - 8 * c ** 2 + 4 * c ** 4) * v ** 2 - v ** 3) / 64.,
            (1 + v - 4 * c ** 4 * v + (-1 + 4 * c ** 4) * v ** 2 - v ** 3) / 64., (
                        1 - (1 - 2 * c ** 2) ** 2 * v - (1 - 2 * c ** 2) ** 2 * v ** 2 + (
                            1 - 16 * c ** 3 * np.sqrt(1 - c ** 2) + 16 * c ** 5 * np.sqrt(1 - c ** 2)) * v ** 3) / 64.,
            (1 - (1 - 2 * c ** 2) ** 2 * v - (1 - 2 * c ** 2) ** 2 * v ** 2 + (
                        1 + 16 * c ** 3 * np.sqrt(1 - c ** 2) - 16 * c ** 5 * np.sqrt(1 - c ** 2)) * v ** 3) / 64.,
            (1 + (1 - 2 * c ** 2) ** 2 * v - (1 - 2 * c ** 2) ** 2 * v ** 2 - v ** 3) / 64.,
            (1 + (1 - 2 * c ** 2) ** 2 * v - (1 - 2 * c ** 2) ** 2 * v ** 2 - v ** 3) / 64., (
                        1 - (1 - 2 * c ** 2) ** 2 * v - (1 - 2 * c ** 2) ** 2 * v ** 2 + (
                            1 + 16 * c ** 3 * np.sqrt(1 - c ** 2) - 16 * c ** 5 * np.sqrt(1 - c ** 2)) * v ** 3) / 64.,
            (1 + 3 * (1 - 2 * c ** 2) ** 2 * v + 3 * (1 - 2 * c ** 2) ** 2 * v ** 2 + (
                        1 - 16 * c ** 3 * np.sqrt(1 - c ** 2) + 16 * c ** 5 * np.sqrt(1 - c ** 2)) * v ** 3) / 64.
        ])
        ids = np.array([
            "000", "001", "002", "003", "010", "011", "012", "013", "020", "021", \
            "022", "023", "030", "031", "032", "033", "100", "101", "102", "103", \
            "110", "111", "112", "113", "120", "121", "122", "123", "130", "131", \
            "132", "133", "200", "201", "202", "203", "210", "211", "212", "213", \
            "220", "221", "222", "223", "230", "231", "232", "233", "300", "301", \
            "302", "303", "310", "311", "312", "313", "320", "321", "322", "323", \
            "330", "331", "332", "333"
        ])

    if name == "Renou-localnoise":
        """ Info: If param_c >~ 0.886 or <~0.464, there is no classical 3-local model."""
        """ In terms of c**2: above 0.785 or below 0.215 no classical 3-local model."""
        param_c = parameter1
        param_s = np.np.sqrt(1 - param_c ** 2)

        # the si and ci functions
        param2_c = {'2': param_c, '3': param_s}
        param2_s = {'2': param_s, '3': -1 * param_c}

        # First create noiseless Salman distribution.
        ids = np.zeros((4, 4, 4)).astype(str)
        p = np.zeros((4, 4, 4))
        for a, b, c in product('0123', repeat=3):
            temp0 = [a, b, c]
            temp = [int(item) for item in temp0]
            ids[temp[0], temp[1], temp[2]] = ''.join(temp0)

            # p(12vi) et al.
            if (a == '0' and b == '1' and c == '2') or (a == '1' and b == '0' and c == '3'):
                p[temp[0], temp[1], temp[2]] = 1 / 8 * param_c ** 2
            elif (c == '0' and a == '1' and b == '2') or (c == '1' and a == '0' and b == '3'):
                p[temp[0], temp[1], temp[2]] = 1 / 8 * param_c ** 2
            elif (b == '0' and c == '1' and a == '2') or (b == '1' and c == '0' and a == '3'):
                p[temp[0], temp[1], temp[2]] = 1 / 8 * param_c ** 2

            elif (a == '0' and b == '1' and c == '3') or (a == '1' and b == '0' and c == '2'):
                p[temp[0], temp[1], temp[2]] = 1 / 8 * param_s ** 2
            elif (c == '0' and a == '1' and b == '3') or (c == '1' and a == '0' and b == '2'):
                p[temp[0], temp[1], temp[2]] = 1 / 8 * param_s ** 2
            elif (b == '0' and c == '1' and a == '3') or (b == '1' and c == '0' and a == '2'):
                p[temp[0], temp[1], temp[2]] = 1 / 8 * param_s ** 2

            # p(vi vj vk) et al.
            elif a in '23' and b in '23' and c in '23':
                p[temp[0], temp[1], temp[2]] = 1 / 8 * (
                            param2_c[a] * param2_c[b] * param2_c[c] + param2_s[a] * param2_s[b] * param2_s[c]) ** 2
            else:
                p[temp[0], temp[1], temp[2]] = 0

        # Let's add local noise.
        new_values = np.zeros_like(p)
        for a, b, c in product('0123', repeat=3):
            temp0 = [a, b, c]
            temp = [int(item) for item in temp0]
            new_values[temp[0], temp[1], temp[2]] = (
                    parameter2 ** 3 * p[temp[0], temp[1], temp[2]] +
                    parameter2 ** 2 * (1 - parameter2) * 1 / 4 * (
                                np.sum(p, axis=2)[temp[0], temp[1]] + np.sum(p, axis=0)[temp[1], temp[2]] +
                                np.sum(p, axis=1)[temp[0], temp[2]]) +
                    parameter2 * (1 - parameter2) ** 2 * 1 / 16 * (
                                np.sum(p, axis=(1, 2))[temp[0]] + np.sum(p, axis=(0, 2))[temp[1]] +
                                np.sum(p, axis=(0, 1))[temp[2]]) +
                    (1 - parameter2) ** 3 * 1 / 64
            )
        p = new_values.flatten()
        ids = ids.flatten()

    if name == "elegant-visibility":
        """ Recreating the elegant distribution with visibility v (parameter2) in each singlet. """
        ids = np.zeros((4, 4, 4)).astype(str)
        for a, b, c in product('0123', repeat=3):
            temp0 = [a, b, c]
            temp = [int(item) for item in temp0]
            ids[temp[0], temp[1], temp[2]] = ''.join(temp0)
        ids = ids.flatten()
        p = np.array([1 / 256 * (4 + 9 * parameter2 + 9 * parameter2 ** 2 + 3 * parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 - 3 * parameter2 + 3 * parameter2 ** 2 + parameter2 ** 3),
                      1 / 256 * (4 - 3 * parameter2 + 3 * parameter2 ** 2 + parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 - 3 * parameter2 + 3 * parameter2 ** 2 + parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 - 3 * parameter2 + 3 * parameter2 ** 2 + parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 - 3 * parameter2 + 3 * parameter2 ** 2 + parameter2 ** 3),
                      1 / 256 * (4 - 3 * parameter2 + 3 * parameter2 ** 2 + parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 - 3 * parameter2 + 3 * parameter2 ** 2 + parameter2 ** 3),
                      1 / 256 * (4 - 3 * parameter2 + 3 * parameter2 ** 2 + parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 + 9 * parameter2 + 9 * parameter2 ** 2 + 3 * parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 - 3 * parameter2 + 3 * parameter2 ** 2 + parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 - 3 * parameter2 + 3 * parameter2 ** 2 + parameter2 ** 3),
                      1 / 256 * (4 - 3 * parameter2 + 3 * parameter2 ** 2 + parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 - 3 * parameter2 + 3 * parameter2 ** 2 + parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 - 3 * parameter2 + 3 * parameter2 ** 2 + parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 - 3 * parameter2 + 3 * parameter2 ** 2 + parameter2 ** 3),
                      1 / 256 * (4 - 3 * parameter2 + 3 * parameter2 ** 2 + parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 - 3 * parameter2 + 3 * parameter2 ** 2 + parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 + 9 * parameter2 + 9 * parameter2 ** 2 + 3 * parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 - 3 * parameter2 + 3 * parameter2 ** 2 + parameter2 ** 3),
                      1 / 256 * (4 - 3 * parameter2 + 3 * parameter2 ** 2 + parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 - 3 * parameter2 + 3 * parameter2 ** 2 + parameter2 ** 3),
                      1 / 256 * (4 - 3 * parameter2 + 3 * parameter2 ** 2 + parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 - 3 * parameter2 + 3 * parameter2 ** 2 + parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 - 3 * parameter2 + 3 * parameter2 ** 2 + parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 - 3 * parameter2 + 3 * parameter2 ** 2 + parameter2 ** 3),
                      1 / 256 * (4 - 3 * parameter2 + 3 * parameter2 ** 2 + parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 + parameter2 - 3 * parameter2 ** 2 - parameter2 ** 3),
                      1 / 256 * (4 + 9 * parameter2 + 9 * parameter2 ** 2 + 3 * parameter2 ** 3)])

    if name == "elegant-localnoise":
        """ Recreating the elegant distribution, with each detector having 1-v (1-parameter2) chance of outputting a uniformly random output, and v chance of working properly. """
        ids = np.zeros((4, 4, 4)).astype(str)
        p = np.zeros((4, 4, 4))
        for a, b, c in product('0123', repeat=3):
            temp0 = [a, b, c]
            temp = [int(item) for item in temp0]
            ids[temp[0], temp[1], temp[2]] = ''.join(temp0)
            if (a == b) and (b == c):
                p[temp[0], temp[1], temp[2]] = 25 / 256
            elif (a == b and b != c) or (b == c and c != a) or (c == a and a != b):
                p[temp[0], temp[1], temp[2]] = 1 / 256
            else:
                p[temp[0], temp[1], temp[2]] = 5 / 256

        # Let's add local noise.
        new_values = np.zeros_like(p)
        for a, b, c in product('0123', repeat=3):
            temp0 = [a, b, c]
            temp = [int(item) for item in temp0]
            new_values[temp[0], temp[1], temp[2]] = (
                    parameter2 ** 3 * p[temp[0], temp[1], temp[2]] +
                    parameter2 ** 2 * (1 - parameter2) * 1 / 4 * (
                                np.sum(p, axis=2)[temp[0], temp[1]] + np.sum(p, axis=0)[temp[1], temp[2]] +
                                np.sum(p, axis=1)[temp[0], temp[2]]) +
                    parameter2 * (1 - parameter2) ** 2 * 1 / 16 * (
                                np.sum(p, axis=(1, 2))[temp[0]] + np.sum(p, axis=(0, 2))[temp[1]] +
                                np.sum(p, axis=(0, 1))[temp[2]]) +
                    (1 - parameter2) ** 3 * 1 / 64
            )

        p = new_values.flatten()
        ids = ids.flatten()
    assert (np.abs(np.sum(p) - 1.0) < (1E-6)), "Improperly normalized p!"
    return np.reshape(p, [4, 4, 4])


def P_GHZ(a, b, c, x, y, z, v):
    def p_ideal(a, b, c, x, y, z):
        if a == b == c:
            return 1 / 2
        else:
            return 0.0

    # prob_array = np.ones([2, 2, 2]) * (1-v)/8
    # prob_array[0][0][0] += 0.5*v
    # prob_array[1][1][1] += 0.5*v
    return v * p_ideal(a, b, c, x, y, z) + (1 - v) / 8


def P_W(a, b, c, x, y, z, v):
    def p_ideal(a, b, c, x, y, z):
        if a + b + c == 1:
            return 1 / 3
        else:
            return 0.0

    return v * p_ideal(a, b, c, x, y, z) + (1 - v) / 8


def P_Mermin(a, b, c, x, y, z, v):
    def p_ideal(a, b, c, x, y, z):
        if (x + y + z) % 2 == 0:
            return 1 / 8
        elif x + y + z == 1:
            return (1 + (-1) ** (a + b + c)) / 8
        elif x + y + z == 3:
            return (1 - (-1) ** (a + b + c)) / 8
        else:
            raise Exception('x, y or z are not 0/1')

    return v * p_ideal(a, b, c, x, y, z) + (1 - v) / 8


def P_Salman(a, b, c, x, y, z, u2):
    prob_array = target_distribution_gen("Renou-visibility", np.sqrt(u2), 1.0).astype(float)
    return prob_array[a, b, c]
